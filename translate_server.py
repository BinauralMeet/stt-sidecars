#!/usr/bin/env python3
"""Translation sidecar for bm/bmMediasoupServer (src/DataServer/translation.ts).

Wire contract (see bm/docs stt-translation#hostwork):

    POST /translate   {"texts": [...], "src": "ja", "dsts": ["en"]}
        -> {"en": ["..."], ...}   -- one array per requested dst, same length/order as `texts`

A `dst` (or `src`) this sidecar has no model for is simply left out of the response; the caller
already treats a missing key as "no translation available" and keeps the original-language
subtitle, so partial coverage degrades gracefully rather than failing the whole request.

Models are pre-converted CTranslate2 checkpoints (see README.md `#models` for how to regenerate
them) -- this process never touches the network at request time.
"""
import logging
import os

import ctranslate2
import sentencepiece as spm
from flask import Flask, request, jsonify
from waitress import serve

MODELS_DIR = os.environ.get('TRANSLATE_MODELS_DIR', '/opt/stt-sidecars/models')
CPU_THREADS = int(os.environ.get('TRANSLATE_THREADS', '4'))
PORT = int(os.environ.get('TRANSLATE_PORT', '8191'))

#  (src, dst) -> model directory name under MODELS_DIR. Add a line here + convert the model
#  (README.md `#models`) to support another language pair.
PAIRS = {
    ('ja', 'en'): 'fugumt-ja-en',
    ('en', 'ja'): 'fugumt-en-ja',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('translate')

app = Flask(__name__)


class Pair:
    def __init__(self, model_dir: str):
        self.sp_src = spm.SentencePieceProcessor(model_file=f'{model_dir}/source.spm')
        self.sp_tgt = spm.SentencePieceProcessor(model_file=f'{model_dir}/target.spm')
        self.translator = ctranslate2.Translator(
            model_dir, device='cpu', inter_threads=1, intra_threads=CPU_THREADS)

    def translate(self, texts: list[str]) -> list[str]:
        batch = [self.sp_src.encode(t, out_type=str) + ['</s>'] for t in texts]
        results = self.translator.translate_batch(batch, beam_size=4, max_decoding_length=256)

        return [self.sp_tgt.decode(r.hypotheses[0]) for r in results]


pairs: dict[tuple[str, str], Pair] = {}
for (src, dst), name in PAIRS.items():
    model_dir = os.path.join(MODELS_DIR, name)
    log.info('loading %s -> %s from %s ...', src, dst, model_dir)
    pairs[(src, dst)] = Pair(model_dir)
log.info('%d language pair(s) loaded, listening on 127.0.0.1:%d', len(pairs), PORT)


@app.post('/translate')
def translate():
    body = request.get_json(silent=True) or {}
    texts = body.get('texts')
    src = (body.get('src') or '').strip().lower()
    dsts = body.get('dsts') or []
    if not isinstance(texts, list) or not texts or not src or not isinstance(dsts, list):
        return jsonify(error='texts, src and dsts are required'), 400

    out = {}
    for dst in dsts:
        dst = (dst or '').strip().lower()
        pair = pairs.get((src, dst))
        if pair is None:
            continue  # unsupported pair -- omitted, not an error (see module docstring)
        try:
            out[dst] = pair.translate(texts)
        except Exception as e:  # noqa: BLE001 -- one bad pair must not sink the others
            log.warning('translate %s->%s failed: %s', src, dst, e)

    return jsonify(out)


@app.get('/health')
def health():
    return jsonify(status='ok', pairs=[f'{s}->{d}' for s, d in pairs])


if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=PORT, threads=4)
