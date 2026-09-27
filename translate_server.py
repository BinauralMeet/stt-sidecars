#!/usr/bin/env python3
"""Translation sidecar for bm/bmMediasoupServer (src/DataServer/translation.ts).

Wire contract (see bm/docs stt-translation#hostwork):

    POST /translate   {"texts": [...], "src": "ja", "dsts": ["en"]}
        -> {"en": ["..."], ...}   -- one array per requested dst, same length/order as `texts`

A `dst` (or `src`) this sidecar has no model for is simply left out of the response; the caller
already treats a missing key as "no translation available" and keeps the original-language
subtitle, so partial coverage degrades gracefully rather than failing the whole request. A
low-confidence hypothesis (`TRANSLATE_MIN_SCORE`) is left out the same way -- these models
occasionally decode a fluent-looking but made-up sentence for a short/garbled/OOD source, and the
wire format has no way to mark "translated" vs "the model gave up" except by omitting the dst.

Models are pre-converted CTranslate2 checkpoints (see README.md `#models` for how to regenerate
them) -- this process never touches the network at request time.
"""
import logging
import os
import zlib

import ctranslate2
import sentencepiece as spm
from flask import Flask, request, jsonify

from sidecar_auth import install as install_auth
from waitress import serve

MODELS_DIR = os.environ.get('TRANSLATE_MODELS_DIR', '/opt/stt-sidecars/models')
CPU_THREADS = int(os.environ.get('TRANSLATE_THREADS', '4'))
PORT = int(os.environ.get('TRANSLATE_PORT', '8191'))
#  Below this (mean log-prob per output token, always <= 0), a hypothesis is dropped instead of
#  being returned as if it were a real translation -- same reasoning and same wire-format caveat
#  (batch size 1 only) as gpu_whisper_server.py's MULTI_MIN_SCORE. -1.2 is an unmeasured
#  placeholder (bm workspace doc `stt-translation#todo` / CHANGELOG 2026-09-27); tune it against
#  real (text, score) pairs once some have been logged. Empty string disables the check.
TRANSLATE_MIN_SCORE = os.environ.get('TRANSLATE_MIN_SCORE', '-1.2')
_min_score = float(TRANSLATE_MIN_SCORE) if TRANSLATE_MIN_SCORE else None
#  Found live 2026-09-27 (bm workspace CHANGELOG same date, gpu_whisper_server.py's matching
#  constant): a translation can loop into repeating a short phrase on its own even when the
#  source text wasn't repetitive, and decoder confidence doesn't catch it (the model is just as
#  "sure" repeating something as saying it once). 2.4 matches faster-whisper's own default for
#  the same signal.
TRANSLATE_MAX_COMPRESSION_RATIO = os.environ.get('TRANSLATE_MAX_COMPRESSION_RATIO', '2.4')
_max_compression = float(TRANSLATE_MAX_COMPRESSION_RATIO) if TRANSLATE_MAX_COMPRESSION_RATIO else None


def _text_compression_ratio(text):
    data = text.encode('utf-8')
    if not data:
        return 1.0

    return len(data) / len(zlib.compress(data, 9))

#  (src, dst) -> model directory name under MODELS_DIR. Add a line here + convert the model
#  (README.md `#models`) to support another language pair.
PAIRS = {
    ('ja', 'en'): 'fugumt-ja-en',
    ('en', 'ja'): 'fugumt-en-ja',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('translate')

app = Flask(__name__)
if install_auth(app):
    log.info('bearer token required (STT_API_KEY is set)')


class Pair:
    def __init__(self, model_dir: str):
        self.sp_src = spm.SentencePieceProcessor(model_file=f'{model_dir}/source.spm')
        self.sp_tgt = spm.SentencePieceProcessor(model_file=f'{model_dir}/target.spm')
        self.translator = ctranslate2.Translator(
            model_dir, device='cpu', inter_threads=1, intra_threads=CPU_THREADS)

    def translate(self, texts: list[str]) -> list[str]|None:
        """Returns the decoded hypotheses, or None if the (single) one was below
        TRANSLATE_MIN_SCORE -- see the module-level comment for why, and why only for a
        single-text batch."""
        batch = [self.sp_src.encode(t, out_type=str) + ['</s>'] for t in texts]
        results = self.translator.translate_batch(
            batch, beam_size=4, max_decoding_length=256, return_scores=True)
        decoded = [(self.sp_tgt.decode(r.hypotheses[0]), self._mean_score(r)) for r in results]
        if (_min_score is None and _max_compression is None) or len(decoded) != 1:
            return [text for text, _score in decoded]
        text, score = decoded[0]
        low_confidence = score is not None and score < _min_score if _min_score is not None else False
        ratio = _text_compression_ratio(text)
        repetitive = _max_compression is not None and ratio > _max_compression
        #  Logged unconditionally (not just on drop) while these are still unmeasured guesses (bm
        #  workspace CHANGELOG 2026-09-27) -- see gpu_whisper_server.py's matching comment. Turn
        #  back to logging only the dropped ones once trusted.
        kept = not (low_confidence or repetitive)
        log.info('score=%s ratio=%.2f kept=%s text=%r',
                  f'{score:.2f}' if score is not None else 'n/a', ratio, kept, text)
        if not kept:

            return None

        return [text]

    @staticmethod
    def _mean_score(result):
        #  Cumulative log-prob isn't comparable across hypotheses of different length -- divide
        #  by the token count actually scored.
        if not result.scores:
            return None

        return result.scores[0] / max(len(result.hypotheses[0]), 1)


pairs: dict[tuple[str, str], Pair] = {}
for (src, dst), name in PAIRS.items():
    model_dir = os.path.join(MODELS_DIR, name)
    log.info('loading %s -> %s from %s ...', src, dst, model_dir)
    pairs[(src, dst)] = Pair(model_dir)
#  172.17.0.1 is docker0's host-side address, reachable from every sandbox container on this
#  host (`bm/docs stt-translation#hostwork`). This has no auth, so anything with container access
#  on this host can reach it -- accepted tradeoff so the dev sandboxes can reach translation.
LISTEN = os.environ.get('TRANSLATE_LISTEN', f'127.0.0.1:{PORT} 172.17.0.1:{PORT}')
log.info('%d language pair(s) loaded, listening on %s', len(pairs), LISTEN)


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
            translated = pair.translate(texts)
            if translated is not None:
                out[dst] = translated
        except Exception as e:  # noqa: BLE001 -- one bad pair must not sink the others
            log.warning('translate %s->%s failed: %s', src, dst, e)

    return jsonify(out)


@app.get('/health')
def health():
    return jsonify(status='ok', pairs=[f'{s}->{d}' for s, d in pairs])


if __name__ == '__main__':
    serve(app, listen=LISTEN, threads=4)
