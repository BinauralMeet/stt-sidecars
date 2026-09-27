#!/usr/bin/env python3
"""GPU recognition (and translation) sidecar for bm/bmMediasoupServer.

Same wire contract as cpu_whisper_server.py, so bmMediasoupServer only needs another entry in
`stt.backends` (see the bm workspace doc, `stt-translation#stt-backend`):

    POST /asr?lang=<hint>[&prompt=<terms>]   body: 16kHz mono 16-bit WAV
        -> {"text": "...", "lang": "ja"}

Why this exists alongside SenseVoice, which is already on the GPU: SenseVoice-small is fast but
has no language model behind it and no way to bias decoding, and it mangles loanwords -- the
katakana vocabulary of a technical meeting is exactly where it fails. Whisper large-v3-turbo has
both, and `initial_prompt` lets the terms a room actually uses be handed to it up front.

`WHISPER_PROMPT` is that vocabulary: a short list of the words this deployment keeps getting
wrong, and `WHISPER_PROMPT_LANG` the language it is written in -- a prompt is decoder context,
so a Japanese word list handed to an English utterance drags the transcript into Japanese.
Keep it under ~200 characters -- it is prepended to the decoder's context, so a long list
costs latency and starts to steer the transcript itself.

The same process also answers the translation contract, for the language pairs the CPU sidecar
has no model for (it carries ja<->en only):

    POST /translate   {"texts": [...], "src": "ja", "dsts": ["zh", "ko"]}
        -> {"zh": ["..."], ...}   -- one array per requested dst, unsupported *or low-confidence*
                                     ones left out (see MULTI_MIN_SCORE below)

One multilingual model covers every pair. It lives here rather than in its own service because
it is small next to the recognizer and shares the same GPU, mode and tunnel; when the GPU is
taken away both stop together, and bmMediasoupServer falls back to the CPU translator for the
pairs that one does know.

`MULTI_MODEL`/`MULTI_KIND` choose it. The default is M2M-100 (MIT): NLLB-200 translates better
but is CC-BY-NC, and this deployment is public, so the licence would follow the whole service
around. Set MULTI_KIND=nllb if that is acceptable where you are running it.

`MULTI_MIN_SCORE` drops a hypothesis whose mean per-token log-prob falls below it instead of
returning it as a translation -- these models occasionally decode a fluent-looking but made-up
sentence for a short/garbled/OOD source, and the wire format has no way to mark "translated" vs
"the model gave up" except by omitting the dst (see `_filter_confident`).
"""
import io
import logging
import os
import sys


def _add_cuda_dll_dirs():
    """Windows does not search site-packages for DLLs, so CTranslate2 fails with
    "Library cublas64_12.dll is not found" even when the nvidia-*-cu12 wheels that contain it are
    installed. Both mechanisms are needed and neither is enough on its own: add_dll_directory()
    only covers loads that opt into the safe search order, and CTranslate2 loads cuBLAS by plain
    name, which searches PATH. A no-op everywhere else, where the loader follows RPATH."""
    if not hasattr(os, 'add_dll_directory'):
        return
    import site
    found = []
    for base in site.getsitepackages():
        root = os.path.join(base, 'nvidia')
        if not os.path.isdir(root):
            continue
        for pkg in sorted(os.listdir(root)):
            for sub in ('bin', 'lib'):
                path = os.path.join(root, pkg, sub)
                if os.path.isdir(path):
                    os.add_dll_directory(path)
                    found.append(path)
    if found:
        os.environ['PATH'] = os.pathsep.join(found) + os.pathsep + os.environ.get('PATH', '')


_add_cuda_dll_dirs()

from flask import Flask, request, jsonify

from sidecar_auth import install as install_auth
from faster_whisper import WhisperModel
from waitress import serve

MODEL = os.environ.get('WHISPER_MODEL', 'large-v3-turbo')
DEVICE = os.environ.get('WHISPER_DEVICE', 'cuda')
COMPUTE = os.environ.get('WHISPER_COMPUTE', 'float16')
PORT = int(os.environ.get('WHISPER_PORT', '8192'))
HOST = os.environ.get('WHISPER_HOST', '0.0.0.0')
PROMPT = os.environ.get('WHISPER_PROMPT', '')
#  A prompt is decoder context, so a Japanese word list pulls an English utterance towards
#  Japanese: "ask not" came back as "アースクリーン". Naming the prompt's own language keeps it
#  out of the way of every other one; empty means apply it always, as before.
PROMPT_LANG = os.environ.get('WHISPER_PROMPT_LANG', '')
#  Several people talk at once in a meeting, and every one of their utterances lands here. One
#  model instance answers one request at a time, so without replicas the second speaker simply
#  waits for the first -- which is the latency people actually notice.
WORKERS = int(os.environ.get('WHISPER_WORKERS', '2'))
BEAM = int(os.environ.get('WHISPER_BEAM', '5'))
MULTI_MODEL = os.environ.get('MULTI_MODEL', '')      #  CTranslate2 dir; empty = no /translate
MULTI_KIND = os.environ.get('MULTI_KIND', 'm2m100')  #  'm2m100' or 'nllb'
MULTI_TOKENIZER = os.environ.get('MULTI_TOKENIZER', 'facebook/m2m100_418M')
#  Below this (mean log-prob per output token, always <= 0), a hypothesis is dropped instead of
#  being returned as if it were a real translation -- these seq2seq models occasionally decode a
#  short, fluent-looking, entirely made-up sentence on a garbled/short/OOD source (a known failure
#  mode, distinct from a low-quality-but-genuine translation) and there is no way to tell the two
#  apart from the text alone. Unset (empty string) disables the check -- every hypothesis is kept
#  as before. -1.2 is a placeholder, not a measured value (bm workspace doc `stt-translation#todo`
#  / CHANGELOG 2026-09-27): nobody has logged real (text, score) pairs from this deployment yet to
#  pick a real cutoff, so this wants tuning against actual meetings before trusting it blindly.
MULTI_MIN_SCORE = os.environ.get('MULTI_MIN_SCORE', '-1.2')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('gpu-whisper')

app = Flask(__name__)
if install_auth(app):
    log.info('bearer token required (STT_API_KEY is set)')

log.info('loading faster-whisper model=%s device=%s compute=%s ...', MODEL, DEVICE, COMPUTE)
model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE, num_workers=WORKERS)
log.info('model loaded, listening on %s:%d (workers: %d, beam: %d, prompt: %s)',
         HOST, PORT, WORKERS, BEAM, PROMPT or '(none)')


@app.post('/asr')
def asr():
    lang_hint = (request.args.get('lang') or '').strip().lower() or None
    prompt = request.args.get('prompt')
    if not prompt and PROMPT and (not PROMPT_LANG or lang_hint == PROMPT_LANG):
        prompt = PROMPT
    audio = request.get_data()
    if not audio:
        return jsonify(text='', lang=lang_hint or ''), 400

    try:
        segments, info = model.transcribe(
            io.BytesIO(audio),
            language=lang_hint,
            initial_prompt=prompt,
            vad_filter=False,  # BM already did VAD upstream; avoid double-guessing endpoints
            #  Beam search is worth its cost here: this runs on a GPU, and the wrong branch is
            #  precisely how a loanword turns into a different word that also fits the audio.
            beam_size=BEAM,
            condition_on_previous_text=False,  # each utterance stands alone; no drift across them
        )
        text = ''.join(seg.text for seg in segments).strip()
        lang = lang_hint or info.language or ''
    except Exception as e:  # noqa: BLE001 -- a bad request must not take the service down
        log.warning('transcribe failed: %s', e)
        return jsonify(text='', lang=lang_hint or ''), 500

    return jsonify(text=text, lang=lang)


#  ---------------------------------------------------------------- translation

#  NLLB names languages by script ('jpn_Jpan'), M2M-100 by plain code ('ja'). Only the languages
#  this deployment actually offers are listed: an unknown one is left out of the answer, which
#  the caller already treats as "no translation available".
NLLB_CODES = {'ja': 'jpn_Jpan', 'en': 'eng_Latn', 'zh': 'zho_Hans', 'ko': 'kor_Hang'}

_min_score = float(MULTI_MIN_SCORE) if MULTI_MIN_SCORE else None


def _mean_score(result):
    """CTranslate2's score is the hypothesis's cumulative log-prob; a longer sentence sums more
    (negative) terms, so raw scores are not comparable across outputs of different length. Divide
    by the token count actually scored (the forced prefix contributes no informative log-prob)."""
    if not result.scores:
        return None
    tokens = max(len(result.hypotheses[0]) - 1, 1)  # -1: the forced target-language token

    return result.scores[0] / tokens


def _filter_confident(decoded):
    """`decoded` is [(text, score), ...], one per input text, same order. Only ever called with
    exactly one item in this deployment (bmMediasoupServer batches one utterance at a time) --
    dropping a low-confidence item would otherwise desync the positional contract the wire format
    promises (module docstring), so a genuine multi-item batch skips the check rather than risk
    that. Returns the list to send back, or None to omit the dst entirely (same wire meaning as
    "unsupported pair": the caller falls back to the original-language subtitle)."""
    if _min_score is None or len(decoded) != 1:
        return [text for text, _score in decoded]
    text, score = decoded[0]
    #  Logged unconditionally (not just on drop) while MULTI_MIN_SCORE is still an unmeasured
    #  guess (bm workspace CHANGELOG 2026-09-27): there is no other way to see what real (text,
    #  score) pairs look like in this deployment, and a threshold picked without seeing the
    #  passing side too is just as much a guess as -1.2 itself. Turn back to logging only the
    #  dropped ones once the threshold is trusted -- this is one line per translated utterance.
    kept = not (score is not None and score < _min_score)
    log.info('translate: score=%s kept=%s text=%r', f'{score:.2f}' if score is not None else 'n/a',
             kept, text)
    if not kept:

        return None

    return [text]

translator = None
multi_tokenizer = None
if MULTI_MODEL:
    import ctranslate2
    from transformers import AutoTokenizer
    log.info('loading %s translator from %s ...', MULTI_KIND, MULTI_MODEL)
    translator = ctranslate2.Translator(MULTI_MODEL, device=DEVICE)
    multi_tokenizer = AutoTokenizer.from_pretrained(MULTI_TOKENIZER)
    log.info('translator loaded')


def _lang_token(lang):
    if MULTI_KIND == 'nllb':
        return NLLB_CODES.get(lang)

    return f'__{lang}__' if lang in NLLB_CODES else None


@app.post('/translate')
def translate():
    if translator is None:
        return jsonify(error='no translation model loaded'), 503
    body = request.get_json(silent=True) or {}
    texts = body.get('texts')
    src = (body.get('src') or '').strip().lower()
    dsts = body.get('dsts') or []
    if not isinstance(texts, list) or not texts or not src or not isinstance(dsts, list):
        return jsonify(error='texts, src and dsts are required'), 400
    src_token = _lang_token(src)
    if src_token is None:
        return jsonify({}), 200

    if MULTI_KIND == 'nllb':
        multi_tokenizer.src_lang = src_token
    else:
        multi_tokenizer.src_lang = src
    encoded = [multi_tokenizer.convert_ids_to_tokens(multi_tokenizer.encode(t)) for t in texts]

    out = {}
    for dst in dsts:
        dst = (dst or '').strip().lower()
        token = _lang_token(dst)
        if token is None or dst == src:
            continue
        try:
            results = translator.translate_batch(
                encoded, target_prefix=[[token]] * len(encoded), beam_size=4,
                max_decoding_length=256, return_scores=True)
            #  The forced language token is part of the output; drop it before detokenizing.
            decoded = [
                (multi_tokenizer.decode(
                    multi_tokenizer.convert_tokens_to_ids(r.hypotheses[0][1:]),
                    skip_special_tokens=True),
                 _mean_score(r))
                for r in results
            ]
            kept = _filter_confident(decoded)
            if kept is not None:
                out[dst] = kept
        except Exception as e:  # noqa: BLE001 -- one bad pair must not sink the others
            log.warning('translate %s->%s failed: %s', src, dst, e)

    return jsonify(out)


@app.get('/health')
def health():
    return jsonify(status='ok', model=MODEL, device=DEVICE,
                   prompt=f'{PROMPT_LANG or "all"}' if PROMPT else None,
                   translator=MULTI_KIND if translator is not None else None)


if __name__ == '__main__':
    #  Enough HTTP threads to keep every model replica fed, plus room for /health.
    serve(app, listen=f'{HOST}:{PORT}', threads=WORKERS + 2)
