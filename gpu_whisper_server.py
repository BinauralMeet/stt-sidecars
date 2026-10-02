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
import re
import sys
import zlib


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
#  Found live on 2026-09-27 (bm workspace CHANGELOG same date) testing deliberately garbled/
#  meaningless speech: avg_logprob alone missed a repetition-looped hallucination entirely
#  (score -0.07, effectively the model's best possible confidence, for "nya nya nya..." repeated
#  hundreds of times) -- a compressible, repetitive transcript is confident *because* it is easy
#  to keep predicting the same thing, not despite it. compression_ratio is faster-whisper's own
#  signal for exactly this (it already retries at higher temperature internally when a segment
#  exceeds this on the way to its final answer -- this is a second, coarser check on top, since a
#  segment can still come back over the line after every retry). 2.4 is faster-whisper's own
#  library default for the same reason; -1.0 for avg_logprob is a placeholder (`#todo`) still
#  being tuned against live (text, avg_logprob) pairs. Either check alone is not enough: a short,
#  genuinely-fine utterance can score similarly to a hallucinated one on avg_logprob alone.
WHISPER_MIN_LOGPROB = os.environ.get('WHISPER_MIN_LOGPROB', '-1.0')
WHISPER_MAX_COMPRESSION_RATIO = os.environ.get('WHISPER_MAX_COMPRESSION_RATIO', '2.4')
_asr_min_logprob = float(WHISPER_MIN_LOGPROB) if WHISPER_MIN_LOGPROB else None
_asr_max_compression = float(WHISPER_MAX_COMPRESSION_RATIO) if WHISPER_MAX_COMPRESSION_RATIO else None
#  Found live 2026-09-27, right after the two checks above: on unclear audio the decoder can
#  latch onto `initial_prompt` itself and read the prompt's own vocabulary back as the
#  transcript -- "ウィンドウ, シェア, コンテンツ, シェア, コンテンツ, スクリーン", every single
#  term a literal entry from WHISPER_PROMPT, comma-joined the same way the prompt is written.
#  Neither avg_logprob (-0.59, unremarkable) nor compression_ratio (1.41, several distinct words)
#  caught it -- a shuffled/repeated subset of a comma list isn't the same shape as one token
#  repeated. Detected by the same shape instead: split the transcript on commas/読点 and check how
#  much of it is literal prompt vocabulary. Real speech mentioning "ウィンドウ" in a sentence
#  looks nothing like this; a bare list of 2+ terms that are almost all prompt words does.
WHISPER_PROMPT_LEAK_RATIO = os.environ.get('WHISPER_PROMPT_LEAK_RATIO', '0.7')
_prompt_leak_ratio = float(WHISPER_PROMPT_LEAK_RATIO) if WHISPER_PROMPT_LEAK_RATIO else None


def _is_prompt_leak(text, prompt):
    if _prompt_leak_ratio is None or not prompt:
        return False
    prompt_terms = {t.strip() for t in re.split(r'[,、]', prompt) if t.strip()}
    if not prompt_terms:
        return False
    output_terms = [t.strip() for t in re.split(r'[,、]', text) if t.strip()]
    if len(output_terms) < 2:
        return False  # a single word, even a prompt word, is an entirely ordinary thing to say

    return sum(1 for t in output_terms if t in prompt_terms) / len(output_terms) >= _prompt_leak_ratio
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
#  Found live on 2026-09-27 (same session that found WHISPER_MAX_COMPRESSION_RATIO above): a
#  translation can loop into repeating a short phrase on its own, independent of whether the
#  source text was itself repetitive -- e.g. a perfectly ordinary Japanese sentence came back as
#  "No. No. No. No." Decoder confidence doesn't catch this either (same reason as the ASR case),
#  so the same compression-ratio trick applies to the *output* text here.
MULTI_MAX_COMPRESSION_RATIO = os.environ.get('MULTI_MAX_COMPRESSION_RATIO', '2.4')
_max_compression = float(MULTI_MAX_COMPRESSION_RATIO) if MULTI_MAX_COMPRESSION_RATIO else None


def _text_compression_ratio(text):
    """Same idea as faster-whisper's own compression_ratio: a hypothesis that is mostly one
    phrase repeated compresses far better than real text does."""
    data = text.encode('utf-8')
    if not data:
        return 1.0

    return len(data) / len(zlib.compress(data, 9))

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
        #  Investigating 2026-09-27 (bm workspace CHANGELOG same date): a downstream translation
        #  confidence score turned out useless for catching a bad *recognition* -- the model
        #  translates a hallucinated/repetition-looped transcript just as confidently as a real
        #  one, since by the time it sees the text there is nothing left to be unsure about.
        #  faster-whisper's own per-segment metrics are exactly what it used to decide "this is
        #  likely silence/nonsense" during decoding, so use them here too (see
        #  WHISPER_MIN_LOGPROB/WHISPER_MAX_COMPRESSION_RATIO above for what was found live).
        texts = []
        weighted_logprob = 0.0
        duration = 0.0
        no_speech_prob = 0.0
        compression_ratio = 0.0
        for seg in segments:
            texts.append(seg.text)
            seg_dur = max(seg.end - seg.start, 1e-6)
            weighted_logprob += seg.avg_logprob * seg_dur
            duration += seg_dur
            no_speech_prob = max(no_speech_prob, seg.no_speech_prob)
            compression_ratio = max(compression_ratio, seg.compression_ratio)
        text = ''.join(texts).strip()
        lang = lang_hint or info.language or ''
        avg_logprob = weighted_logprob / duration if duration else None
        low_confidence = (_asr_min_logprob is not None and avg_logprob is not None
                          and avg_logprob < _asr_min_logprob)
        repetitive = (_asr_max_compression is not None and compression_ratio > _asr_max_compression)
        prompt_leak = _is_prompt_leak(text, prompt)
        log.info('asr: avg_logprob=%s no_speech_prob=%.2f compression_ratio=%.2f '
                 'suppressed=%s text=%r',
                 f'{avg_logprob:.2f}' if avg_logprob is not None else 'n/a',
                 no_speech_prob, compression_ratio, low_confidence or repetitive or prompt_leak, text)
        if low_confidence or repetitive or prompt_leak:
            #  Same wire meaning as no speech detected: bmMediasoupServer's stt.ts already treats
            #  an empty text as "nothing to show" (`!res.text` short-circuits before emitting),
            #  so nothing downstream needs to learn a new case for this.
            text = ''
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
    if (_min_score is None and _max_compression is None) or len(decoded) != 1:
        return [text for text, _score in decoded]
    text, score = decoded[0]
    low_confidence = score is not None and score < _min_score if _min_score is not None else False
    ratio = _text_compression_ratio(text)
    repetitive = _max_compression is not None and ratio > _max_compression
    #  Logged unconditionally (not just on drop) while these are still unmeasured guesses (bm
    #  workspace CHANGELOG 2026-09-27): there is no other way to see what real (text, score,
    #  ratio) triples look like in this deployment, and a threshold picked without seeing the
    #  passing side too is just as much a guess as the defaults themselves. Turn back to logging
    #  only the dropped ones once trusted -- this is one line per translated utterance.
    kept = not (low_confidence or repetitive)
    log.info('translate: score=%s ratio=%.2f kept=%s text=%r',
             f'{score:.2f}' if score is not None else 'n/a', ratio, kept, text)
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


def warm_up():
    """Runs one throwaway recognition per model replica, and one translation, before listening.

    The first inference after loading pays for CUDA/cuBLAS initialisation: measured 1.6s right
    after a mode restart on rtx5070ti2 (24s on the very first start of a fresh install) against
    0.2s for every request after it. Paying it here means that once this port answers, every
    request is fast -- the meeting's first utterance included. A failure only costs that speed,
    so it is logged and serving goes on."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    t0 = time.time()
    #  Two seconds of faint noise: enough for the encoder and a few decoder steps to run.
    audio = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.01).astype(np.float32)

    def one(_):
        segments, _info = model.transcribe(audio, language='ja', beam_size=BEAM,
                                           vad_filter=False, condition_on_previous_text=False)
        list(segments)

    try:
        #  num_workers replicas each initialise on first use, so warm them side by side.
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(one, range(WORKERS)))
        if translator is not None:
            multi_tokenizer.src_lang = 'en'
            tokens = multi_tokenizer.convert_ids_to_tokens(multi_tokenizer.encode('Warming up.'))
            translator.translate_batch([tokens], target_prefix=[[_lang_token('ja')]],
                                       beam_size=4, max_decoding_length=16)
        log.info('warmed up in %.1fs', time.time() - t0)
    except Exception as e:  # noqa: BLE001 -- a cold first request is better than no service
        log.warning('warm-up failed (first requests will be slow): %s', e)


if __name__ == '__main__':
    warm_up()
    #  Enough HTTP threads to keep every model replica fed, plus room for /health.
    serve(app, listen=f'{HOST}:{PORT}', threads=WORKERS + 2)
