#!/usr/bin/env python3
"""cpuWhisper sidecar for bm/bmMediasoupServer's STT fallback chain.

Wire contract (see bm/docs stt-translation#stt-backend and
bmMediasoupServer/src/MediaServer/SttBackend.ts):

    POST /asr?lang=<hint>   body: 16kHz mono 16-bit WAV   ->  {"text": "...", "lang": "ja"}

`lang` is an optional hint (BM's best guess for the speaker's language); an empty/absent value
lets faster-whisper auto-detect. This is the CPU degradation target -- always available, no GPU,
no external network dependency once the model is cached locally.
"""
import io
import logging
import os

from flask import Flask, request, jsonify

from sidecar_auth import install as install_auth
from faster_whisper import WhisperModel
from waitress import serve

MODEL_SIZE = os.environ.get('CPU_WHISPER_MODEL', 'small')
CPU_THREADS = int(os.environ.get('CPU_WHISPER_THREADS', '4'))
PORT = int(os.environ.get('CPU_WHISPER_PORT', '8190'))
#  Vocabulary this deployment keeps getting wrong -- handed to the decoder as context so
#  loanwords and names stand a chance. Same knob as gpu_whisper_server.py; see its docstring.
PROMPT = os.environ.get('WHISPER_PROMPT', '')
#  Same signals, same reasoning and same defaults as gpu_whisper_server.py's matching constants
#  (found live 2026-09-27, bm workspace CHANGELOG same date): avg_logprob alone missed a
#  repetition-looped hallucination entirely (near-perfect confidence repeating the same token),
#  so both a confidence floor and faster-whisper's own compression-ratio signal are checked.
WHISPER_MIN_LOGPROB = os.environ.get('WHISPER_MIN_LOGPROB', '-1.0')
WHISPER_MAX_COMPRESSION_RATIO = os.environ.get('WHISPER_MAX_COMPRESSION_RATIO', '2.4')
_asr_min_logprob = float(WHISPER_MIN_LOGPROB) if WHISPER_MIN_LOGPROB else None
_asr_max_compression = float(WHISPER_MAX_COMPRESSION_RATIO) if WHISPER_MAX_COMPRESSION_RATIO else None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('cpu-whisper')

app = Flask(__name__)
if install_auth(app):
    log.info('bearer token required (STT_API_KEY is set)')

log.info('loading faster-whisper model=%s cpu_threads=%d ...', MODEL_SIZE, CPU_THREADS)
model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8', cpu_threads=CPU_THREADS)
#  172.17.0.1 is docker0's host-side address, reachable from every sandbox container on this
#  host (`bm/docs stt-translation#hostwork`). This has no auth, so anything with container access
#  on this host can reach it -- accepted tradeoff so the dev sandboxes can reach the fallback ASR.
#  Space-separated host:port list, as waitress takes it. The default serves this host and its
#  containers only; a sidecar that has to answer another machine needs its own address here --
#  and `STT_API_KEY` set, since anything that can reach it can read back what was said.
LISTEN = os.environ.get('CPU_WHISPER_LISTEN', f'127.0.0.1:{PORT} 172.17.0.1:{PORT}')
log.info('model loaded, listening on %s', LISTEN)


@app.post('/asr')
def asr():
    lang_hint = (request.args.get('lang') or '').strip().lower() or None
    prompt = request.args.get('prompt') or PROMPT or None
    audio = request.get_data()
    if not audio:
        return jsonify(text='', lang=lang_hint or ''), 400

    try:
        segments, info = model.transcribe(
            io.BytesIO(audio),
            language=lang_hint,
            initial_prompt=prompt,
            vad_filter=False,  # BM already did VAD upstream; avoid double-guessing endpoints
        )
        texts = []
        weighted_logprob = 0.0
        duration = 0.0
        compression_ratio = 0.0
        for seg in segments:
            texts.append(seg.text)
            seg_dur = max(seg.end - seg.start, 1e-6)
            weighted_logprob += seg.avg_logprob * seg_dur
            duration += seg_dur
            compression_ratio = max(compression_ratio, seg.compression_ratio)
        text = ''.join(texts).strip()
        lang = lang_hint or info.language or ''
        avg_logprob = weighted_logprob / duration if duration else None
        low_confidence = (_asr_min_logprob is not None and avg_logprob is not None
                          and avg_logprob < _asr_min_logprob)
        repetitive = (_asr_max_compression is not None and compression_ratio > _asr_max_compression)
        log.info('asr: avg_logprob=%s compression_ratio=%.2f suppressed=%s text=%r',
                 f'{avg_logprob:.2f}' if avg_logprob is not None else 'n/a',
                 compression_ratio, low_confidence or repetitive, text)
        if low_confidence or repetitive:
            text = ''  # same wire meaning as no speech detected -- see gpu_whisper_server.py
    except Exception as e:  # noqa: BLE001 -- this is the last fallback, must never crash the loop
        log.warning('transcribe failed: %s', e)
        return jsonify(text='', lang=lang_hint or ''), 500

    return jsonify(text=text, lang=lang)


@app.get('/health')
def health():
    return jsonify(status='ok', model=MODEL_SIZE)


if __name__ == '__main__':
    serve(app, listen=LISTEN, threads=4)
