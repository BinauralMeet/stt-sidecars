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
        text = ''.join(seg.text for seg in segments).strip()
        lang = lang_hint or info.language or ''
    except Exception as e:  # noqa: BLE001 -- this is the last fallback, must never crash the loop
        log.warning('transcribe failed: %s', e)
        return jsonify(text='', lang=lang_hint or ''), 500

    return jsonify(text=text, lang=lang)


@app.get('/health')
def health():
    return jsonify(status='ok', model=MODEL_SIZE)


if __name__ == '__main__':
    serve(app, listen=LISTEN, threads=4)
