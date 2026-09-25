#!/usr/bin/env python3
"""GPU recognition sidecar for bm/bmMediasoupServer's STT.

Same wire contract as cpu_whisper_server.py, so bmMediasoupServer only needs another entry in
`stt.backends` (see the bm workspace doc, `stt-translation#stt-backend`):

    POST /asr?lang=<hint>[&prompt=<terms>]   body: 16kHz mono 16-bit WAV
        -> {"text": "...", "lang": "ja"}

Why this exists alongside SenseVoice, which is already on the GPU: SenseVoice-small is fast but
has no language model behind it and no way to bias decoding, and it mangles loanwords -- the
katakana vocabulary of a technical meeting is exactly where it fails. Whisper large-v3-turbo has
both, and `initial_prompt` lets the terms a room actually uses be handed to it up front.

`WHISPER_PROMPT` is that vocabulary: a short list of the words this deployment keeps getting
wrong. Keep it under ~200 characters -- it is prepended to the decoder's context, so a long list
costs latency and starts to steer the transcript itself.
"""
import io
import logging
import os

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from waitress import serve

MODEL = os.environ.get('WHISPER_MODEL', 'large-v3-turbo')
DEVICE = os.environ.get('WHISPER_DEVICE', 'cuda')
COMPUTE = os.environ.get('WHISPER_COMPUTE', 'float16')
PORT = int(os.environ.get('WHISPER_PORT', '8192'))
HOST = os.environ.get('WHISPER_HOST', '0.0.0.0')
PROMPT = os.environ.get('WHISPER_PROMPT', '')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('gpu-whisper')

app = Flask(__name__)

log.info('loading faster-whisper model=%s device=%s compute=%s ...', MODEL, DEVICE, COMPUTE)
model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
log.info('model loaded, listening on %s:%d (prompt: %s)', HOST, PORT, PROMPT or '(none)')


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
            #  Beam search is worth its cost here: this runs on a GPU, and the wrong branch is
            #  precisely how a loanword turns into a different word that also fits the audio.
            beam_size=5,
            condition_on_previous_text=False,  # each utterance stands alone; no drift across them
        )
        text = ''.join(seg.text for seg in segments).strip()
        lang = lang_hint or info.language or ''
    except Exception as e:  # noqa: BLE001 -- a bad request must not take the service down
        log.warning('transcribe failed: %s', e)
        return jsonify(text='', lang=lang_hint or ''), 500

    return jsonify(text=text, lang=lang)


@app.get('/health')
def health():
    return jsonify(status='ok', model=MODEL, device=DEVICE, prompt=bool(PROMPT))


if __name__ == '__main__':
    serve(app, listen=f'{HOST}:{PORT}')
