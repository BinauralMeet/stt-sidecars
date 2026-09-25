# stt-sidecars

The two CPU-only HTTP sidecars that `bm/bmMediasoupServer`'s STT/translation feature falls back
to when the GPU recognition service isn't available. Design and wire contracts are documented in
the `bm/` workspace doc (`docs/bin/doc show stt-translation`, specifically `#fallback`,
`#stt-backend` and `#hostwork`) -- this repo only holds what that doc calls "(別リポジトリ)
STT/翻訳サイドカー": deliberately not inside `binaural-meet`/`bmMediasoupServer`, since neither of
those repos should carry ML runtime dependencies.

Both services run directly from this checkout (no build/deploy step) using a Python venv and
converted models that live outside git, under `/opt/stt-sidecars/` (parallel to `/opt/lm-tool`'s
layout for the same reason: multi-hundred-MB binaries have no business in a git history).

| Service | Port | Endpoint | Backing |
|---|---|---|---|
| `cpu_whisper_server.py` | 8190 | `POST /asr?lang=<hint>[&prompt=<terms>]` (body: 16kHz mono s16le WAV) | faster-whisper `small`, CPU int8 |
| `gpu_whisper_server.py` | 8192 | same contract as above | faster-whisper `large-v3-turbo`, CUDA fp16. **Runs on a GPU machine, not this host** — see `#gpu` |
| `translate_server.py` | 8191 | `POST /translate` `{texts,src,dsts}` | CTranslate2 + FuguMT (`staka/fugumt-{ja-en,en-ja}`), CPU int8 |

## Setup (once per host)

```sh
mkdir -p /opt/stt-sidecars/models /opt/stt-sidecars/hf-cache
python3 -m venv /opt/stt-sidecars/venv
/opt/stt-sidecars/venv/bin/pip install -r requirements.txt

./scripts/convert_models.sh   # see `#models` below for why FuguMT, not Helsinki-NLP/opus-mt-en-jap

# transformers+torch were only needed for the conversion above:
/opt/stt-sidecars/venv/bin/pip uninstall -y transformers torch

sudo cp systemd/stt-cpu-whisper.service systemd/stt-translate.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stt-cpu-whisper stt-translate
```

`faster-whisper` downloads its model from Hugging Face on first start (cached under
`HF_HOME=/opt/stt-sidecars/hf-cache`, see the unit file); after that it needs no network.
`translate_server.py` never touches the network -- everything it needs is under
`/opt/stt-sidecars/models/` from the conversion step above.

## Verifying

```sh
curl http://127.0.0.1:8190/health
curl http://127.0.0.1:8191/health
curl -X POST http://127.0.0.1:8191/translate -H 'Content-Type: application/json' \
  -d '{"texts":["こんにちは、聞こえていますか?"],"src":"ja","dsts":["en"]}'
```

Then in `bmMediasoupServer/config.js`, add to `stt.backends` / set `translation.endpoint`
(`stt-translation#fallback` has the exact shape) and restart the media/main server.

## Deploying the GPU recognizer {#gpu}

`gpu_whisper_server.py` is the answer to SenseVoice mangling katakana: SenseVoice-small is fast
but carries no language model and cannot be biased, while Whisper has both and takes an
`initial_prompt` of the terms a room actually uses (`WHISPER_PROMPT` in the unit file).

It is written to the same contract as the CPU sidecar, so `bmMediasoupServer` needs nothing but
another entry in `stt.backends` — no `upload`/`langParam`, unlike SenseVoice.

It does **not** run on this host (no GPU here). It is deployed on `rtx5070ti`, under
`C:\Home\work\gpuwhisper\` with its own venv, and that machine's `control_api.py` owns it as a
mode alongside `hidream`/`sensevoice`/`irodori`:

```sh
curl -X POST http://<rtx5070ti>:8100/activate/gpuwhisper   # stops hidream, starts this
curl http://<rtx5070ti>:8100/status                        # active_modes
```

`control_api.py` passes it `WHISPER_PROMPT` (the katakana vocabulary) and pins it to
`127.0.0.1`. Two things bit us there and are worth knowing before touching another Windows GPU
box:

- **CTranslate2 cannot find CUDA on Windows by itself.** The `nvidia-*-cu12` wheels put
  `cublas64_12.dll` under `site-packages/nvidia/*/bin`, which Windows does not search.
  `os.add_dll_directory()` alone is not enough either -- CTranslate2 loads the library by plain
  name, which searches `PATH` -- so `gpu_whisper_server.py` does both before importing it.
- **Pin cuBLAS to the CUDA minor version the GPU stack actually runs.** With the 12.9 wheel the
  process died mid-inference, taking the whole service with it and leaving the request hanging;
  `nvidia-cublas-cu12==12.8.*` (matching the machine's torch cu128) is stable.

There is no reverse-proxy path to that machine (unlike `/SENSEVOICE`), so the media server
reaches it through the SSH tunnel `bm/start-dev.sh` opens on 8192.

## Known limits

- Only `ja<->en`. Adding a language means converting another OPUS-MT-family model
  (`scripts/convert_models.sh`) and adding a line to `PAIRS` in `translate_server.py`.
- `cpu_whisper_server.py` is single-model, single-process (waitress, 4 threads): concurrent
  transcriptions serialize on CPU. Fine for an occasional-fallback role; not meant to carry a
  whole meeting's STT load the way the GPU backend does.
- Both are plain HTTP, no auth -- matches `config.js`'s
  `{kind:'cpuWhisper', endpoint:'http://172.17.0.1:8190/asr'}` example, which names no
  `apiKeyEnv`. They listen on `127.0.0.1` **and** `172.17.0.1` (docker0's host-side address,
  see `waitress`'s `listen=` in both servers), so every sandbox container on this host can
  reach them unauthenticated -- accepted so BM's dev checkout (which runs inside a container,
  `start-dev.sh`) can reach the fallback chain. `ufw allow in on docker0 to any port 8190/8191
  proto tcp` bounds this to docker0 traffic only, not the public interface
  (`bm/docs stt-translation#hostwork`, `CHANGELOG#stt-sidecar-docker0-expose`). Do not bind
  these to a public interface without adding auth.

## 設計判断の記録 {#models}

**なぜ `Helsinki-NLP/opus-mt-en-jap` ではなく `staka/fugumt-en-ja` か**: 当初 hostwork の
「CTranslate2 + opus-mt」という記述通り Helsinki-NLP の `opus-mt-ja-en`/`opus-mt-en-jap` を
`ct2-transformers-converter` で変換して試したところ、`ja->en` は正しく動いたが
(`"こんにちは、元気ですか?" -> "Hello. How are you?"`)、`en->ja` は流暢だが無関係な文
(`"Hello, how are you?" -> "陰府よ、おまえはどうしているのか"` = 直訳「Sheol, how are you」)
を返した。`ct2`変換やトークナイズのバグではなく、素の`transformers`
(`MarianMTModel.generate`)でも同じ出力になることを確認した(2026-09-24) —
`opus-mt-en-jap`はJW300(宗教文書コーパス)primarily学習という、モデル自体の性質。
`staka/fugumt-en-ja`/`fugumt-ja-en`(JParaCrawl学習)に替えたところ両方向とも自然な会話文で
正しく訳せることを確認したため、こちらを採用した。

**なぜ`translator.translate_batch`に手動で`"</s>"`を追加しているか**: これも上の検証中に発見
——ソーストークン列の末尾に`</s>`を足さずに渡すと、有効なsentencepieceトークン化であっても
文法的には流暢だが無限ループ/無関係な出力になる(HFの`MarianTokenizer.__call__`は自動でこれを
付与するが、`sp.encode()`を素で呼ぶ経路にはその処理が無い)。CTranslate2自体のドキュメント例
(`opennmt.net/CTranslate2`のOPUS-MTガイド)もこの点に触れておらず、実機で出力を見比べて
初めて気づいた。
