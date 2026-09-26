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
| `cpu_whisper_server.py` (2nd instance) | 8190 on **ai4**, reached at `172.17.0.1:8193` here | same contract | faster-whisper `medium`, CPU int8. **Runs on ai4, not this host** — see `#ai4` |
| `gpu_whisper_server.py` | 8192 | same contract as above, plus `POST /translate` | faster-whisper `large-v3-turbo` + M2M-100 418M, CUDA. **Runs on a GPU machine, not this host** — see `#gpu` |
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

The same process also answers `/translate` for the pairs the CPU translator has no model for
(it carries ja<->en only), with one multilingual model: M2M-100 418M by default, NLLB-200 with
`MULTI_KIND=nllb`. M2M is the default because NLLB is CC-BY-NC and this deployment is public,
so the licence would follow the whole service rather than just the model. Convert it once:

```sh
/opt/stt-sidecars/venv-gpu/bin/pip install transformers sentencepiece
/opt/stt-sidecars/venv-gpu/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
ct2-transformers-converter --model facebook/m2m100_418M --output_dir <dir> --quantization float16
/opt/stt-sidecars/venv-gpu/bin/pip uninstall -y torch   # only the conversion needed it
```

then point `MULTI_MODEL` at `<dir>`. Torch is only for reading the checkpoint, so the CPU wheel
is the one to install -- the CUDA build is ten times the download for no benefit.

It does **not** run on this host (no GPU here). It is deployed on `rtx5070ti`, under
`C:\Home\work\gpuwhisper\` with its own venv, and that machine's `control_api.py` owns it as a
mode alongside `hidream`/`sensevoice`/`irodori`:

```sh
curl -X POST http://<rtx5070ti>:8100/activate/gpuwhisper   # stops hidream, starts this
curl http://<rtx5070ti>:8100/status                        # active_modes
```

`control_api.py` passes it `WHISPER_PROMPT` (the katakana vocabulary) with
`WHISPER_PROMPT_LANG=ja`, and exposes it on the LAN so the reverse proxy can publish it. Three
things bit us there and are worth knowing before touching another Windows GPU box:

- **A prompt is decoder context, not a dictionary lookup.** The Japanese word list handed to an
  English utterance turned "ask not" into "アースクリーン": Whisper followed the prompt's language.
  `WHISPER_PROMPT_LANG` keeps it to the language it is written in.

- **CTranslate2 cannot find CUDA on Windows by itself.** The `nvidia-*-cu12` wheels put
  `cublas64_12.dll` under `site-packages/nvidia/*/bin`, which Windows does not search.
  `os.add_dll_directory()` alone is not enough either -- CTranslate2 loads the library by plain
  name, which searches `PATH` -- so `gpu_whisper_server.py` does both before importing it.
- **Pin cuBLAS to the CUDA minor version the GPU stack actually runs.** With the 12.9 wheel the
  process died mid-inference, taking the whole service with it and leaving the request hanging;
  `nvidia-cublas-cu12==12.8.*` (matching the machine's torch cu128) is stable.

There is no reverse-proxy path to that machine (unlike `/SENSEVOICE`), so the media server
reaches it through the SSH tunnel `bm/start-dev.sh` opens on 8192.

## Second CPU instance on ai4 {#ai4}

ai1 (this host) runs low on memory under normal dev-sandbox load — `earlyoom` has SIGTERM'd
a `faster-whisper` load attempt mid-benchmark here. ai2/ai3/ai4 are spare machines with
identical hardware sitting otherwise idle, so a second, bigger CPU instance runs on ai4 instead
of adding load here. Same `cpu_whisper_server.py`, unmodified, just a bigger
`CPU_WHISPER_MODEL` and more `CPU_WHISPER_THREADS` (8, not 4 — matching physical core count;
using both hyperthreads made it *slower*, measured 2026-09-26 on the i7-11800H in this fleet):

```sh
# on ai4, same as "Setup" above but skip translate_server.py/models/ entirely -- this instance
# only ever runs cpu_whisper_server.py
mkdir -p /opt/stt-sidecars/hf-cache
python3 -m venv /opt/stt-sidecars/venv
/opt/stt-sidecars/venv/bin/pip install faster-whisper flask waitress
# copy cpu_whisper_server.py to /opt/stt-sidecars/, then:
sudo cp systemd/stt-cpu-whisper-ai4.service /etc/systemd/system/stt-cpu-whisper.service
sudo systemctl daemon-reload
sudo systemctl enable --now stt-cpu-whisper
```

Unlike the ai1 instances, **this one is not exposed on `172.17.0.1` for a ufw docker0 rule to
open** — ai4 has no devsandbox containers to reach it from, so there is nothing to scope a rule
to. Instead ai1 reaches it through a permanent, narrowly-scoped SSH tunnel:

- `ai4-stt-tunnel.service` (systemd, on ai1, `Restart=always`): `ssh -N` forwarding
  `127.0.0.1:8193` and `172.17.0.1:8193` (ai1) to `127.0.0.1:8190` (ai4).
- The key it uses (`/root/.ssh/id_ed25519_ai4-stt-tunnel` on ai1) can do *only* that: ai4's
  `hase` account's `authorized_keys` restricts it with
  `command="/bin/echo restricted: port-forwarding only",restrict,port-forwarding,
  permitopen="127.0.0.1:8190"`. `restrict,port-forwarding` alone is not enough — it blocks an
  interactive pty but still lets `ssh host somecommand` run `somecommand` (measured: `ssh ...
  whoami` went through). The `command=` forces every exec/session request to that harmless
  echo instead, regardless of what the client asked for; `-N` (pure port-forward, no session at
  all) is unaffected. `permitopen` also verified: forwarding to a non-listed port locally
  accepts the TCP connection (that's just the local `ssh -L` listener) but no bytes cross —
  ai4's sshd refuses to open the channel.
- `ufw allow in on docker0 to any port 8193 proto tcp` on ai1 (same pattern as 8190/8191)
  scopes the *local* end of the tunnel to sandbox containers only.

Reproducing this on ai2/ai3 is the same recipe with a fresh dedicated keypair per host (never
reuse the ai4 one) and the next free local port (8194, ...).

## The dedicated ai4 instance {#ai4}

`systemd/stt-cpu-whisper-ai4.service` is a second CPU instance on a host that has nothing else
to do, so the media servers keep getting subtitles while the GPU is busy with someone else's
work. It runs `medium` rather than `small` (16 cores, nothing competing for them) and, unlike
the ai1 instance, is not niced down.

That host has no checkout of this repo: the two files it runs live in `/opt/stt-sidecars/`
directly, so updating it is a copy rather than a pull:

```sh
scp cpu_whisper_server.py sidecar_auth.py ai4:/opt/stt-sidecars/
ssh ai4 sudo systemctl restart stt-cpu-whisper
```

Both files are needed -- `cpu_whisper_server.py` imports `sidecar_auth`.

Measured there: 11 seconds of speech in 1.9s with `small`; `medium` is slower again, so this
instance answers after the speaker has finished rather than while they are talking. That is what
a fallback is for -- interim subtitles switch themselves off when a backend cannot outrun speech
(see `stt-translation#limits`).

## Serving another machine {#remote}

Both sidecars listen on loopback (plus docker0) by default, which is the whole of their security
model: anything that can reach one can spend its CPU and read back what was said. Pointing a
media server on another host at one means changing both halves of that:

- **`CPU_WHISPER_LISTEN` / `TRANSLATE_LISTEN`** (space-separated `host:port`, as waitress takes
  it) to add the address that host is reached on.
- **`STT_API_KEY`** to require `Authorization: Bearer <key>`, which is what bmMediasoupServer
  already sends when the backend entry names an `apiKeyEnv`. `/health` stays open so probes work.

Neither covers the wire itself: **the audio goes out in plain HTTP**. On a public address, put it
behind TLS (or restrict it to the media servers by firewall) before pointing a meeting at it.

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
