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
| `cpu_whisper_server.py` | 8190 | `POST /asr?lang=<hint>` (body: 16kHz mono s16le WAV) | faster-whisper `small`, CPU int8 |
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

## Known limits

- Only `ja<->en`. Adding a language means converting another OPUS-MT-family model
  (`scripts/convert_models.sh`) and adding a line to `PAIRS` in `translate_server.py`.
- `cpu_whisper_server.py` is single-model, single-process (waitress, 4 threads): concurrent
  transcriptions serialize on CPU. Fine for an occasional-fallback role; not meant to carry a
  whole meeting's STT load the way the GPU backend does.
- Both are plain HTTP on `127.0.0.1`, no auth -- matches `config.js`'s
  `{kind:'cpuWhisper', endpoint:'http://localhost:8190/asr'}` example, which names no
  `apiKeyEnv`. Do not bind these to a public interface without adding one.

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
