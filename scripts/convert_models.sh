#!/bin/sh
# Regenerates /opt/stt-sidecars/models/ from upstream HF checkpoints. Run once at setup, or after
# bumping a model choice below. Needs `transformers`+`torch` from requirements.txt (see its
# comment -- these two are the only conversion-only deps, safe to drop afterwards).
#
# Why FuguMT (staka/fugumt-{en-ja,ja-en}) instead of Helsinki-NLP/opus-mt-en-jap: the Helsinki
# en->jap checkpoint is trained mostly on JW300 (a religious-text corpus) and produces fluent but
# unrelated output for ordinary conversation (verified 2026-09-24, e.g. "Hello, how are you?" ->
# a Sheol/"陰府" sentence). FuguMT is JParaCrawl-trained and tested correct on casual sentences
# both directions. Helsinki-NLP/opus-mt-ja-en (the other direction) tested fine, but we use FuguMT
# for both directions for one less thing to reason about.
set -e
VENV=/opt/stt-sidecars/venv
OUT=/opt/stt-sidecars/models

"$VENV/bin/ct2-transformers-converter" --model staka/fugumt-ja-en \
  --output_dir "$OUT/fugumt-ja-en" --quantization int8 \
  --copy_files source.spm target.spm vocab.json --force

"$VENV/bin/ct2-transformers-converter" --model staka/fugumt-en-ja \
  --output_dir "$OUT/fugumt-en-ja" --quantization int8 \
  --copy_files source.spm target.spm vocab.json --force

echo "done: $OUT/fugumt-ja-en, $OUT/fugumt-en-ja"
