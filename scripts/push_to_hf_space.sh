#!/usr/bin/env bash
# Publish the demo to a Hugging Face Space (Docker SDK). Run from the repo root after `hf auth login`
# (or with HF_TOKEN set to a write token). Creates the Space if it does not exist.
set -euo pipefail
SPACE="${1:-divyanshg03/f1-winner-predictor}"
TMP="$(mktemp -d)"
git archive HEAD | tar -x -C "$TMP"                      # tracked files of the current commit only
cp deploy/hf_space_README.md "$TMP/README.md"            # Spaces reads its config from README front matter
hf repo create "$SPACE" --repo-type space --space-sdk docker --exist-ok
hf upload "$SPACE" "$TMP" . --repo-type space --commit-message "Deploy from GitHub $(git rev-parse --short HEAD)"
echo "Space: https://huggingface.co/spaces/$SPACE"
