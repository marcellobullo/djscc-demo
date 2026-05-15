#!/usr/bin/env bash

set -euo pipefail

REPO_ID="marcellobullo/adjscc"
TARGET_DIR="model_checkpoints"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

mkdir -p "${TARGET_DIR}"

FILES=(
  "AWGN_rate_8_AD_JSCC_SNR_random_EP_3.pth"
  "AWGN_rate_16_AD_JSCC_SNR_random_EP_3.pth"
)

for FILE in "${FILES[@]}"; do
  echo "Downloading ${FILE}..."
  if command -v curl >/dev/null 2>&1; then
    # curl is pre-installed on Windows 10/11 and standard on macOS
    curl -L -o "${TARGET_DIR}/${FILE}" "${BASE_URL}/${FILE}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${TARGET_DIR}/${FILE}" "${BASE_URL}/${FILE}"
  else
    # Fallback to python if neither curl nor wget is found
    python3 -c "import urllib.request; urllib.request.urlretrieve('${BASE_URL}/${FILE}', '${TARGET_DIR}/${FILE}')"
  fi
done

echo "Done. Files saved in ${TARGET_DIR}/"