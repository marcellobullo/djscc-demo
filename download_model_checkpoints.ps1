$ErrorActionPreference = "Stop"

$REPO_ID = "marcellobullo/adjscc"
$TARGET_DIR = "model_checkpoints"
$BASE_URL = "https://huggingface.co/$REPO_ID/resolve/main"

New-Item -ItemType Directory -Force -Path $TARGET_DIR | Out-Null

$FILES = @(
  "AWGN_rate_8_AD_JSCC_SNR_random_EP_3.pth",
  "AWGN_rate_16_AD_JSCC_SNR_random_EP_3.pth"
)

foreach ($FILE in $FILES) {
    Write-Host "Downloading $FILE..."
    $TargetFile = Join-Path $TARGET_DIR $FILE
    $SourceUrl = "$BASE_URL/$FILE"
    Invoke-WebRequest -Uri $SourceUrl -OutFile $TargetFile
}

Write-Host "Done. Files saved in $TARGET_DIR/"
