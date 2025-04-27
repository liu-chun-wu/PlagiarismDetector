#!/bin/bash

# 設定下載 URL 和目標目錄
DOWNLOAD_URL="https://zenodo.org/api/records/4621403/files-archive"  # 替換為正確的 URL
TARGET_DIR="/MPC_data"

# 確保目標目錄存在
if [ ! -d "$TARGET_DIR" ]; then
  echo "Creating target directory: $TARGET_DIR"
fi

# 設定下載的檔案名稱
FILENAME="$(basename $DOWNLOAD_URL)"
FILEPATH="$TARGET_DIR/$FILENAME"

# 下載檔案
echo "Downloading file from $DOWNLOAD_URL..."
curl -L "$DOWNLOAD_URL" -o "$FILEPATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to download the file."
  exit 1
fi

# 解壓縮檔案到目標目錄
echo "Extracting file to $TARGET_DIR..."
unzip -o "$FILEPATH" -d "$TARGET_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to extract the file."
  exit 1
fi

# 清理暫存檔案
echo "Cleaning up temporary file: $FILEPATH"
rm "$FILEPATH"

# 完成
echo "Download and extraction completed successfully! Files are available in $TARGET_DIR."
