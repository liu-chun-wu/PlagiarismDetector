# PlagiarismDetector

[MPC data source link](https://zenodo.org/records/4621403)

```bash
bash MPC_download_and_extract.sh
```

```bash
pip freeze>requirements.txt
```

```bash
pip install -r requirements.txt
```

main : 處理主邏輯，包括資料加載、訓練、評估等核心部分。

utils : 包含通用的函數，例如 preprocess_function 和資料處理相關邏輯。

data_loader : 處理資料的載入與拆分邏輯，例如使用 datasets 函式庫載入資料，並進行預處理。

model : 包括模型的加載、初始化與訓練參數的設定。

evaluation : 負責模型的測試與結果的可視化分析。

constants : 存參數的地方。
