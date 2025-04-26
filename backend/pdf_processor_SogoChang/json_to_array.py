import json
import os


def tansfer_json_to_array(save_dir, file_name):
    # 指定要讀取的 JSON 檔案路徑
    os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在
    load_path = os.path.join(save_dir, file_name)

    # 讀取 JSON 並還原成二維陣列
    with open(load_path, 'r', encoding='utf-8') as f:
        loaded_array = json.load(f)

    return loaded_array
