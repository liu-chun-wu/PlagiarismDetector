import json
import os


def tansfer_array_to_json(array, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在
    save_path = os.path.join(save_dir, file_name)

    # 寫入 JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(array, f, ensure_ascii=False, indent=4)

    print(f"陣列已儲存至：{save_path}")
