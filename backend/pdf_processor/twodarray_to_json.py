import json
import os


def tansfer2darraytojson(array_2d, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在
    save_path = os.path.join(save_dir, "array_data.json")

    # 寫入 JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(array_2d, f, ensure_ascii=False, indent=4)

    print(f"二維陣列已儲存至：{save_path}")
