import os
import shutil

base_dir = "summ_json"  # folder containing crop_0.csv ... crop_79.csv

for i in range(80):
    group = i // 10  # 0..7
    folder = os.path.join(base_dir, f"category_summ_{group}")
    os.makedirs(folder, exist_ok=True)

    src = os.path.join(base_dir, f"question_{i}.json")
    dst = os.path.join(folder, f"question_{i}.json")

    if os.path.exists(src):
        shutil.copy2(src, dst)   # copy instead of move
    else:
        print(f"Missing file: {src}")
