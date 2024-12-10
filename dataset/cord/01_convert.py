from PIL import Image
from io import BytesIO
import json
import pandas as pd
import os


# `parquet` file
parquet_files = [
    './data/origin/train-00000-of-00004-b4aaeceff1d90ecb.parquet',
    './data/origin/train-00001-of-00004-7dbbe248962764c5.parquet',
    './data/origin/train-00002-of-00004-688fe1305a55e5cc.parquet',
    './data/origin/train-00003-of-00004-2d0cd200555ed7fd.parquet',
]

# dir json save
image_dir = './data/convert_data/cord_images'
json_dir = './data/convert_data/cord_json'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

# index
image_index = 1

for num, file_path in enumerate(parquet_files):
    df = pd.read_parquet(file_path)

    for index, row in df.iterrows():
        # image save
        image_data = row['image']['bytes']
        image = Image.open(BytesIO(image_data))
        image_file_path = f'{image_dir}/image_{image_index}.jpg'
        image.save(image_file_path)

        # ground_truth save
        ground_truth_str = row['ground_truth']
        ground_truth_dict = json.loads(ground_truth_str)

        # JSON save
        with open(f'{json_dir}/image_{image_index}.json', 'w', encoding='utf-8') as json_file:
            json.dump(ground_truth_dict, json_file, ensure_ascii=False, indent=4)

        image_index += 1

print(f"Images saved in: {image_dir}")
print(f"JSON files saved in: {json_dir}")
