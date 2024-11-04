# %%

from PIL import Image
from io import BytesIO
import json
import pandas as pd
import os


# train-00000-of-00004-b4aaeceff1d90ecb.parquet
# train-00001-of-00004-7dbbe248962764c5.parquet
# train-00002-of-00004-688fe1305a55e5cc.parquet
# train-00003-of-00004-2d0cd200555ed7fd.parquet

num = 3

df = pd.read_parquet('./data/origin/train-00003-of-00004-2d0cd200555ed7fd.parquet')

image_dir = f'./data/convert_data/cord_images_{num}'
json_dir = f'./data/convert_data/cord_json_{num}'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

# 이미지 저장
for index, row in df.iterrows():
    image_data = row['image']['bytes']
    image = Image.open(BytesIO(image_data))
    image.save(f'./data/convert_data/cord_images_{num}/image_{index+1}.jpg')  # jpg 파일명을 단순하게 붙여줬다.

# json 저장
for index, row in df.iterrows():
    ground_truth_str = row['ground_truth']
    ground_truth_dict = json.loads(ground_truth_str)
    with open(f'./data/convert_data/cord_json_{num}/image_{index+1}.json', 'w', encoding='utf-8') as json_file:
        json.dump(ground_truth_dict, json_file, ensure_ascii=False, indent=4)

# %%


# `parquet` 파일 목록
parquet_files = [
    './data/origin/train-00000-of-00004-b4aaeceff1d90ecb.parquet',
    './data/origin/train-00001-of-00004-7dbbe248962764c5.parquet',
    './data/origin/train-00002-of-00004-688fe1305a55e5cc.parquet',
    './data/origin/train-00003-of-00004-2d0cd200555ed7fd.parquet'
]

# 이미지와 JSON 파일을 저장할 경로
image_dir = './data/convert_data/cord_images'
json_dir = './data/convert_data/cord_json'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

# 전체 파일에서 이미지와 JSON을 저장하기 위한 인덱스
image_index = 1

# 각 `parquet` 파일을 순회하며 이미지와 JSON 파일을 저장
for num, file_path in enumerate(parquet_files):
    df = pd.read_parquet(file_path)

    for index, row in df.iterrows():
        # 이미지 저장
        image_data = row['image']['bytes']
        image = Image.open(BytesIO(image_data))
        image_file_path = f'{image_dir}/image_{image_index}.jpg'
        image.save(image_file_path)

        # ground_truth 저장
        ground_truth_str = row['ground_truth']
        ground_truth_dict = json.loads(ground_truth_str)

        # JSON 파일로 저장
        with open(f'{json_dir}/image_{image_index}.json', 'w', encoding='utf-8') as json_file:
            json.dump(ground_truth_dict, json_file, ensure_ascii=False, indent=4)

        # 다음 이미지를 위해 인덱스를 증가시킴
        image_index += 1

print(f"Images saved in: {image_dir}")
print(f"JSON files saved in: {json_dir}")
