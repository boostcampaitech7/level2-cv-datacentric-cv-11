# %%
import json
import os
import shutil
from sklearn.model_selection import train_test_split

# 언어 지정
# # chinese, thai, vietnamese, japanese
language = "vietnamese"

# 경로 설정
json_path = f'./data/{language}_receipt/ufo/train.json'
image_folder = f'./data/{language}_receipt/img/train/'
train_image_folder = f'{language}_train_images'
val_image_folder = f'{language}_val_images'

# 폴더 생성
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)

# JSON 파일 로드
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 이미지 ID 목록 추출
image_ids = list(data['images'].keys())
# print(image_ids)

# train/validation split
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

# dict 딕셔너리 초기화
train_data = {'images': {img_id: data['images'][img_id] for img_id in train_ids}}
val_data = {'images': {img_id: data['images'][img_id] for img_id in val_ids}}

# JSON 파일로 저장
with open(f'train_{language}.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(f'val_{language}.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

# 이미지 파일을 각각의 폴더로 복사
for img_id in train_ids:
    img_path = os.path.join(image_folder, img_id)
    if os.path.exists(img_path + '.jpg'):
        img_path += '.jpg'
    shutil.copy(img_path, train_image_folder)

for img_id in val_ids:
    img_path = os.path.join(image_folder, img_id)
    if os.path.exists(img_path + '.jpg'):
        img_path += '.jpg'
    shutil.copy(img_path, val_image_folder)

print("Train/Validation split 완료! JSON 파일과 이미지 저장")

# %%
