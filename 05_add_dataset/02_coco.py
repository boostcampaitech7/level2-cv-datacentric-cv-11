# %%
import datetime
from typing import Dict
import json
import os

# num = 0

# JSON 파일이 있는 폴더 경로
# json_folder = f'./data/convert_data/cord_json_{num}/'
# output_path = f'./data/convert_data/cord_json_convert_{num}.json'

json_folder = f'./data/convert_data/cord_json/'
output_path = f'./data/convert_data/cord_json_convert.json'

# 기존 정보
info = {
    'year': 2024,
    'version': '1.0',
    'description': 'OCR Competition Data',
    'contributor': 'Naver Boostcamp',
    'url': 'www',
}
licenses = {
    'id': '1',
    'name': 'For Naver Boostcamp Competition',
    'url': None
}
categories = [{'id': 1, 'name': 'word'}]

# COCO 데이터 초기화
img_id = 1
annotation_id = 1
images = []
annotations = []

file_names = os.listdir(json_folder)
sorted_file_names = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))  # 파일 정렬

for file_name in sorted_file_names:
    if file_name.endswith('.json') and img_id:
        input_path = os.path.join(json_folder, file_name)

        with open(input_path, 'r') as f:
            file = json.load(f)
        image = {
            'id': img_id,
            'width': file['meta']['image_size']['width'],
            'height': file['meta']['image_size']['height'],
            'file_name': f'image_{img_id}.jpg',
            "license": 1,
            "flickr_url": None,
            "coco_url": None,
            'data_captured': None
        }
        images.append(image)

        for ann_info in file['valid_line']:
            for word_info in ann_info['words']:
                quad_info = word_info['quad']
                x1 = quad_info['x1']
                y1 = quad_info['y1']
                x2 = quad_info['x2']
                y3 = quad_info['y3']

                # COCO 형식으로 bbox 좌표 계산
                min_x = x1
                min_y = y1
                width = x2 - x1
                height = y3 - y1

                segmentation = [
                    [min_x, min_y, min_x + width, min_y, min_x + width, min_y + height, min_x, min_y + height]
                ]

                coco_annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": width * height,
                    "bbox": [min_x, min_y, width, height],
                    "iscrowd": 0,
                    'tags': ['Auto']
                }
                annotations.append(coco_annotation)
                annotation_id += 1

        img_id += 1

# 모든 데이터를 COCO 포맷으로 합치기
coco = {
    'info': info,
    'images': images,
    'annotations': annotations,
    'licenses': licenses,
    'categories': categories
}

# JSON 파일로 저장
with open(output_path, 'w') as f:
    json.dump(coco, f, indent=4)

# %%

now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d %H:%M:%S')

# 커스텀 coco포맷 json파일 -> ufo포맷
# input_path = '/data/ephemeral/home/level2-cv-datacentric-cv-10/data/medical/ufo/add_json.json'
# output_path = '/data/ephemeral/home/level2-cv-datacentric-cv-10/data/medical/ufo/_add_json.json'

# cvat작업 coco포맷 json파일 -> ufo포맷
input_path = f'./data/convert_data/cord_json_convert.json'
output_path = f'./data/convert_data/cord_json_convert_ufo.json'

ufo = {
    'images': {}
}

# %%


def coco_bbox_to_ufo(bbox):
    min_x, min_y, width, height = bbox
    return [
        [min_x, min_y],
        [min_x + width, min_y],
        [min_x + width, min_y + height],
        [min_x, min_y + height]
    ]


def coco_to_ufo(file: Dict, output_path: str) -> None:
    anno_id = 1
    for annotation in file['annotations']:
        file_info = file['images'][int(annotation['image_id'])-1]
        image_name = file_info['file_name']
        if image_name not in ufo['images']:
            anno_id = 1
            ufo['images'][image_name] = {
                "paragraphs": {},
                "words": {},
                "chars": {},
                "img_w": file_info["width"],
                "img_h": file_info["height"],
                "tags": ["autoannotated"],
                "relations": {},
                "annotation_log": {
                    "worker": "",
                    "timestamp": now,
                    "tool_version": "LabelMe or CVAT",
                    "source": None
                },
                "license_tag": {
                    "usability": True,
                    "public": False,
                    "commercial": True,
                    "type": None,
                    "holder": "Upstage"
                }
            }

            # anno_id = 1
        ufo['images'][image_name]['words'][str(anno_id).zfill(4)] = {
            "transcription": "",
            "points":  coco_bbox_to_ufo(annotation["bbox"]),
            "orientation": "Horizontal",
            "language": None,
            "tags": ['Auto'],
            "confidence": None,
            "illegibility": False
        }
        anno_id += 1

    with open(output_path, "w") as f:
        json.dump(ufo, f, indent=4)


# %%

with open(input_path, 'r') as f:
    file = json.load(f)
coco_to_ufo(file, output_path)
