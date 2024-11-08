import datetime
from typing import Dict
import json

now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d %H:%M:%S')

# coco to ufo
input_path = f'./data/convert_data/cord_json_convert_coco.json'
output_path = f'./data/convert_data/cord_json_convert_ufo.json'

ufo = {
    'images': {}
}


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


with open(input_path, 'r') as f:
    file = json.load(f)
coco_to_ufo(file, output_path)
