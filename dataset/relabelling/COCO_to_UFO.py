#!/usr/bin/env python
# coding: utf-8

# In[23]:


import json
from typing import Dict, Any
from collections import Counter
import re


# In[24]:


def rename_filename(filename):
    # Match the original pattern and extract groups
    match = re.match(r'(\w+)-(\w+)-(\w+)_(\w+)-(\w+2?)_(\d+)_page(\d+)_jpg\.rf\.\w+\.jpg', filename)
    if match:
        # Extract matched groups and reformat them
        new_filename = f"{match.group(1)}.{match.group(2)}.{match.group(3)}_{match.group(4)}.{match.group(5)}_{match.group(6)}_page{match.group(7)}.jpg"
        return new_filename
    else:
        # Raise an error if the filename doesn't match the pattern
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")

def convert_to_your_format(data: Dict[str, Any]):
    your_format = {"images": {}}

    # {imd_id : 파일명} 형태의 dictionary
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        image_name = image_id_to_filename[image_id]
        original_image_name = rename_filename(image_name)
        bbox = annotation["segmentation"]

        tl = [bbox[0][0], bbox[0][1]]
        tr = [bbox[0][2], bbox[0][3]]
        br = [bbox[0][4], bbox[0][5]]
        bl = [bbox[0][6], bbox[0][7]]
        
        # COCO에서 UFO로 변환시 비는 정보는 placeholder로 대체
        if original_image_name not in your_format["images"]:
            your_format["images"][original_image_name] = {
                "paragraphs": {},
                "words": {},
                "chars": {},
                "img_w": data["images"][image_id]["width"], 
                "img_h": data["images"][image_id]["height"],  
                "num_patches": None,
                "tags": [], 
                "relations": {},
                "annotation_log": {
                    "worker": "worker",
                    "timestamp": "2024-11-04",
                    "tool_version": "",
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

        your_format["images"][original_image_name]["words"][str(annotation["id"]).zfill(4)] = {
            "transcription": "",  
            "points": [tl, tr, br, bl]
        }

    return your_format


# In[26]:


# Load COCO JSON
with open("/Users/cherry/Downloads/cherry_ocr/chinese_relabel3_coco.json") as f:
    coco_data = json.load(f)

# UFO로 변환
your_format_data = convert_to_your_format(coco_data)

# UFO JSON Save
with open("chinese_relabel3_ufo.json", "w") as f:
    json.dump(your_format_data, f, indent=4)


# In[ ]:




