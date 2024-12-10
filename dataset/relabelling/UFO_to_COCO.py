#!/usr/bin/env python
# coding: utf-8

# ## UFO to COCO

# In[11]:


import json
from typing import Dict, Any
from collections import Counter


# In[12]:


def convert_to_coco_format(data: Dict[str, Any]):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}], 
    }

    image_id_counter = 1
    annotation_id_counter = 1

    for file_name, file_data in data["images"].items():
        image_id = image_id_counter

        coco_image = {
            "id": image_id,
            "width": file_data["img_w"],
            "height": file_data["img_h"],
            "file_name": file_name,
            "license": 123, 
            "flickr_url": None, 
            "coco_url": None, 
            "date_captured": "2023-05-21 17:02:52"  
        }
        coco_data["images"].append(coco_image)

        for word_id, word_data in file_data["words"].items():
            annotation_id = annotation_id_counter
            [tl, tr, br, bl] = word_data["points"]
            width = max(tl[0], tr[0], br[0], bl[0]) - min(tl[0], tr[0], br[0], bl[0])
            height = max(tl[1], tr[1], br[1], bl[1]) - min(tl[1], tr[1], br[1], bl[1])
            x = min(tl[0], tr[0], br[0], bl[0])
            y = min(tl[1], tr[1], br[1], bl[1])
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # 전부 text
                "segmentation": [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]],
                "area": width * height,
                "bbox": [x, y, width, height],
                "iscrowd": 0  
            }
            coco_data["annotations"].append(coco_annotation)

            annotation_id_counter += 1  # 새로운 word 마다 +1

        image_id_counter += 1  # 새로운 image 마다 +1

    return coco_data


# In[13]:


# Load UFO json
with open("/Users/cherry/Downloads/cherry_ocr/code/data/vietnamese_receipt/ufo/train.json") as f:
    data = json.load(f)


# In[14]:


# Convert to COCO
coco_data = convert_to_coco_format(data)


# In[15]:


# Save COCO json
with open("vietnamese_train_coco_seg.json", "w") as f:
    json.dump(coco_data, f)


# In[ ]:




