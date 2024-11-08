import os
import os.path as osp
import json
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image, ImageDraw

def parse_args():
    parser = argparse.ArgumentParser(description="Convert UFO JSONs from multiple folds to COCO, perform filtering, WBF ensemble, and convert back to UFO")
    
    parser.add_argument('--json_files', type=str, nargs='+', required=True,
                        help="List of paths to UFO JSON files from different folds (e.g., /path/to/annotations0.json /path/to/annotations1.json ...)")
    parser.add_argument('--test_img_dirs', type=str, nargs='+', required=True,
                        help="List of directories containing test images for each language (e.g., ./data/chinese_receipt/img/test ./data/japanese_receipt/img/test ...)")
    parser.add_argument('--save_dir', type=str, default='ensemble_results',
                        help="Directory to save the ensemble UFO results")
    parser.add_argument('--weights', type=float, nargs='+', required=True,
                        help="List of weights corresponding to each JSON file (e.g., 1.0 0.8 0.6 0.4 0.2)")
    parser.add_argument('--iou_thr', type=float, default=0.5,
                        help="IOU threshold for WBF")
    parser.add_argument('--skip_box_thr', type=float, default=0.0,
                        help="Skip boxes with confidence scores lower than this threshold")
    parser.add_argument('--min_box_area', type=float, default=0.0005,
                        help="Minimum normalized area of box to keep (e.g., 0.0005 for 0.05%)")
    parser.add_argument('--max_box_area', type=float, default=0.99,
                        help="Maximum normalized area of box to keep (e.g., 0.99)")
    parser.add_argument('--min_num_thr', type=int, default=2,
                        help="Minimum number of overlapping boxes to consider")
    parser.add_argument('--max_num_thr', type=int, default=1,
                        help="Maximum number of overlapping boxes allowed to consider as noise")
    
    args = parser.parse_args()
    return args

def load_json_files(json_files):
    """
    Load all JSON files from the specified list of file paths.

    Args:
        json_files (list of str): List of JSON file paths.

    Returns:
        list of dict: Each dict represents data from one JSON file.
    """
    all_data = []
    for json_file in json_files:
        if not osp.exists(json_file):
            print(f"Warning: JSON file {json_file} does not exist. Skipping.")
            continue
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                all_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON file {json_file}. Error: {e}")
    return all_data

def get_image_sizes(test_img_dirs):
    """
    Get sizes of all images in the test image directories.

    Args:
        test_img_dirs (list of str): List of paths to test image directories.

    Returns:
        dict: Mapping from unique image filename to (width, height).
    """
    image_sizes = {}
    for test_img_dir in test_img_dirs:
        image_fpaths = glob(osp.join(test_img_dir, '*'))
        for img_path in image_fpaths:
            image_name = osp.basename(img_path) 
            unique_image_name = image_name  # 언어 접두사 제거
        # for img_path in image_fpaths:
        #     image_name = osp.basename(img_path)
        #     # Extract language name from directory path (assuming structure: .../{language}_receipt/img/test)
        #     language = osp.basename(osp.dirname(osp.dirname(img_path)))  # e.g., 'chinese' from './data/chinese_receipt/img/test'
        #     unique_image_name = f"{language}_{image_name}"
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    image_sizes[unique_image_name] = (width, height)
            except Exception as e:
                print(f"Warning: Unable to open image {img_path}. Error: {e}")
    return image_sizes

def intersection_area(box1, box2):
    """
    Calculate the intersection area of box2 relative to box1.

    Args:
        box1 (list): [x_min, y_min, x_max, y_max] of the first box.
        box2 (list): [x_min, y_min, x_max, y_max] of the second box.

    Returns:
        float: Intersection area ratio relative to box1's area.
    """
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)

    if box1_area == 0:
        return 0
    return (inter_width * inter_height) / box1_area

def extract_annotations_from_json(all_json_data, image_sizes, min_box_area, max_box_area):
    """
    Extract annotations from loaded JSON data and convert to COCO format.

    Args:
        all_json_data (list of dict): Loaded JSON data from all folds.
        image_sizes (dict): Mapping from unique image filename to (width, height).
        min_box_area (float): Minimum normalized area to keep.
        max_box_area (float): Maximum normalized area to keep.

    Returns:
        dict: COCO formatted data with filtered annotations.
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}]
    }
    annotation_id = 1
    image_id_map = {}

    # Assign unique image IDs
    for idx, (image_name, size) in enumerate(image_sizes.items(), 1):
        coco_data["images"].append({
            "id": idx,
            "file_name": image_name,
            "width": size[0],
            "height": size[1]
        })
        image_id_map[image_name] = idx

    # Iterate over all JSON data
    for fold_data in all_json_data:
        images = fold_data.get('images', {})
        for image_name, image_data in images.items():
            unique_image_name = image_name  # Assuming JSON 파일에 이미 고유한 이름 사용
            if unique_image_name not in image_id_map:
                print(f"Warning: Image {unique_image_name} not found in test images. Skipping.")
                continue  # Skip images not in test set

            words = image_data.get('words', {})
            for word_id, word_data in words.items():
                points = word_data.get('points', [])
                if len(points) != 4:
                    print(f"Warning: Word ID {word_id} in image {unique_image_name} does not have 4 points. Skipping.")
                    continue
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

                # Get image size
                width, height = image_sizes.get(unique_image_name, (1, 1))
                if width == 0 or height == 0:
                    print(f"Warning: Image size for {unique_image_name} is zero. Skipping.")
                    continue

                # Normalize coordinates
                norm_x_min = x_min / width
                norm_y_min = y_min / height
                norm_x_max = x_max / width
                norm_y_max = y_max / height

                # Calculate normalized area
                box_area = (norm_x_max - norm_x_min) * (norm_y_max - norm_y_min)
                if box_area < min_box_area or box_area > max_box_area:
                    continue  # Filter out small or excessively large boxes

                # Ensure normalized coordinates are within [0, 1]
                if not (0 <= norm_x_min <= 1 and 0 <= norm_x_max <= 1 and 0 <= norm_y_min <= 1 and 0 <= norm_y_max <= 1):
                    continue

                # Append to annotations
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id_map[unique_image_name],
                    "category_id": 1,
                    "bbox": [norm_x_min, norm_y_min, norm_x_max - norm_x_min, norm_y_max - norm_y_min],
                    "area": (norm_x_max - norm_x_min) * (norm_y_max - norm_y_min),
                    "iscrowd": 0,
                    "score": 1.0  # 기본 점수, 필요 시 조정 가능
                })
                annotation_id += 1

    return coco_data

def perform_wbf_with_filtering(coco_data, image_sizes, iou_thr=0.5, skip_box_thr=0.0, min_area_thr=0.005, max_area_thr=0.99, min_num_thr=2, max_num_thr=1):
    """
    Perform Weighted Boxes Fusion with small box filtering and noise filtering.

    Args:
        coco_data (dict): COCO formatted data with annotations.
        image_sizes (dict): Mapping from unique image filename to (width, height).
        iou_thr (float): IOU threshold for WBF.
        skip_box_thr (float): Boxes with scores lower than this threshold will be skipped.
        min_area_thr (float): Minimum area overlap ratio to consider as overlapping.
        max_area_thr (float): Maximum area overlap ratio to consider as noise.
        min_num_thr (int): Minimum number of overlapping boxes to consider.
        max_num_thr (int): Maximum number of overlapping boxes allowed as noise.

    Returns:
        dict: Final ensemble results in UFO format.
    """
    final_results = {"images": {}}

    # Organize annotations by image
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Iterate over each image and perform WBF with filtering
    for img_info in tqdm(coco_data["images"], desc="Performing WBF with Filtering"):
        image_id = img_info["id"]
        image_name = img_info["file_name"]
        width, height = img_info["width"], img_info["height"]

        if image_id not in annotations_by_image:
            continue  # No annotations for this image

        anns = annotations_by_image[image_id]
        boxes = []
        scores = []
        labels = []

        # Collect all boxes for WBF
        for ann in anns:
            bbox = ann["bbox"]  # [x_min, y_min, width, height] normalized
            score = ann.get("score", 1.0)
            label = ann["category_id"]

            # Convert bbox to [x_min, y_min, x_max, y_max]
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h

            # Denormalize coordinates
            x_min_abs = x_min * width
            y_min_abs = y_min * height
            x_max_abs = x_max * width
            y_max_abs = y_max * height

            # Normalize again for WBF as required by ensemble_boxes
            boxes.append([x_min_abs / width, y_min_abs / height, x_max_abs / width, y_max_abs / height])
            scores.append(score)
            labels.append(label)

        if not boxes:
            continue  # No valid boxes to process

        # Perform initial WBF to get preliminary boxes
        # Note: This step can be adjusted based on specific requirements
        # Here, we proceed to apply custom filtering before WBF

        # Apply small box filtering and noise filtering
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for i in range(len(boxes)):
            overlap_count = 0
            noise_count = 0
            box1 = boxes[i]

            for j in range(len(boxes)):
                if i == j:
                    continue
                box2 = boxes[j]
                overlap = intersection_area(box1, box2)
                if overlap > min_area_thr:
                    overlap_count += 1
                if overlap > max_area_thr:
                    noise_count += 1

            if overlap_count >= min_num_thr and noise_count <= max_num_thr:
                filtered_boxes.append(boxes[i])
                filtered_scores.append(scores[i])
                filtered_labels.append(labels[i])

        if not filtered_boxes:
            continue  # No boxes left after filtering

        # Perform WBF on filtered boxes
        boxes_fusion, scores_fusion, labels_fusion = weighted_boxes_fusion(
            [filtered_boxes],
            [filtered_scores],
            [filtered_labels],
            weights=None,  # Scores already reflect weights
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )

        # Convert back to absolute coordinates and save
        word_annotations = {}
        for idx, box in enumerate(boxes_fusion):
            x1, y1, x2, y2 = box
            x1_abs = x1 * width
            y1_abs = y1 * height
            x2_abs = x2 * width
            y2_abs = y2 * height
            point = [
                [x1_abs, y1_abs],
                [x2_abs, y1_abs],
                [x2_abs, y2_abs],
                [x1_abs, y2_abs]
            ]
            word_annotations[str(idx)] = {'points': point}

        final_results["images"][image_name] = {"words": word_annotations}

    return final_results

def perform_ensemble_with_filtering(args, folds=5):
    # Load all JSON files
    print("Loading JSON files...")
    all_json_data = load_json_files(args.json_files)
    print(f"Loaded {len(all_json_data)} JSON files.")

    # Get list of test images and their sizes
    print("Retrieving test image sizes...")
    image_sizes = get_image_sizes(args.test_img_dirs)
    print(f"Found {len(image_sizes)} test images.")

    # Validate weights
    num_folds = folds
    if len(args.weights) != num_folds:
        raise ValueError(f"Number of weights ({len(args.weights)}) does not match number of folds ({num_folds}).")

    # Extract annotations and convert to COCO format
    print("Extracting annotations and converting to COCO format...")
    coco_data = extract_annotations_from_json(
        all_json_data,
        image_sizes,
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area
    )
    print(f"Total annotations after initial filtering: {len(coco_data['annotations'])}")

    # Perform WBF with additional filtering
    print("Performing Weighted Boxes Fusion (WBF) with Filtering...")
    final_ensemble = perform_wbf_with_filtering(
        coco_data,
        image_sizes,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        min_area_thr=0.005,  # Example threshold, adjust as needed
        max_area_thr=0.99,   # Example threshold, adjust as needed
        min_num_thr=args.min_num_thr,
        max_num_thr=args.max_num_thr
    )

    # Convert final ensemble to UFO format
    print("Converting final ensemble results to UFO format...")
    # Here, final_ensemble is already in UFO-like format

    # Save ensemble results in UFO format
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = osp.join(args.save_dir, 'ensemble.json')
    with open(output_path, 'w') as f:
        json.dump(final_ensemble, f, indent=4)
    print(f"Ensemble results saved to {output_path}")

def main():
    args = parse_args()
    perform_ensemble_with_filtering(args, folds=5)

if __name__ == '__main__':
    main()

# 실행 방법
# python csv_to_coco_wbf_ensemble.py \
#     --csv_files ./predictions/output_latest_relabel_custom_fold0.csv \
#                ./predictions/output_latest_relabel_custom_fold1.csv \
#                ./predictions/output_latest_relabel_custom_fold2.csv \
#                ./predictions/output_latest_relabel_custom_fold3.csv \
#                ./predictions/output_latest_relabel_custom_fold4.csv \
#     --test_img_dirs ./data/chinese_receipt/img/test \
#                    ./data/japanese_receipt/img/test \
#                    ./data/thai_receipt/img/test \
#                    ./data/vietnamese_receipt/img/test \
#     --save_dir ./ensemble_results \
#     --weights 0.7 1.0 0.6 0.8 0.9 \
#     --iou_thr 0.5 \
#     --skip_box_thr 0.0 \
#     --min_box_area 0.0005 \
#     --max_box_area 0.99 \
#     --min_num_boxes 2 \
#     --max_num_noise 1

#  python to_coco_wbf_ensemble.py     --csv_files ./predictions/output_latest_relabel_custom_fold0.csv                ./predictions/output_latest_relabel_custom_fold1.csv                ./predictions/output_latest_relabel_custom_fold2.csv                ./predictions/output_latest_relabel_custom_fold3.csv                ./predictions/output_latest_relabel_custom_fold4.csv     --test_img_dirs ./data/chinese_receipt/img/test                    ./data/japanese_receipt/img/test                    ./data/thai_receipt/img/test                    ./data/vietnamese_receipt/img/test     --save_dir ./ensemble_results     --weights 0.7 1.0 0.6 0.8 0.9     --iou_thr 0.5     --skip_box_thr 0.0     --min_box_area 0.0005     --max_box_area 0.99     --min_num_boxes 2     --max_num_noise 1