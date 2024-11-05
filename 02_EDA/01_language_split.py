# %%
import json
import os
import re


def split_json_by_language(input_json_path):
    # JSON 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 언어별 데이터를 저장할 딕셔너리 초기화
    language_data = {lang: {"images": {}} for lang in ["zh", "ja", "vi", "th", "image"]}

    pattern = re.compile(r'extractor\.(zh|ja|vi|th)\.')

    language_map = {
        "zh": "chinese",
        "ja": "japanese",
        "vi": "vietnamese",
        "th": "thai",
        "image": "image"
    }

    # 각 이미지의 파일 이름을 확인하고 언어별로 분류
    for image_name, image_data in data["images"].items():
        match = pattern.search(image_name)
        if match:
            lang = match.group(1)
            language_data[lang]["images"][image_name] = image_data
        else:
            # 언어 코드가 없으면 image 그룹에 추가
            language_data["image"]["images"][image_name] = image_data

    # 언어별 JSON 파일로 저장
    for lang, lang_data in language_data.items():
        output_path = f"./val_split/{language_map[lang]}_output.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lang_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {output_path}")


# 예측 json 파일 경로
split_json_by_language('val_data.json')
