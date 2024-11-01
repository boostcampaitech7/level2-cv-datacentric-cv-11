import json
from pathlib import Path


def convert_word_ids_to_val_format(val_predict_data, start_index=1):
    updated_data = val_predict_data.copy()

    for image_id, image_data in updated_data['images'].items():
        words = image_data['words']

        # 기존 words 딕셔너리의 모든 word_id를 새로운 형식으로 변환하여 저장
        new_words = {}
        for i, (word_id, word_info) in enumerate(words.items(), start=start_index):
            # 지정된 start_index부터 4자리 형식의 `word_id` 생성 (예: "1" -> "0001")
            new_word_id = f"{i:04d}"
            new_words[new_word_id] = word_info

        # 변환된 words 딕셔너리로 교체
        updated_data['images'][image_id]['words'] = new_words

    return updated_data

# 변환된 데이터를 JSON 파일로 저장하는 함수


def save_converted_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Converted data saved to {output_path}")


# 사용 예시
# JSON 파일 로드
input_path = "./data/vietnamese_receipt/ufo/vietnamese_val_predict.json"
output_path = "./data/vietnamese_receipt/ufo/vietnamese_val_predict_converted.json"

with open(input_path, 'r', encoding='utf-8') as f:
    val_predict_data = json.load(f)

# word_id 변환
val_predict_data_converted = convert_word_ids_to_val_format(val_predict_data, start_index=1)

# 변환된 데이터를 JSON 파일로 저장
save_converted_json(val_predict_data_converted, output_path)
