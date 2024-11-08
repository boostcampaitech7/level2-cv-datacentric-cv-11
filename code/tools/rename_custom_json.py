import os
import json
import re
import shutil

def backup_file(original_path, backup_path):
    """
    원본 파일을 백업 경로로 복사합니다.
    """
    shutil.copy2(original_path, backup_path)
    print(f"백업 생성 완료: {backup_path}")

def update_image_filenames(json_path, prefix_old='image_', prefix_new='image.cu.'):
    """
    JSON 파일 내의 이미지 파일 이름을 업데이트합니다.
    

    Args:
        json_path (str): 업데이트할 JSON 파일의 경로.
        prefix_old (str): 기존 파일명 접두사. 기본값은 'image_'.
        prefix_new (str): 새로운 파일명 접두사. 기본값은 'image.cu.'.
    """
    # 백업 파일 경로 설정
    backup_path = json_path + '.backup'
    backup_file(json_path, backup_path)
    
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 'images' 키가 있는지 확인
    if 'images' not in data:
        print(f"'images' 키가 JSON 파일에 존재하지 않습니다: {json_path}")
        return
    
    updated_images = {}
    pattern = re.compile(rf'^{re.escape(prefix_old)}(\d+)\.jpg$', re.IGNORECASE)
    
    for image_name, image_info in data['images'].items():
        match = pattern.match(image_name)
        if match:
            number = match.group(1)
            new_image_name = f"{prefix_new}{number}.jpg"
            updated_images[new_image_name] = image_info
            print(f"'{image_name}' -> '{new_image_name}'")
        else:
            # 기존 형식과 일치하지 않는 파일명은 그대로 유지
            updated_images[image_name] = image_info
            print(f"형식 일치하지 않아 변경하지 않음: {image_name}")
    
    # 'images' 키 업데이트
    data['images'] = updated_images
    
    # 업데이트된 JSON 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"JSON 파일 업데이트 완료: {json_path}")

def main():
    # JSON 파일 경로 설정
    json_dir = '/home/naring/workplace/code/data/custom_receipt/ufo'
    json_filename = 'val.json'
    json_path = os.path.join(json_dir, json_filename)
    
    # JSON 파일이 존재하는지 확인
    if not os.path.exists(json_path):
        print(f"JSON 파일이 존재하지 않습니다: {json_path}")
        return
    
    # 이미지 파일명 업데이트
    update_image_filenames(json_path)

if __name__ == "__main__":
    main()
