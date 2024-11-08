import json
import os
import os.path as osp
from sklearn.model_selection import KFold
import argparse

def main():
    parser = argparse.ArgumentParser(description="KFold split of image data for multiple languages")
    parser.add_argument('--seed', type=int, default=137, help="Random seed for KFold")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds for KFold")
    parser.add_argument('--root_dir', type=str, default='./data', help="Root directory of the dataset")

    args = parser.parse_args()

    seed = args.seed
    folds = args.folds
    root_dir = args.root_dir

    # 언어별 폴더 처리
    languages = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt']
    # languages = ['custom_receipt']
    
    for language in languages:
        language_dir = osp.join(root_dir, language)
        json_file = osp.join(language_dir, 'ufo_relabel_ver3/train.json')
        
        # JSON 파일 로드
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 이미지 목록을 가져옴
        image_list = list(data['images'].items())

        # KFold 인스턴스 생성
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

        # 각 fold에 대한 훈련 및 검증 데이터셋 생성
        for fold, (train_idx, val_idx) in enumerate(kf.split(image_list)):
            train_images = dict([image_list[i] for i in train_idx])
            val_images = dict([image_list[i] for i in val_idx])
            
            train_data = {'images': train_images}
            val_data = {'images': val_images}
            
            # JSON 파일 생성
            with open(osp.join(language_dir, f'ufo_relabel_ver3/train{fold}.json'), 'w', encoding='utf-8') as file:
                json.dump(train_data, file, indent=4, ensure_ascii=False)
            with open(osp.join(language_dir, f'ufo_relabel_ver3/valid{fold}.json'), 'w', encoding='utf-8') as file:
                json.dump(val_data, file, indent=4, ensure_ascii=False)
            
            print(f"{language} - number of fold {fold} train images : ", len(train_images))
            print(f"{language} - number of fold {fold} validation images : ", len(val_images))

if __name__ == "__main__":
    main()
