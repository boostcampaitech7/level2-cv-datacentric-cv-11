# 🔤 OCR Data-Centric 대회

## 🙂 팀 소개

## Members 
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/naringles"><img height="110px"  src="https://avatars.githubusercontent.com/u/61579399?v=4"></a>
            <br/>
            <a href="https://github.com/naringles"><strong>임동훈</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/hanseungsoo13"><img height="110px"  src="https://avatars.githubusercontent.com/u/75753717?v=4"/></a>
            <br/>
            <a href="https://github.com/hanseungsoo13"><strong>한승수</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Jeong-AYeong"><img height="110px"  src="https://avatars.githubusercontent.com/u/87751593?v=4"/></a>
            <br/>
            <a href="https://github.com/Jeong-AYeong"><strong>정아영</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Ai-BT"><img height="110px" src="https://avatars.githubusercontent.com/u/97381138?v=4"/></a>
            <br />
            <a href="https://github.com/Ai-BT"><strong>김대환</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/cherry-space"><img height="110px" src="https://avatars.githubusercontent.com/u/177336350?v=4"/></a>
            <br />
            <a href="https://github.com/cherry-space"><strong>김채리</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/SkyBlue-boy"><img height="110px"  src="https://avatars.githubusercontent.com/u/63849988?v=4"/></a>
              <br />
              <a href="https://github.com/SkyBlue-boy"><strong>박윤준</strong></a>
              <br />
          </td>
    </tr>
</table>

##  Roles

|Name|Roles|
|:-------:|:--------------------------------------------------------------:|
|임동훈| KFold, Ensemble, Augmentation
|한승수| EDA, Augmentation
|정아영| Utilizing external datasets(Sroie)
|김대환| Utilizing external datasets(CORD)
|김채리| EDA, Re-Labeling
|박윤준| EDA, Re-Labeling

## 🎙️ 프로젝트 소개
|일본어|중국어|베트남어|태국어|
|:----:|:----:|:----:|:----:|
|![image](https://github.com/user-attachments/assets/4d5c2aca-2156-4f26-925a-a924ebcbb70d) |![image](https://github.com/user-attachments/assets/71fcb327-1502-407d-9d56-467ec07994a0) |![image](https://github.com/user-attachments/assets/dfaeb40b-c1a1-450b-ac69-826f9247ca28) |![image](https://github.com/user-attachments/assets/bbaafe2f-b63f-4375-9a76-17f7de0b7b39) |

OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술입니다. 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되거나 주차장에 들어가면 차량 번호가 자동으로 인식되는 등 일상생활에 이미 보편적으로 사용되고 있습니다. 이번 대회는 OCR의 대표적인 model 중 하나인 EAST model을 활용하여 진료비 계산서 영수증안에 있는 글자를 인식하는 대회입니다. 

이번 대회는  Data-Centric 대회로 다음과 같은 제약사항이 있습니다. 

- 대회에서 주어지는 EAST model만을 사용해야 하며 model과 관련된 코드를 바꿔서는 안됩니다.
- 이미지넷 기학습 가중치 외에는 사용이 불가합니다.

즉 이번 대회는 모델을 고정한 상태로 데이터만을 활용하여 OCR model의 성능을 최대한 끌어 올리는 프로젝트 입니다. 

이번 대회는 `부스트캠프 AI Tech` CV 트랙내에서 진행된 대회이며 F1-Score로 최종평가를 진행하였습니다. 

## 📆 프로젝트 일정

프로젝트 전체 일정

- 2024.10.28 ~ 2024.11.07

프로젝트 세부 일정

- 2024.10.28 ~ 2024.10.29 : OCR에 대해 알아보기, EDA
- 2024.10.30 ~ 2024.11.01 : Train dataset과 Validation dataset 분리, Baseline 고도화
- 2024.11.02 ~ 2024.11.03 : 합성데이터 제작, Cord 등 외부 데이터 수집
- 2024.11.04 ~ 2024.11.05 : Random Augmentation 실험, Re-Labelling 실험
- 2024.11.06 ~ 2024.11.07 : Ensemble


## 🤔 Wrap-Up Report
링크에 들어가시면 프로젝트에 대한 랩업리포트를 확인할 수 있습니다.
- [Wrap-Up Report](https://cactus-panama-b7c.notion.site/Data-Centric-OCR-Wrap-up-Report-54123f2d7cc2497d9e31b9f0619ea356?pvs=4)
  
## 🗒️ 프로젝트 결과

- 프로젝트 결과 최종적으로 아래와 같은 결과를 얻었습니다. (Public 11/23등, Private 11/23등)
    - Public
    
    ![public](https://github.com/user-attachments/assets/7f28fe34-acaf-48c9-bbbc-e05ba47e54ce)

    
    - Private
    
    ![private](https://github.com/user-attachments/assets/9a71d007-ad66-4e06-97a3-010ec64e5883)
    

# 🔄️ Directory

```
├── README.md
├── Visualize
│   ├── bbox_viewer.ipynb
│   ├── ensem_hyp_compare.ipynb
│   ├── synthetic_visualize.ipynb
│   └── visualize.ipynb
├── dataset
│   ├── cord
│   │   ├── 01_convert.py
│   │   ├── 02_json_to_coco.py
│   │   ├── 03_coco_to_ufo.py
│   │   ├── rename_custom_images.py
│   │   └── rename_custom_json.py
│   ├── kfold
│   │   ├── create_kfold_json.py
│   │   └── split_train_val.py
│   ├── relabelling
│   │   ├── COCO_to_UFO.py
│   │   └── UFO_to_COCO.py
│   └── synthetic
│       └── synthetic_data.py
├── requirements.txt
├── src
│   ├── dataset_add_custom.py
│   ├── ensemble.py
│   ├── inference.py
│   └── train.py
├── tree.txt
└── utils
    ├── RandAugment.py
    ├── calculate_norm.py
    ├── deteval.py
    ├── ensemble_wbf.py
    └── save_bbox.ipynb
```
- 베이스라인 모델인 EAST 모델이 정의되어 있는 `model.py`, `loss.py`, `east_dataset.py`, `detect.py` 파일은 변경하지 않았으므로 업로드하지 않았습니다.
- `Visualize`: 시각화를 위한 코드입니다. 예측 결과, 앙상블 결과, 합성데이터를 시각화 합니다.
- `dataset`: 학습데이터를 구축하기 위한 폴더입니다.
    | File(.ipynb/.py) | Description |
    | --- | --- |
    | cord | Cord 데이터셋을 활용하기 위한 폴더입니다. |
    | kfold | 데이터셋을 K-Fold로 나누어 저장합니다.  |
    | relabelling  | 기존 데이터셋에 relabelling을 적용하기 위해 라벨의 유형을 바꿉니다.  |
    | synthetic  | 기존 데이터셋을 활용해 이미지와 bbox를 합성한 합성 데이터셋을 생성합니다.  |
- `src`: `train`, `inference`, `ensemble`을 위한 코드들입니다.
- `utils`: 학습과정에서 필요한 기능들입니다.
    | File(.ipynb/.py) | Description |
    | --- | --- |
    | RandAugment | 학습데이터에 random Augmentation을 적용합니다. |
    | calculate norm | 이미지 정규화를 위한 평균과 표준편차를 계산합니다. |
    | deteval  | 평가 metric을 구현한 코드입니다.  |
    | ensemble_wbf  | wbf 알고리즘으로 앙상블을 계산합니다.  |
    | save bbox | 예측한 bbox를 이미지로 저장하는 코드입니다. |

# ⚠️ Dataset 출처

- 대회에서 사용된  `부스트캠프 AI Tech`임을 알려드립니다.
