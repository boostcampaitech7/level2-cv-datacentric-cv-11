import json
import glob
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

#이미지, 글자 masking, 영수증 masking 생성
def synthetic_preprocess(img_file, json_file):
    img = cv2.imread(img_file)
    image_name = img_file.split('/')[-1]
    points_list = []
    for word in json_file['images'][image_name]['words'].keys():
        points = json_file['images'][image_name]['words'][word]['points']
        points_list.append(points)

    x_lim = np.clip(np.array(points_list).reshape(-1,2)[:,0],a_min=0,a_max=img.shape[1])
    y_lim = np.clip(np.array(points_list).reshape(-1,2)[:,1],a_min=0,a_max=img.shape[0])

    xy_lim = np.stack([x_lim,y_lim],axis=1)
    poly_list = xy_lim.reshape(-1,4,2)

    x2 = xy_lim.max(axis=0)[0]
    y2 = xy_lim.max(axis=0)[1]
    x1 = xy_lim.min(axis=0)[0]
    y1 = xy_lim.min(axis=0)[1]

    image_json = json_file['images'][image_name]
    return img, poly_list, [x1,y1,x2,y2], image_json

# 글자 마스킹을 영수증 배경에 안착시키는 과정
def matching_receipts(img_background,img_document,mask):
    img1, points_list1, receipt_point1,json1 = img_document
    img2, points_list2, receipt_point2,json2 = img_background

    x2_1 = receipt_point2[0] * img1.shape[1] / img2.shape[1]
    y2_1 = receipt_point2[1] * img1.shape[0] / img2.shape[0]
    x2_2 = receipt_point2[2] * img1.shape[1] / img2.shape[1]
    y2_2 = receipt_point2[3] * img1.shape[0] / img2.shape[0]

    x1_1 = receipt_point1[0]
    x1_2 = receipt_point1[2]
    y1_1 = receipt_point1[1]
    y1_2 = receipt_point1[3]

    forward_width = x1_2 - x1_1
    forward_height = y1_2 - y1_1

    back_width = x2_2 - x2_1
    back_height = y2_2 - y2_1

    x_var = np.array([[1,0]]*4)
    y_var = np.array([[0,1]]*4)

    if back_width - forward_width < 0 or back_height - forward_height < 0 :
        scale_factor = min(back_width/forward_width,back_height/forward_height)
        scaling_matrix = np.float32([[scale_factor,0,0],[0,scale_factor,0],[0,0,1]])

        rows, cols = img1.shape[:2]
        scaled_size = (int(cols * scale_factor), int(rows * scale_factor))
        img1 = cv2.warpPerspective(img1,scaling_matrix,(img1.shape[1],img1.shape[0]))
        mask = cv2.warpPerspective(mask,scaling_matrix,(img1.shape[1],img1.shape[0]))

        for i in json1['words']:
            json1['words'][i]['points'] = (np.array(json1['words'][i]['points']) * scale_factor).tolist()

        x1_1 = receipt_point1[0] * scale_factor
        x1_2 = receipt_point1[2] * scale_factor
        y1_1 = receipt_point1[1] * scale_factor
        y1_2 = receipt_point1[3] * scale_factor
    
    if  x2_1 - x1_1 > 0 or x1_2 - x2_2 > 0:
        move = x2_1 - x1_1
        translation_matrix = np.float32([[1,0,move],
                                        [0,1,0],
                                        [0,0,1]])
        img1 = cv2.warpPerspective(img1,translation_matrix,(img1.shape[1],img1.shape[0]))
        mask = cv2.warpPerspective(mask,translation_matrix,(img1.shape[1],img1.shape[0]))

        for i in json1['words']:
            json1['words'][i]['points'] = (np.array(json1['words'][i]['points']) + x_var * move).tolist()
        

    if  y2_1 - y1_1 > 0 or y1_2 - y2_2 > 0:
        move = y2_1 - y1_1
        translation_matrix = np.float32([[1,0,0],
                                        [0,1,move],
                                        [0,0,1]])
        img1 = cv2.warpPerspective(img1,translation_matrix,(img1.shape[1],img1.shape[0]))
        mask = cv2.warpPerspective(mask,translation_matrix,(img1.shape[1],img1.shape[0]))

        for i in json1['words']:
            json1['words'][i]['points'] = (np.array(json1['words'][i]['points']) + y_var * move).tolist()

    return img1, json1, mask

#합성 이미지 생성
def synthtic_receipts(background,receipt):
    paper_image = background[0]
    receipt_image = receipt[0]
    
    receipt_mask = np.zeros(receipt_image.shape[:2])
    paper_mask = np.zeros(paper_image.shape[:2])

    img_mask = cv2.fillPoly(receipt_mask,np.int32(receipt[1]),1).copy()
    back_mask = cv2.fillPoly(paper_mask,np.int32(background[1]),1).copy()

    receipt_image, receipt_json, img_mask = matching_receipts(background, receipt, img_mask)

    paper_image = cv2.resize(paper_image,(receipt_image.shape[1],receipt_image.shape[0]))
    back_mask = cv2.resize(back_mask,(img_mask.shape[1],img_mask.shape[0]))

    img_masks = (cv2.blur(paper_image, (99, 99)) * np.expand_dims(back_mask,axis=-1) + paper_image* (np.expand_dims(1-back_mask,axis=-1))) * np.expand_dims(1- img_mask,axis=-1) + receipt_image * np.expand_dims(img_mask,axis=-1)
    return img_masks, receipt_json

# 시각화
def visualize_img(img):
    img = img.astype('uint8')
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cmap='gray')
    plt.show()

def main():
    languages = ['chinese','japanese','thai','vietnamese']

    for lang in languages:
        data_path = '/data/ephemeral/home/code/data/'
        output_dir = data_path + lang + '_receipt/img/train'
        json_path = data_path + lang +'_receipt/ufo/' + 'train.json'
        image_files = glob.glob(data_path + lang +'_receipt/img/' + 'train' + '/*.jpg')
        new_json={'images':[]}
        new_json_path = data_path + lang +'_receipt/ufo/' + 'synthetic.json'

        with open(json_path, 'r') as json_file:
            jf = json.load(json_file)
        change_language = languages.copy()

        for id in range(len(image_files)):
            img = cv2.imread(image_files[id])
            image_name = image_files[id].split('/')[-1]
            j_file = jf['images'][image_name]

            try: 
                change_language.remove(lang)
            except:
                pass
            back_lang = random.choice(change_language)
            
            img_background = synthetic_preprocess(image_files[id],jf)
            img_document = synthetic_preprocess(image_files[id],jf)

            image, json_f = synthtic_receipts(img_background,img_document)

            # 이미지 저장 (필요한 경우 주석 해제)
            output_path = os.path.join(output_dir, f'output_{id}.jpg')
            cv2.imwrite(output_path, image)

            new_json['images'].append({'output_'+str(id)+'.jpg':json_f})

            with open(new_json_path, 'w') as json_file:
                json.dump(new_json, json_file, indent=4)

        print(lang + " finish")
    print("All images processed")

if __name__ == '__main__':
    main()