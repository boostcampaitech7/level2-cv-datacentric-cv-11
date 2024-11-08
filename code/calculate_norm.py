import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
import argparse

def calculate_norm(img_list):

    mean_ = np.array([np.mean(x, axis=(0, 1)) for x in tqdm(img_list, ascii=True)])
    mean_r = mean_[..., 0].mean() / 255.0
    mean_g = mean_[..., 1].mean() / 255.0
    mean_b = mean_[..., 2].mean() / 255.0


    std_ = np.array([np.std(x, axis=(0, 1)) for x in tqdm(img_list, ascii=True)])
    std_r = std_[..., 0].mean() / 255.0
    std_g = std_[..., 1].mean() / 255.0
    std_b = std_[..., 2].mean() / 255.0
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def main():
    parser = argparse.ArgumentParser(description='Calculate mean and standard deviation of images in a directory.')
    parser.add_argument('data_dir', type=str, help='Directory containing the images')

    args = parser.parse_args()

    img_path = glob.glob(os.path.join(args.data_dir, '*.jpg'))

    img_list = []
    for m in img_path:
        img = Image.open(m)

        assert img.mode == 'RGB'
        
        img = np.array(img)
        img_list.append(img)

    mean, std = calculate_norm(img_list)
    print("Mean: ", mean)
    print("Standard Deviation: ", std)

if __name__ == "__main__":
    main()