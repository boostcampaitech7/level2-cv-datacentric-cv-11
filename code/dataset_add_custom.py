import os.path as osp
import os.path as osp
import math
import json
from PIL import Image, ImageDraw

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from numba import njit
from skimage.util import random_noise

from RandAugment import RandAugment

@njit
def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices

@njit
def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v

@njit
def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

@njit
def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

@njit
def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

@njit
def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                  start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices

@njit
def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices

#rotate_img 수정.
def rotate_img(img, vertices, angle_range=10, angle=None):
    '''rotate image by a specified angle or within a range
    Input:
        img         : PIL Image
        vertices    : vertices of text region <numpy.ndarray, (n,8)>
        angle_range : random rotate range if angle is None
        angle       : specific angle to rotate
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    if angle is None:
        angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, expand=True)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices



def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels

def black_noise_wrapper(amount=0.05):
    def black_noise(image, rows, cols):
        num_noise = np.ceil(amount * rows * cols)

        # random pixel
        coords = [
            np.random.randint(0, i - 1, int(num_noise)) for i in [rows, cols]
        ]
        image[coords[0], coords[1], :] = 0

        return image

    return black_noise

def polygon_noise_wrapper(num_polygons=10000, min_radius=5, max_radius=20):
    def polygon_noise(image, cols, rows):
        image_size = (cols, rows)
        noise_image = Image.new("RGBA", image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(noise_image)

        for _ in range(num_polygons):
            # 중심점 설정
            center_x = np.random.randint(0, image_size[0])
            center_y = np.random.randint(0, image_size[1])

            # 다각형 꼭짓점 생성
            polygon = []
            for _ in range(np.random.randint(3, 6)):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(min_radius, max_radius)
                x = int(center_x + np.cos(angle) * radius)
                y = int(center_y + np.sin(angle) * radius)
                polygon.append((x, y))

            # 투명도 설정
            transparency = np.random.randint(30, 100)
            # 다각형 그리기
            draw.polygon(polygon, fill=(140, 215, 245, transparency))

        original_image = Image.fromarray(image).convert("RGBA")
        composite_image = Image.alpha_composite(original_image, noise_image)

        return np.array(composite_image)[:, :, :3]

    return polygon_noise

def add_pepper(image, p):
    # 이미지가 PIL 형식이라면 numpy 배열로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 랜덤으로 pepper noise 추가
    if np.random.random() < p:
        noise_img = random_noise(image, mode='pepper', amount=0.08)
        noise_img = np.array(255 * noise_img, dtype='uint8')  # noise 이미지 생성
        return noise_img
    else:
        return image


def random_choice_augmentations(probability):
    amount = np.random.uniform(0, 0.1)  # black noise parameter
    num_polygons = np.random.randint(7000, 20000)  # polygon num
    min_radius, max_radius = 5, 20  # polygon size
    random_choice = A.OneOf([
        A.RandomShadow(num_shadows_lower=2, num_shadows_upper=5, always_apply=True),
        A.RandomSnow(p=1),
        A.RandomRain(blur_value=1, always_apply=True),
        A.RandomBrightnessContrast(always_apply=True),
        # A.Lambda(image=black_noise_wrapper(amount), p=1),
        # A.Lambda(image=polygon_noise_wrapper(num_polygons, min_radius, max_radius),p=1),
    ], p=probability)
    return random_choice

def binarization(image, **kwargs):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    return binary_image

def sharpening(image, strength):
    image = image.astype('uint8')
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    b = (1 - strength) / 8
    sharpening_kernel = np.array([[b, b, b],
                                  [b, strength, b],
                                  [b, b, b]])
    kernel = np.ones((3, 3), np.uint8)
    gray_image = cv2.erode(gray_image, kernel, iterations=1)
    output = cv2.filter2D(gray_image, -1, sharpening_kernel)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    return output

class SceneTextDataset(Dataset):
    def __init__(self, root_dir,
                 split='train',
                 relabeled=None,
                 fold_num=None,                         
                 image_size=2048,
                 crop_size=1024,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=True,
                 normalize=True,
                 binarization=False,
                 add_pepper=False,
                 add_sharpening=False,
                 augmentation=False):
        self._lang_list = ['chinese', 'japanese', 'thai', 'vietnamese', 'custom'] ## custom dataset 추가
        self.root_dir = root_dir
        self.split = split
        self.relabeled = relabeled
        total_anno = dict(images=dict())
        for nation in self._lang_list:
            with open(osp.join(root_dir, f'{nation}_receipt/ufo{self.relabeled}/{split}{fold_num}.json'), 'r', encoding='utf-8') as f:
                anno = json.load(f)
            total_anno['images'].update(anno['images'])

        self.anno = total_anno
        self.image_fnames = sorted(self.anno['images'].keys())

        self.image_size = image_size
        self.crop_size = crop_size
        self.color_jitter = color_jitter
        self.normalize = normalize

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

        # Augmentation flags
        self.binarization = binarization
        self.add_pepper = add_pepper
        self.add_sharpening = add_sharpening
        self.augmentation = augmentation

    def _infer_dir(self, fname):
        lang_indicator = fname.split('.')[1]
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'
        elif lang_indicator == 'cu': ## custom dataset 추가.
            lang = 'custom'            
        else:
            raise ValueError
        # `valid` 모드일 경우 json 경로가 train이기 때문에 이미지 경로를 train으로 설정
        img_dir = 'train' if self.split == 'valid' else self.split
        return osp.join(self.root_dir, f'{lang}_receipt', 'img', img_dir)   
         
        # if self.split=='valid': #valid 모드일 경우 json 경로가 train이기 때문에 이미지 불러오는 경로 train 유지.
        #     return osp.join(self.root_dir, f'{lang}_receipt', 'img', 'train')
        # else:
        #     return osp.join(self.root_dir, f'{lang}_receipt', 'img', self.split)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self._infer_dir(image_fname), image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            num_pts = np.array(word_info['points']).shape[0]
            if num_pts > 4:
                continue
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(1)
        vertices = np.array(vertices, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)

        # 추가된 부분: ±90도 회전을 확률적으로 적용
        if np.random.rand() < 0.5:
            angle = np.random.choice([-90, 90])
            image, vertices = rotate_img(image, vertices, angle=angle)

        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        # # exception
        # if self.augmentation and any([self.binarization, self.color_jitter, self.normalize]):
        #     warnings.warn("Only one of augmentation and others should be declared.")
        #     raise ValueError

        ##RandAugmentation
        transform_train = RandAugment(2,8)
        image = transform_train(image)
        
        funcs = []

        if self.augmentation:
            funcs.append(random_choice_augmentations(probability=1))

        if self.add_pepper:
            image = add_pepper(image, 0.5)
            
        if self.add_sharpening:
            image = sharpening(image, strength=7)

        if self.binarization:
            funcs.append(A.Lambda(image=binarization, p=1.0))

        if self.color_jitter:
            funcs.append(A.ColorJitter())

        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        transform = A.Compose(funcs)
        image = transform(image=image)['image']

        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask