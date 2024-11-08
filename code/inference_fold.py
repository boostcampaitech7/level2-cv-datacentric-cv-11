import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda:1' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Ensure output directory exists
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    # Iterate over each fold checkpoint
    for fold in range(5):
        ckpt_filename = f'latest_fold{fold}.pth'
        ckpt_fpath = osp.join(args.model_dir, ckpt_filename)

        if not osp.exists(ckpt_fpath):
            print(f"Checkpoint file does not exist: {ckpt_fpath}. Skipping this fold.")
            continue

        output_fname = f'output{fold}.csv'  # As per user request; consider using .json if appropriate
        output_fpath = osp.join(args.output_dir, output_fname)

        print(f'\nProcessing Fold {fold}:')
        print(f'Loading checkpoint: {ckpt_fpath}')

        # Perform inference
        ufo_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                  args.batch_size, split='test')

        # Save the result to the corresponding output file
        with open(output_fpath, 'w') as f:
            json.dump(ufo_result, f, indent=4)
        
        print(f'Saved inference results to {output_fpath}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
