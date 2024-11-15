import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from github_repo.code.dataset.dataset_add_custom import SceneTextDataset
from model import EAST

import wandb #import wandb

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8) #8
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)      
    #fold, relabel 관련 augment 추가.
    parser.add_argument('--fold', type=str, default=None)           #fold 번호 설정 (없을경우 fold 사용 안함)
    parser.add_argument('--cuda', type=str, default=0)              #CUDA 번호 설정
    parser.add_argument('--relabel_tag', type=str, default=None)    #ufo 폴더 뒤에 붙는 태그 (없을경우 */ufo/*)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, fold, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, resume, cuda, relabel_tag):
    # dataset = SceneTextDataset(
    #     data_dir,
    #     split='train',
    #     image_size=image_size,
    #     crop_size=input_size,
    # )
    train_dataset = SceneTextDataset(
        root_dir=data_dir,
        split='train',
        relabeled=relabel_tag, #ufo 폴더 뒤에 붙는 태그
        fold_num=fold,
        image_size=image_size,
        crop_size=input_size,    
        color_jitter=True,      # 색상 변화 적용
        normalize=True,         # 정규화 적용
        binarization=False,      # 이진화 적용
        add_pepper=False,        # 페퍼 노이즈 적용
        add_sharpening=True,    # 샤프닝 적용
        augmentation=False,      # 랜덤한 augmentation 적용
)
    train_dataset = EASTDataset(train_dataset)
    # num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataset = SceneTextDataset(
        data_dir,
        split='valid',
        relabeled=relabel_tag, #ufo 폴더 뒤에 붙는 태그
        fold_num=fold,        
        image_size=image_size,
        crop_size=input_size,
        color_jitter=True,      # 색상 변화 적용
        normalize=True,         # 정규화 적용
        binarization=False,      # 이진화 적용
        add_pepper=False,        # 페퍼 노이즈 적용
        add_sharpening=False,    # 샤프닝 적용
        augmentation=False,      # 랜덤한 augmentation 적용        
    )
    val_dataset = EASTDataset(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )    

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    #이어서 학습하기
    if resume is not None:
        resume_path = osp.join(model_dir,resume)
        model.load_state_dict(torch.load(resume_path,weights_only=True))
        print('model_loaded_successfully!')

    #########################################################################
    #Seungsoo: Connect Wandb
    run = wandb.init(
    # Set the project where this run will be logged
    project="naring-ocr",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": scheduler,
        "epochs": max_epoch,
        "fold": fold  # Fold 번호 기록        
    })

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        num_batches = math.ceil(len(train_dataset) / batch_size)        
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Train Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        ###############################################################
        wandb.log({"Train_mean_loss": epoch_loss / num_batches,
                    "Train loss": epoch_loss, 
                    'Train Cls loss': extra_info['cls_loss'],
                    'Train Angle loss': extra_info['angle_loss'],
                    'Train IoU loss': extra_info['iou_loss']})
        ################################################################      
           
        # Validation Phase
        model.eval()
        val_loss = 0
        num_batches_val = len(val_loader)
        with tqdm(total=num_batches_val, desc="Validation", leave=True) as pbar_val:
            with torch.no_grad():
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    val_loss += loss.item()

                    pbar_val.update(1)
                    val_dict = {
                        'Val Cls loss': extra_info['cls_loss'], 
                        'Val Angle loss': extra_info['angle_loss'], 
                        'Val IoU loss': extra_info['iou_loss']
                    }
                    pbar_val.set_postfix(val_dict)

        print('Validation Mean loss: {:.4f}'.format(val_loss / len(val_loader)))
                
        ###############################################################
        wandb.log({
            "Val_mean_loss": val_loss / len(val_loader),
            "Val loss": val_loss, 
            'Val Cls loss': extra_info['cls_loss'],
            'Val Angle loss': extra_info['angle_loss'],
            'Val IoU loss': extra_info['iou_loss']
        })
        ################################################################        

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'latest_relabel_custom_fold{fold}_1024.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)