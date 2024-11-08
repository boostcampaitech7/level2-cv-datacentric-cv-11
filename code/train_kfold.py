import os
import os.path as osp
import time
from datetime import timedelta
from argparse import ArgumentParser
import math
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--fold', type=int, default=2)  # 사용할 fold 번호 (0, 1, 2)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def train(train_loader, model, optimizer, scheduler, device, epoch, num_batches):
    model.train()
    epoch_loss = 0
    epoch_start = time.time()

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
                'Cls loss': extra_info['cls_loss'],
                'Angle loss': extra_info['angle_loss'],
                'IoU loss': extra_info['iou_loss']
            }
            pbar.set_postfix(val_dict)

    return epoch_loss / num_batches


def validate(valid_loader, model, device):
    model.eval()
    total_loss = 0
    num_batches = len(valid_loader)

    with torch.no_grad():
        for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
            loss, _ = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            total_loss += loss.item()

    return total_loss / num_batches


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, fold):
    # 학습 데이터셋 준비
    train_dataset = SceneTextDataset(
        data_dir,
        split=f'train{fold}',  # train0, train1, train2 중 하나 사용
        image_size=image_size,
        crop_size=input_size,
    )
    train_dataset = EASTDataset(train_dataset)

    # 검증 데이터셋 준비
    valid_dataset = SceneTextDataset(
        data_dir,
        split=f'valid{fold}',  # valid0, valid1, valid2 중 하나 사용
        image_size=image_size,
        crop_size=input_size,
    )
    valid_dataset = EASTDataset(valid_dataset)

    # 데이터로더 준비
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 모델 초기화
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)

    # Optimizer와 Scheduler 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    num_batches = math.ceil(len(train_dataset) / batch_size)
    best_val_loss = float('inf')

    # 학습 시작
    for epoch in range(max_epoch):
        epoch_start = time.time()

        # Train
        train_loss = train(train_loader, model, optimizer, scheduler, device, epoch, num_batches)

        # Validate
        val_loss = validate(valid_loader, model, device)

        scheduler.step()

        print('Epoch {}: Train Loss: {:.4f} | Val Loss: {:.4f} | Elapsed time: {}'.format(
            epoch + 1, train_loss, val_loss, timedelta(seconds=time.time() - epoch_start)))

        # 모델 저장
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'ver3_aug_sharp_fold{fold}_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, ckpt_fpath)

        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = osp.join(model_dir, f'ver3_aug_sharp_fold{fold}_best.pth')
            torch.save({
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
