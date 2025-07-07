import torch
import time
import torch.nn as nn
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from networks.edl_cafm import EDL_CAFM
from utils.datasets import prepare_Beijing_dataset, prepare_TLCGIS_dataset

from networks.CMMPNet import CMMPNet
from model.CMNEXT.CMNEXT import CMNEXT
from framework import RoadSegmentationModule

def get_model(model_name):
    if model_name == 'EDL':
        model = CMMPNet(block_size='1,2,4', num_classes=2)
    elif model_name == 'CMNEXT':
        model = CMNEXT('B2', num_classes=1)
    elif model_name == 'EDL_CAFM':
        model = EDL_CAFM(block_size='1,2,4', num_classes=2)
    else:
        print("[ERROR] cannot find model ", model_name)
        assert (False)
    return model

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.encoding = 'utf-8'

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_dataloader(args):
    if args.dataset == 'BJRoad':
        train_ds, val_ds, test_ds = prepare_Beijing_dataset(args)
    elif args.dataset == 'TLCGIS' or args.dataset.find('Porto') >= 0:
        train_ds, val_ds, test_ds = prepare_TLCGIS_dataset(args)
    else:
        print("[ERROR] cannot find dataset ", args.dataset)
        assert (False)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        drop_last=False
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False
    )
    return train_dl, val_dl, test_dl

def train_val_test(args):

    seed_everything(args.random_seed)

    net = get_model(args.model)
    optimizer_params = {'lr': args.lr, 'weight_decay': 1e-5}

    # 如果是仅测试模式，加载模型
    if args.test_only:
        model = RoadSegmentationModule.load_from_checkpoint(
            args.ckpt_path,
            net=net,
            optimizer_params=optimizer_params,
            dataset=args.dataset,
            test_output_dir=WEIGHT_SAVE_DIR  # 指定测试结果保存路径，与 WEIGHT_SAVE_DIR 保持一致
        )
    else:
        model = RoadSegmentationModule(
            net,
            optimizer_params,
            dataset=args.dataset,
            test_output_dir=WEIGHT_SAVE_DIR  # 在训练过程中也指定测试结果保存路径
        )

    train_dl, val_dl, test_dl = get_dataloader(args)

    logger = TensorBoardLogger(save_dir=WEIGHT_SAVE_DIR, name='lightning_logs')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_iou',
        dirpath=WEIGHT_SAVE_DIR,
        filename=f'model-{args.dataset}-{args.random_seed}-{{epoch:02d}}-{{val_iou:.4f}}',
        save_top_k=1,
        mode='max',
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.use_gpu else 'cpu',
        devices=args.gpu_ids if args.use_gpu else None,
        default_root_dir=WEIGHT_SAVE_DIR,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    if args.test_only:
        # 仅测试模式
        trainer.test(model, dataloaders=test_dl)
    else:
        # 训练和验证
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        # 测试最佳模型
        trainer.test(model, dataloaders=test_dl, ckpt_path='best')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CMMPNet')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sat_dir', type=str, default='')
    parser.add_argument('--mask_dir', type=str, default='')
    parser.add_argument('--gps_dir', type=str, default='')
    parser.add_argument('--test_sat_dir', type=str, default='')
    parser.add_argument('--test_mask_dir', type=str, default='')
    parser.add_argument('--test_gps_dir', type=str, default='')
    parser.add_argument('--lidar_dir', type=str, default='')
    parser.add_argument('--split_train_val_test', type=str, default='')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model/')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=430)
    parser.add_argument('--dataset', type=str, default='TLCGIS')
    parser.add_argument('--down_scale', type=bool, default=False)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Number of batches to accumulate gradients over')
    # 新增参数
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--ckpt_path', type=str, default='', help='Path to the checkpoint file for testing')

    args = parser.parse_args()

    WEIGHT_SAVE_DIR = os.path.join(
        args.weight_save_dir,
        f"{args.model}_{args.dataset}_{args.random_seed}_" + time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) + "/"
    )

    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.makedirs(WEIGHT_SAVE_DIR)
    print("Log dir: ", WEIGHT_SAVE_DIR)

    gpu_ids = [int(id_) for id_ in args.gpu_ids.split(',')]
    args.gpu_ids = gpu_ids

    path = os.path.abspath(os.path.dirname(__file__))
    sys.stdout = Logger(WEIGHT_SAVE_DIR + 'train.log')

    train_val_test(args)
    print("[DONE] finished")

