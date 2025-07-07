import torch
import pytorch_lightning as pl
from loss import CustomDiceKLLoss
from utils.metrics import IoU
import os
import torchvision.utils as vutils

class RoadSegmentationModule(pl.LightningModule):
    def __init__(self, net, optimizer_params, dataset, test_output_dir=None):
        super(RoadSegmentationModule, self).__init__()
        self.net = net
        self.dataset = dataset
        self.loss_fn = CustomDiceKLLoss(num_classes=2)
        self.metrics = IoU(threshold=0.5)  # 假设这个类已包含CONN计算
        self.optimizer_params = optimizer_params
        self.total_steps = None  # 将在训练开始时定义
        self.test_output_dir = test_output_dir  # 新增，用于指定测试结果保存路径

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if self.total_steps is None:
            self.total_steps = self.trainer.estimated_stepping_batches

        # img, mask = batch
        img, mask, mask_filenames = batch
        m, alpha = self(img)
        loss = self.loss_fn(m, mask, alpha, self.current_epoch, self.trainer.max_epochs)

        output = m[:, 1, :, :].unsqueeze(1)
        metrics = self.metrics(output, mask)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_iou', metrics['iou'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_precision', metrics['precision'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('train_recall', metrics['recall'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('train_f1', metrics['f1'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('train_conn', metrics['conn'], prog_bar=False, on_step=False, on_epoch=True)  # 新增CONN指标

        return loss

    def validation_step(self, batch, batch_idx):
        # img, mask = batch
        img, mask, mask_filenames = batch
        m, alpha = self(img)
        loss = self.loss_fn(m, mask, alpha, self.current_epoch, self.trainer.max_epochs)

        output = m[:, 1, :, :].unsqueeze(1)
        metrics = self.metrics(output, mask)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_iou', metrics['iou'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_precision', metrics['precision'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_recall', metrics['recall'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_f1', metrics['f1'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('val_conn', metrics['conn'], prog_bar=False, on_step=False, on_epoch=True)  # 新增CONN指标

        return loss

    def test_step(self, batch, batch_idx):
        img, mask, mask_filenames = batch  # 解包 mask 文件名
        m, alpha = self(img)
        loss = self.loss_fn(m, mask, alpha, self.current_epoch, self.trainer.max_epochs)

        output = m[:, 1, :, :].unsqueeze(1)
        metrics = self.metrics(output, mask)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_iou', metrics['iou'], prog_bar=True)
        self.log('test_precision', metrics['precision'], prog_bar=False)
        self.log('test_recall', metrics['recall'], prog_bar=False)
        self.log('test_f1', metrics['f1'], prog_bar=False)
        self.log('test_conn', metrics['conn'], prog_bar=True)  # 新增CONN指标，使用进度条显示

        # 保存测试结果的可视化
        if self.test_output_dir is not None:
            os.makedirs(self.test_output_dir, exist_ok=True)
            for i in range(img.size(0)):
                mask_filename = mask_filenames[i]  # 获取当前样本的 mask 文件名
                # 保留扩展名
                new_prefix = f'batch{batch_idx}_img{i}_' + os.path.basename(mask_filename)

                # 处理输入图像（反归一化）
                input_image = img[i] * 0.5 + 0.5
                input_image_path = os.path.join(self.test_output_dir, f'{new_prefix}_input.png')
                vutils.save_image(input_image, input_image_path)

                # 保存真实掩码
                gt_mask = mask[i]
                gt_mask_path = os.path.join(self.test_output_dir, f'{new_prefix}_gt.png')
                vutils.save_image(gt_mask.float(), gt_mask_path)

                # 保存预测掩码（阈值化）
                pred_mask = output[i]
                pred_mask = (pred_mask > 0.5).float()
                pred_mask_path = os.path.join(self.test_output_dir, f'{new_prefix}_pred.png')
                vutils.save_image(pred_mask, pred_mask_path)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.optimizer_params['lr'],
            weight_decay=self.optimizer_params.get('weight_decay', 1e-5)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        return [optimizer], [scheduler]