import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        # y_pred: (B, 1, H, W), y_true: (B, 1, H, W)
        if y_true.shape[2:] != y_pred.shape[2:]:
            y_true = F.interpolate(y_true.float(), size=y_pred.shape[2:], mode='nearest')

        eps = 1e-6
        y_pred = (y_pred > self.threshold).float()
        y_true = y_true.float()

        intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
        union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + eps) / (union + eps)
        precision = intersection / (y_pred.sum(dim=(1, 2, 3)) + eps)
        recall = intersection / (y_true.sum(dim=(1, 2, 3)) + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # 计算CONN指标
        conn = self._calculate_conn(y_pred, y_true)

        metrics = {
            "iou": iou.mean().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": f1.mean().item(),
            "conn": conn.mean().item()  # 添加CONN指标
        }
        return metrics

    def _calculate_conn(self, y_pred, y_true):
        """计算连通性指标(CONN)"""
        batch_size = y_pred.size(0)
        conn_values = []

        for i in range(batch_size):
            # 将预测和真实掩码转换为numpy数组进行连通区域分析
            pred_mask = y_pred[i, 0].cpu().numpy() > 0.5
            true_mask = y_true[i, 0].cpu().numpy() > 0.5

            # 计算预测掩码中的连通组件
            pred_labels, num_pred = ndimage.label(pred_mask)

            # 计算真实掩码中的连通组件
            true_labels, num_true = ndimage.label(true_mask)

            # 计算交集区域的连通组件
            intersection_mask = pred_mask & true_mask
            intersection_labels, num_intersection = ndimage.label(intersection_mask)

            # 计算CONN指标
            if num_pred == 0 or num_true == 0:
                conn = 0.0
            else:
                conn = 2 * num_intersection / (num_pred + num_true)

            conn_values.append(torch.tensor(conn, device=y_pred.device))

        return torch.stack(conn_values)