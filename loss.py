import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDiceKLLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomDiceKLLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, m, y_true, alpha, current_epoch, max_epochs):
        if y_true.shape[2:] != m.shape[2:]:
            y_true = F.interpolate(y_true.float(), size=m.shape[2:], mode='nearest')
        y_true_one_hot = F.one_hot(y_true.squeeze(1).long(), num_classes=self.num_classes)  # (B, H, W, num_classes)
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()  # (B, num_classes, H, W)

        dice_loss = self.dice_loss(m, y_true_one_hot)

        kl_div = self.kl_divergence(alpha)

        annealing_coef = min(1.0, current_epoch / max_epochs)

        # 总损失
        total_loss = dice_loss + annealing_coef * kl_div

        return total_loss

    def dice_loss(self, y_pred, y_true, smooth=1e-6):
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

    def kl_divergence(self, alpha):
        # alpha 的形状为 (B, num_classes, H, W)
        epsilon = 1e-8  # 避免数值不稳定性
        alpha = alpha + epsilon  # 防止 alpha 为零
        beta = torch.ones_like(alpha)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        kl = kl.mean()

        return kl
