import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, effective_num_joints: int = None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, t, t_w):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = t.reshape(pred_jts.shape)
        gt_uv_weight = t_w.reshape(pred_jts.shape)
        
        ################## TODO ###################
        # nf_loss = -(nf_loss)
        ################## TODO ###################

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()
        
class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super(CombinedLoss, self).__init__()
        self.mse_loss = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
        self.rle_loss = RLELoss()
        self.mse_weight = cfg.LOSS.MSE_WEIGHT
        self.rle_weight = cfg.LOSS.RLE_WEIGHT
        self.use_init_heatmap_loss = cfg.LOSS.USE_INIT_HEATMAP_LOSS
        self.init_heatmap_weight = cfg.LOSS.INIT_HEATMAP_WEIGHT
        
    def forward(self, output, mse_target, mse_target_weight,
                rle_target, rle_target_weight):
        loss_mse = self.mse_loss(output[0], mse_target, mse_target_weight)
        loss_rle = self.rle_loss(output[1], rle_target, rle_target_weight)

        if self.use_init_heatmap_loss:
            loss_init_heatmap = self.mse_loss(output[2], mse_target, mse_target_weight)
            total_loss = self.mse_weight * loss_mse + self.rle_weight * loss_rle + self.init_heatmap_weight * loss_init_heatmap
            # total_loss = self.mse_weight * loss_mse + self.init_heatmap_weight * loss_init_heatmap
            return total_loss, loss_mse, loss_rle, loss_init_heatmap
            # return total_loss, loss_mse, loss_init_heatmap
        else:
            total_loss = self.mse_weight * loss_mse + self.rle_weight * loss_rle
            # total_loss = self.mse_weight * loss_mse
            return total_loss, loss_mse, loss_rle

def get_loss_function(cfg, device):
    if cfg.LOSS.NAME == 'JointsMSELoss':
        loss_fn = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
    elif cfg.LOSS.NAME == 'CombinedLoss':
        loss_fn = CombinedLoss(cfg)
    else:
        raise ValueError(f"Unknown loss function: {cfg.LOSS.NAME}")
    
    return loss_fn.to(device)
