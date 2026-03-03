import torch
import torch.nn as nn
import numpy as np

def dist_acc(dists, thr=0.5, percentage=True):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        less_thr_count = np.less(dists[dist_cal], thr).sum() * 1.0
        if percentage:
            return less_thr_count / num_dist_cal
        else:
            return less_thr_count, num_dist_cal  # less_thr_count = match  / num_dist_cal （val）
    else:
        if percentage:
            return -1
        else:
            return -1, -1

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):  # batch num
        for c in range(preds.shape[1]):  # keypoint type
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)  # Euclidean distance
            else:
                dists[c, n] = -1
    return dists

def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK (),
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)  # use a fixed length as a measure rather than the length of body parts

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, effective_num_joints: int = None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        if effective_num_joints is None:
            effective_num_joints = num_joints
        
        # Reshape heatmaps to (batch_size, num_joints, height * width)
        heatmaps_pred = output.view(batch_size, num_joints, -1).split(1, 1)
        heatmaps_gt = target.view(batch_size, num_joints, -1).split(1, 1)
        
        loss = 0.0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)

            if self.use_target_weight:
                # Apply target weight to the loss computation
                loss += self.criterion(
                    heatmap_pred * target_weight[:, idx].unsqueeze(1), 
                    heatmap_gt * target_weight[:, idx].unsqueeze(1)
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        # Average the loss by the number of effective joints
        average_loss = loss / effective_num_joints

        # Debugging information
        print(f"Batch Loss: {average_loss.item()}")
        print(f"Batch Size: {batch_size}")
        print(f"Number of Joints: {num_joints}")
        
        return average_loss

# Example of using the accuracy calculation function
def example_usage():
    batch_size = 4
    num_joints = 17
    height, width = 64, 48

    output = torch.rand(batch_size, num_joints, height, width)
    target = torch.rand(batch_size, num_joints, height, width)
    target_weight = torch.ones(batch_size, num_joints)

    output = target

    criterion = JointsMSELoss(use_target_weight=True)
    loss = criterion(output, target, target_weight)

    _, avg_acc, cnt, _ = accuracy(output.numpy(), target.numpy())

    print(f"Loss: {loss.item()}")
    print(f"Accuracy: {avg_acc}")

# Run the example usage
example_usage()