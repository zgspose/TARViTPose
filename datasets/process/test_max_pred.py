import numpy as np
import torch
import time

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

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

def get_max_preds_gpu(batch_heatmaps):
    '''
    get predictions from score maps
    batch_heatmaps: torch.Tensor([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, torch.Tensor), \
        'batch_heatmaps should be torch.Tensor'
    assert batch_heatmaps.dim() == 4, 'batch_images should be 4-dim'

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    width = batch_heatmaps.size(3)

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    maxvals, idx = torch.max(heatmaps_reshaped, 2)
    
    maxvals = maxvals.unsqueeze(2)
    idx = idx.unsqueeze(2)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float()
    preds *= pred_mask

    return preds, maxvals

def compare_results(cpu_preds, cpu_maxvals, gpu_preds, gpu_maxvals, tolerance=1e-5):
    """
    Compare the results from CPU and GPU implementations.
    
    :param cpu_preds: Predictions from CPU implementation (numpy array)
    :param cpu_maxvals: Max values from CPU implementation (numpy array)
    :param gpu_preds: Predictions from GPU implementation (torch tensor)
    :param gpu_maxvals: Max values from GPU implementation (torch tensor)
    :param tolerance: Tolerance for floating point differences
    :return: True if results match within tolerance, False otherwise
    """
    # Convert GPU results to CPU numpy arrays
    gpu_preds_np = gpu_preds.cpu().numpy()
    gpu_maxvals_np = gpu_maxvals.cpu().numpy()

    # check time comparison
    
    # Check shapes
    if cpu_preds.shape != gpu_preds_np.shape or cpu_maxvals.shape != gpu_maxvals_np.shape:
        print("Shape mismatch!")
        return False

    # Check predictions
    preds_diff = np.abs(cpu_preds - gpu_preds_np)
    if np.max(preds_diff) > tolerance:
        print(f"Max difference in predictions: {np.max(preds_diff)}")
        return False
    
    
    # Check max values
    maxvals_diff = np.abs(cpu_maxvals - gpu_maxvals_np)
    if np.max(maxvals_diff) > tolerance:
        print(f"Max difference in max values: {np.max(maxvals_diff)}")
        return False
    
    print("Results match within tolerance!")
    return True

# Example usage
def main():
    # Create sample input
    batch_size, num_joints, height, width = 32, 17, 256, 192
    batch_heatmaps = np.random.rand(batch_size, num_joints, height, width).astype(np.float32)
    
    start = time.time()
    # CPU version
    for i in range(100):
        cpu_preds, cpu_maxvals = get_max_preds(batch_heatmaps)
    
    # Check time
    print(f"CPU time: {time.time() - start}")

    
    # GPU version
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    batch_heatmaps_gpu = torch.from_numpy(batch_heatmaps).to(device)

    start = time.time()
    for i in range (100):
        gpu_preds, gpu_maxvals = get_max_preds_gpu(batch_heatmaps_gpu)

    # Check time
    print(f"GPU time: {time.time() - start}")
    
    # Compare results
    results_match = compare_results(cpu_preds, cpu_maxvals, gpu_preds, gpu_maxvals)
    print(f"Results match: {results_match}")

if __name__ == "__main__":
    main()