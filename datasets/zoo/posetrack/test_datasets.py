import yaml
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('/home/pace/Poseidon/'))
from datasets.zoo.posetrack.PoseTrack import PoseTrack 
from posetimation import get_cfg, update_config 
from engine.defaults import default_parse_args
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
import torch

def load_config(config_path):
    """ Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)

    return cfg

def test_load_images(dataset, num_samples=10):
    """ Test loading and displaying images from the PoseTrack dataset.
    Args:
        dataset (PoseTrack): Dataset instance.
        num_samples (int): Number of samples to display.
    Returns:
        None
    """
    for i in range(num_samples):
        try:
            x, meta, target_heatmaps, target_heatmaps_weight = dataset[i]
            print(f"Sample {i}: Image loaded successfully.")
            print(f"Image shape: {x.shape}")
            
            # Let's save the current frame (index 1)
            num_frames = x.shape[1]
            central_frame_idx = num_frames // 2
            current_frame = x[central_frame_idx]  # Shape: [C, H, W]
            
            print("X shape: ", x.shape)
            print("current_frame shape: ", current_frame.shape)

            name = f"sample_{i}.jpg"
            
            # Convert to numpy array
            numpy_image = current_frame.cpu().numpy()
            
            # Get image dimensions
            height, width = numpy_image.shape[1:]
            
            # Convert the numpy array to a cv2 image
            cv2_image = np.transpose(numpy_image, (1, 2, 0))
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
            cv2_image = (cv2_image * 255)
            
            # Draw the keypoints on the image
            joints = meta['joints']
            joints_vis = meta['joints_vis']
            
            for joint, joint_vis in zip(joints, joints_vis):
                x, y, _ = joint
                visible = joint_vis[0]
                if visible > 0:  # Check if the keypoint is visible
                    x_pixel = int(x)
                    y_pixel = int(y)
                    cv2.circle(cv2_image, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
            
            # Save the image
            cv2.imwrite(name, cv2_image)
            
            # Also save the heatmap
            heatmap = target_heatmaps.sum(dim=0).cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            cv2.imwrite(f"heatmap_{i}.jpg", heatmap)
            
            print(f"Center: {meta['center']}")
            print(f"Scale: {meta['scale']}")
            print(f"Rotation: {meta['rotation']}")
            print("-" * 50)
        except Exception as e:
            print(f"Failed to process sample {i}: {str(e)}")

def test_motion_augmentation(dataset, num_samples=1):
    """Test and visualize motion augmentations by saving sequences of frames.

    Args:
        dataset (PoseTrack): Dataset instance.
        num_samples (int): Number of samples to process.
    """
    for i in range(num_samples):
        try:
            # Retrieve a sample from the dataset
            x, meta, target_heatmaps, target_heatmaps_weight = dataset[i]
            print(f"Sample {i}: Data loaded successfully.")
            print(f"x shape: {x.shape}")  # Expected shape: [num_frames, C, H, W]
            
            # Ensure x has the correct dimensions
            assert len(x.shape) == 4, f"Expected x to have 4 dimensions, got {len(x.shape)}"
            
            num_frames, C, H, W = x.shape
            print(f"Number of frames: {num_frames}, Image size: {H}x{W}")
            
            # Get joints and visibility for the central frame
            joints = meta['joints']
            joints_vis = meta['joints_vis']
            
            # Create a directory to save the frames for this sample
            sample_dir = f"sample_{i}"
            os.makedirs(sample_dir, exist_ok=True)
            
            # Process each frame in the sequence
            for frame_idx in range(num_frames):
                current_frame = x[frame_idx]  # Shape: [C, H, W]
                
                # Convert to numpy array and transpose to [H, W, C]
                numpy_image = current_frame.cpu().numpy()
                cv2_image = np.transpose(numpy_image, (1, 2, 0))  # Shape: [H, W, C]
                
                # Convert from tensor (normalized between [0,1]) to uint8 image
                cv2_image = np.transpose(numpy_image, (1, 2, 0))
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
                cv2_image = (cv2_image * 255)
                
                # Draw the keypoints on the central frame only
                if frame_idx == num_frames // 2:
                    for joint, joint_vis in zip(joints, joints_vis):
                        x_coord, y_coord, _ = joint
                        visible = joint_vis[0]
                        if visible > 0:  # Check if the keypoint is visible
                            x_pixel = int(x_coord)
                            y_pixel = int(y_coord)
                            cv2.circle(cv2_image, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
                
                # Save the image
                frame_name = os.path.join(sample_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_name, cv2_image)
                print(f"Saved frame: {frame_name}")
            
            # Also save the heatmap for the central frame
            heatmap = target_heatmaps.sum(dim=0).cpu().numpy()  # Sum over joints
            heatmap = (heatmap / heatmap.max()) * 255.0  # Normalize to [0,255]
            heatmap = heatmap.astype(np.uint8)
            
            # Resize heatmap to match image size if necessary
            heatmap = cv2.resize(heatmap, (W, H))
            
            heatmap_name = os.path.join(sample_dir, "heatmap.jpg")
            cv2.imwrite(heatmap_name, heatmap)
            print(f"Saved heatmap: {heatmap_name}")
            
            print(f"Center: {meta['center']}")
            print(f"Scale: {meta['scale']}")
            print(f"Rotation: {meta['rotation']}")
            print("-" * 50)
        except Exception as e:
            print(f"Failed to process sample {i}: {str(e)}")

def main():
    # Load configuration from a YAML file.

    args = default_parse_args()
    print("args")
    print(args)
    cfg = setup(args)

    #config_path = '/home/pace/Poseidon/configs/configDCPose.yaml'
    #cfg = load_config(config_path)

    # Initialize the PoseTrack dataset.
    pose_track_dataset = PoseTrack(cfg, TRAIN_PHASE)

    # check the length of the dataset
    print(f"Number of samples in the dataset: {len(pose_track_dataset)}")

    # Test loading images.
    #test_load_images(pose_track_dataset)

    # Test motion augmentation.
    test_motion_augmentation(pose_track_dataset)

if __name__ == '__main__':
    main()

    # python datasets/zoo/posetrack/test_datasets.py --config /home/pace/Poseidon/configs/configPoseidon.yaml