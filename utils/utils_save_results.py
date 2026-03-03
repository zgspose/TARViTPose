import json
import os
import datetime
import torch 
import numpy as np
import cv2


def save_results(cfg, results, experiment_dir):
    # Save the results
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

def save_config(cfg, experiment_dir):
    # Save the configuration file
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)


def create_output_folder(output_dir):
    # Get the current date
    current_day = datetime.datetime.now().strftime("%Y-%m-%d")
    experiment_dir = os.path.join(output_dir, current_day)
    
    # Initialize the incremental counter
    counter = 0
    
    # Loop to find an available directory name
    while os.path.exists(experiment_dir):
        counter += 1
        experiment_dir = os.path.join(output_dir, f"{current_day}_{counter}")
    
    # Create the directory
    os.makedirs(experiment_dir)
    
    # Check if the directory was created
    if not os.path.exists(experiment_dir):
        raise Exception("The directory was not created")

    # print the path to the experiment directory
    print(f"Experiment directory: {experiment_dir}")
    
    return experiment_dir

def save_model(model, optimizer, epoch, experiment_dir, scheduler, is_best=False):
    model_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    last_model_path = os.path.join(experiment_dir, 'last_model.pt')
    torch.save(model_state, last_model_path)

    if is_best:
        best_model_path = os.path.join(experiment_dir, 'best_model.pt')
        torch.save(model_state, best_model_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) if checkpoint['scheduler_state_dict'] is not None else None
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"=> Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}', starting from scratch")
        start_epoch = 0  # Start from scratch if no checkpoint exists
    return model, optimizer, start_epoch, scheduler

def draw_keypoints(image, keypoints, save_path):
    """
    Draw keypoints on the image and save it.

    Args:
        image (Tensor): Image tensor of shape (C, H, W).
        keypoints (Tensor): Keypoints tensor of shape (num_keypoints, 2).
        save_path (str): Path to save the image with keypoints.
    """
    # Convert the image tensor to a NumPy array and transpose to (H, W, C)
    img = image.permute(1, 2, 0).cpu().numpy()

    # get image size
    height, width, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = (img * 255)
    
    # Plot the keypoints
    for kp in keypoints:
        if kp[0] > 0 and kp[1] > 0:
            # Convert the keypoints to the image coordinates
            kp[0] *= width 
            kp[1] *= height

            cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

    # Save the image
    cv2.imwrite(save_path, img)
    

def save_batch_examples(input_batch, keypoints_batch, output_dir, batch_idx, num_examples=5):
    """
    Save examples of the batch with keypoints drawn on them.

    Args:
        input_batch (Tensor): Batch of input images of shape (B, C, H, W).
        keypoints_batch (Tensor): Batch of keypoints of shape (B, num_keypoints, 2).
        output_dir (str): Directory to save the images.
        batch_idx (int): Current batch index.
        num_examples (int): Number of examples to save.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_examples = min(num_examples, input_batch.size(0))


    for i in range(num_examples):
        image = input_batch[i]
        keypoints = keypoints_batch[i]

        # create train_example folder
        example_dir = os.path.join(output_dir, 'train_example')

        # create path to save the image
        save_path = os.path.join(example_dir, f'batch_{batch_idx}_example_{i}.jpg')

        # Draw keypoints on the image and save it
        draw_keypoints(image, keypoints, save_path)


def draw_AP(json_path, save_path):
    
    # open json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # get the AP values
    epochs = list(data.keys())
    ap_means = [data[epoch]['performance_values']['Mean'] for epoch in epochs]

    # plot the AP values
    plt.plot(epochs, ap_means, label='mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.tile('mAP vs Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)