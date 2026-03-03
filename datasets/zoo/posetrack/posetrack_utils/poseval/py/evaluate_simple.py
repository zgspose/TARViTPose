from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
from .evaluateAP import evaluateAP
from .evaluatePCKh import evaluatePCKh
from .evaluateTracking import evaluateTracking
from .eval_helpers import Joint, printTable, load_data_dir, getCum
import os
import json
import numpy as np


def load_data_dir_jhmdb(dirs):
    """
    Load ground truth and prediction data from specified directories.
    
    Parameters:
        dirs (list): List containing empty string, path to ground truth directory, and path to predictions directory.
    
    Returns:
        tuple: Two lists containing ground truth frames and predicted frames.
    """
    _, gt_dir, pred_dir = dirs
    gt_frames_all = []
    pred_frames_all = []
    
    # Load all JSON files in ground truth directory
    for gt_file in os.listdir(gt_dir):
        if gt_file.endswith('.json'):
            with open(os.path.join(gt_dir, gt_file), 'r') as file:
                gt_data = json.load(file)
                for annotation in gt_data['annotations']:
                    # Mapping keypoints using COCO indices
                    frame = {
                        'id': annotation['image_id'],
                        'keypoints': annotation['keypoints'],
                        'bbox': annotation['bbox'],
                        'category_id': annotation['category_id']
                    }

                    gt_frames_all.append(frame)
    
    # Load all JSON files in prediction directory
    for pred_file in os.listdir(pred_dir):
        if pred_file.endswith('.json'):
            video_id = pred_file.split('.')[0]
            with open(os.path.join(pred_dir, pred_file), 'r') as file:
                pred_data = json.load(file)
                for annolist in pred_data['annolist']:
                    for annorect in annolist['annorect']:
                        frame_number = int(annolist['image']['name'].split('/')[-1].split('.')[0])  # Get frame number
                        frame_id = int(video_id) * 10000 + frame_number  # Combine video and frame to match GT format
                        keypoints = [
                            (kp['x'][0], kp['y'][0], kp['score'][0] if 'score' in kp else 0.0)
                            for kp in annorect['annopoints'][0]['point']
                        ]
                        frame = {
                            'id': frame_id,
                            'keypoints': keypoints,
                            'bbox': annorect.get('score', [0, 0, 0, 0]),  # Adjust if bbox is available in your prediction
                            'category_id': 1  # Assuming single-category (person) detection
                        }
                        pred_frames_all.append(frame)

    return gt_frames_all, pred_frames_all

def calculate_pck(gt_frames, pred_frames, keypoint_names, threshold=0.05):
    """
    Calculate the PCK (Percentage of Correct Keypoints) for ground truth and predicted frames.

    Parameters:
        gt_frames (list): List of dictionaries containing ground truth frames.
        pred_frames (list): List of dictionaries containing predicted frames.
        keypoint_names (list): List of keypoint names for mapping.
        threshold (float): Normalized distance threshold for PCK (default is 0.5).

    Returns:
        dict: Dictionary containing PCK values for each keypoint.
    """
    pck_results = {name: {'correct': 0, 'total': 0} for name in keypoint_names}

    # Create a mapping of frame IDs for fast lookup
    pred_frame_dict = {frame['id']: frame for frame in pred_frames}

    for gt_frame in gt_frames:
        frame_id = gt_frame['id']
        if frame_id in pred_frame_dict:
            pred_frame = pred_frame_dict[frame_id]
            gt_keypoints = np.array(gt_frame['keypoints']).reshape(-1, 3)
            pred_keypoints = np.array(pred_frame['keypoints']).reshape(-1, 3)
            
            # Calculate the distance between predicted and ground truth keypoints
            for i, keypoint_name in enumerate(keypoint_names):
                if gt_keypoints[i, 2] > 0 and pred_keypoints[i, 2] > 0:  # Check visibility (v > 0)
                    distance = np.linalg.norm(gt_keypoints[i, :2] - pred_keypoints[i, :2])
                    bbox = gt_frame['bbox']
                    normalization_factor = max(bbox[2], bbox[3])  # Use the maximum dimension of the bounding box
                    
                    if distance / normalization_factor <= threshold:
                        pck_results[keypoint_name]['correct'] += 1
                    pck_results[keypoint_name]['total'] += 1

    # Calculate PCK percentages
    pck_percentages = {name: (results['correct'] / results['total'] * 100 if results['total'] > 0 else 0)
                       for name, results in pck_results.items()}

    # remove left and right ear
    pck_percentages.pop("left_ear", None)
    pck_percentages.pop("right_ear", None)

    # remove neck and belly
    pck_percentages.pop("neck", None)
    pck_percentages.pop("belly", None)

    # calculate mean of right and left shoulder
    pck_percentages["shoulder"] = (pck_percentages["right_shoulder"] + pck_percentages["left_shoulder"]) / 2
    # remove right and left shoulder
    pck_percentages.pop("right_shoulder", None)
    pck_percentages.pop("left_shoulder", None)

    # calculate mean of right and left hip
    pck_percentages["hip"] = (pck_percentages["right_hip"] + pck_percentages["left_hip"]) / 2
    # remove right and left hip
    pck_percentages.pop("right_hip", None)
    pck_percentages.pop("left_hip", None)

    # calculate mean of right and left elbow
    pck_percentages["elbow"] = (pck_percentages["right_elbow"] + pck_percentages["left_elbow"]) / 2
    # remove right and left elbow
    pck_percentages.pop("right_elbow", None)
    pck_percentages.pop("left_elbow", None)

    # calculate mean of right and left knee
    pck_percentages["knee"] = (pck_percentages["right_knee"] + pck_percentages["left_knee"]) / 2
    # remove right and left knee
    pck_percentages.pop("right_knee", None)
    pck_percentages.pop("left_knee", None)

    # calculate mean of right and left wrist
    pck_percentages["wrist"] = (pck_percentages["right_wrist"] + pck_percentages["left_wrist"]) / 2
    # remove right and left wrist
    pck_percentages.pop("right_wrist", None)
    pck_percentages.pop("left_wrist", None)

    # calculate mean of right and left ankle
    pck_percentages["ankle"] = (pck_percentages["right_ankle"] + pck_percentages["left_ankle"]) / 2
    # remove right and left ankle
    pck_percentages.pop("right_ankle", None)
    pck_percentages.pop("left_ankle", None)

    # add mean  
    pck_percentages["Mean"] = np.mean(list(pck_percentages.values())) 

    # round to 2 decimal places
    pck_percentages = {key: round(value, 1) for key, value in pck_percentages.items()}

    return pck_percentages


def evaluate(gtdir, preddir, eval_pose=True, eval_track=True,
             eval_upper_bound=False, dataset_name='posetrack'):
    logger = logging.getLogger(__name__)

    if dataset_name == "posetrack":
        gtFramesAll, prFramesAll = load_data_dir(['', gtdir, preddir])
    else:
        gtFramesAll, prFramesAll = load_data_dir_jhmdb(['', gtdir, preddir])

    logger.info('# gt frames  : {}'.format(str(len(gtFramesAll))))
    logger.info('# pred frames: {}'.format(str(len(prFramesAll))))

    print('# gt frames  : {}'.format(str(len(gtFramesAll))))
    print('# pred frames: {}'.format(str(len(prFramesAll))))

    apAll = np.full((Joint().count + 1, 1), np.nan)
    preAll = np.full((Joint().count + 1, 1), np.nan)
    recAll = np.full((Joint().count + 1, 1), np.nan)
    cum = None
    track_cum = None
    if eval_pose:

        if dataset_name == "posetrack":
            apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll)
            logger.info('Average Precision (AP) metric:')
            

            print('Average Precision (AP) metric:')
            # printTable(apAll)
            cum = printTable(apAll)
        #pckAll = evaluatePCKh(gtFramesAll, prFramesAll)
        else :
            keypoint_names = [
                    "neck",
                    "belly",
                    "head",
                    "left_ear",
                    "right_ear",
                    "right_shoulder",
                    "left_shoulder",
                    "right_hip",
                    "left_hip",
                    "right_elbow",
                    "left_elbow",
                    "right_knee",
                    "left_knee",
                    "right_wrist",
                    "left_wrist",
                    "right_ankle",
                    "left_ankle"
                ]
            print("PCK(0.2) results:", calculate_pck(gtFramesAll, prFramesAll, keypoint_names, threshold=0.2))
            print("PCK(0.1) results:", calculate_pck(gtFramesAll, prFramesAll, keypoint_names, threshold=0.1))

            cum = calculate_pck(gtFramesAll, prFramesAll, keypoint_names, threshold=0.05)
            print("PCK(0.05) results:", cum)


    metrics = np.full((Joint().count + 4, 1), np.nan)
    # print(eval_track)
    if eval_track:
        # print(xy)
        metricsAll = evaluateTracking(
            gtFramesAll, prFramesAll, eval_upper_bound)

        for i in range(Joint().count + 1):
            metrics[i, 0] = metricsAll['mota'][0, i]
        metrics[Joint().count + 1, 0] = metricsAll['motp'][0, Joint().count]
        metrics[Joint().count + 2, 0] = metricsAll['pre'][0, Joint().count]
        metrics[Joint().count + 3, 0] = metricsAll['rec'][0, Joint().count]
        logger.info('Multiple Object Tracking (MOT) mmetrics:')
        # print('Multiple Object Tracking (MOT) mmetrics:')
        track_cum = printTable(metrics, motHeader=True)
    # return (apAll, preAll, recAll), metrics
    # print(xy)
    return cum, track_cum
