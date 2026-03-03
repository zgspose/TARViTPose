#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
from datasets.zoo.posetrack import PoseTrack_Official_Keypoint_Ordering, PoseTrack_COCO_Keypoint_Ordering, PoseTrack_Keypoint_Pairs, PoseTrack_Keypoint_Name_Colors


def coco2jhmdb(pose, global_score=1):
    """
    Converts COCO keypoints to JHMDB keypoints.
    
    :param pose: Array of shape (3, 17) representing COCO keypoints [x, y, confidence].
    :param global_score: Overall confidence score multiplier.
    :return: Array of shape (17, 3) representing JHMDB keypoints [x, y, confidence].
    """
    data = []
    global_score = float(global_score)

    # JHMDB keypoints indices in relation to COCO
    jhmdb_kps_map = {
        3: 3,   # left_ear
        4: 4,   # right_ear
        5: 6,   # right_shoulder
        6: 5,   # left_shoulder
        7: 12,  # right_hip
        8: 11,  # left_hip
        9: 8,   # right_elbow
        10: 7,  # left_elbow
        11: 14, # right_knee
        12: 13, # left_knee
        13: 10, # right_wrist
        14: 9,  # left_wrist
        15: 16, # right_ankle
        16: 15  # left_ankle
    }

    for k in range(17):
        if k in jhmdb_kps_map:
            ind = jhmdb_kps_map[k]
            local_score = (pose[2, ind] + pose[2, ind]) / 2.0
            conf = local_score * global_score

            data.append({'id': [k],
                         'x': [float(pose[0, ind])],
                         'y': [float(pose[1, ind])],
                         'score': [conf]})
        elif k == 0:  # 'neck'
            rsho = jhmdb_kps_map[5]  # right_shoulder (mapped to COCO index)
            lsho = jhmdb_kps_map[6]  # left_shoulder (mapped to COCO index)
            x_neck = (pose[0, rsho] + pose[0, lsho]) / 2.0
            y_neck = (pose[1, rsho] + pose[1, lsho]) / 2.0
            local_score = (pose[2, rsho] + pose[2, lsho]) / 2.0
            conf_neck = local_score * global_score

            data.append({'id': [k],
                         'x': [float(x_neck)],
                         'y': [float(y_neck)],
                         'score': [conf_neck]})
        elif k == 1:  # 'belly'
            lhip = jhmdb_kps_map[8]  # left_hip (mapped to COCO index)
            rhip = jhmdb_kps_map[7]  # right_hip (mapped to COCO index)
            x_belly = (pose[0, lhip] + pose[0, rhip]) / 2.0
            y_belly = (pose[1, lhip] + pose[1, rhip]) / 2.0
            local_score = (pose[2, lhip] + pose[2, rhip]) / 2.0
            conf_belly = local_score * global_score

            data.append({'id': [k],
                         'x': [float(x_belly)],
                         'y': [float(y_belly)],
                         'score': [conf_belly]})
        elif k == 2:  # 'head'
            nose = 0  # COCO nose index
            neck_x = (pose[0, jhmdb_kps_map[5]] + pose[0, jhmdb_kps_map[6]]) / 2.0
            neck_y = (pose[1, jhmdb_kps_map[5]] + pose[1, jhmdb_kps_map[6]]) / 2.0
            x_head = (pose[0, nose] + neck_x) / 2.0
            y_head = (pose[1, nose] + neck_y) / 2.0
            local_score = (pose[2, nose] + (pose[2, jhmdb_kps_map[5]] + pose[2, jhmdb_kps_map[6]]) / 2.0) / 2.0
            conf_head = local_score * global_score

            data.append({'id': [k],
                         'x': [float(x_head)],
                         'y': [float(y_head)],
                         'score': [conf_head]})

    return data

def coco2posetrack_ord(preds, global_score=1):
    # print(xy)
    data = []
    src_kps = PoseTrack_COCO_Keypoint_Ordering
    dst_kps = PoseTrack_Official_Keypoint_Ordering

    global_score = float(global_score)
    dstK = len(dst_kps)
    for k in range(dstK):
        # print(k,dst_kps[k])

        if dst_kps[k] in src_kps:
            ind = src_kps.index(dst_kps[k])
            local_score = (preds[2, ind] + preds[2, ind]) / 2.0
            # conf = global_score
            conf = local_score * global_score
            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({'id': [k],
                             'x': [float(preds[0, ind])],
                             'y': [float(preds[1, ind])],
                             'score': [conf]})
        elif dst_kps[k] == 'neck':
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')
            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0
            # conf_msho = global_score
            conf_msho = local_score * global_score

            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({'id': [k],
                             'x': [float(x_msho)],
                             'y': [float(y_msho)],
                             'score': [conf_msho]})
        elif dst_kps[k] == 'head_top':
            # print(xy)
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')

            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0

            nose = src_kps.index('nose')
            x_nose = preds[0, nose]
            y_nose = preds[1, nose]
            x_tophead = x_nose - (x_msho - x_nose)
            y_tophead = y_nose - (y_msho - y_nose)
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0
            #
            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if True:
                data.append({
                    'id': [k],
                    'x': [float(x_tophead)],
                    'y': [float(y_tophead)],
                    'score': [local_score]})
    return data


def coco2posetrack_ord_infer(pose, global_score=1, output_posetrack_format=False):
    # pose [x,y,c]
    src_kps = PoseTrack_COCO_Keypoint_Ordering
    dst_kps = PoseTrack_Official_Keypoint_Ordering
    if not output_posetrack_format:
        data = np.zeros((len(dst_kps), 3))
    else:
        data = []
    for dst_index, posetrack_keypoint_name in enumerate(dst_kps):
        if posetrack_keypoint_name in src_kps:
            index = src_kps.index(posetrack_keypoint_name)
            local_score = (pose[index, 2] + pose[index, 2]) / 2
            conf = local_score * global_score
            if not output_posetrack_format:
                data[dst_index, :] = pose[index]
                data[dst_index, 2] = conf
            else:
                data.append({'id': [dst_index],
                             'x': [float(pose[index, 0])],
                             'y': [float(pose[index, 1])],
                             'score': [conf]})


        elif posetrack_keypoint_name == 'neck':
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')
            x_msho = (pose[rsho, 0] + pose[lsho, 0]) / 2.0
            y_msho = (pose[rsho, 1] + pose[lsho, 1]) / 2.0
            local_score = (pose[rsho, 2] + pose[lsho, 2]) / 2.0
            conf_msho = local_score * global_score

            # if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
            if not output_posetrack_format:
                data[dst_index, 0] = float(x_msho)
                data[dst_index, 1] = float(y_msho)
                data[dst_index, 2] = conf_msho
            else:
                data.append({'id': [dst_index],
                             'x': [float(x_msho)],
                             'y': [float(y_msho)],
                             'score': [conf_msho]})

        elif posetrack_keypoint_name == 'head_top':
            # print(xy)
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')

            x_msho = (pose[rsho, 0] + pose[lsho, 0]) / 2.0
            y_msho = (pose[rsho, 1] + pose[lsho, 1]) / 2.0

            nose = src_kps.index('nose')
            x_nose = pose[nose, 0]
            y_nose = pose[nose, 1]
            x_tophead = x_nose - (x_msho - x_nose)
            y_tophead = y_nose - (y_msho - y_nose)
            local_score = (pose[rsho, 2] + pose[lsho, 2]) / 2.0
            if not output_posetrack_format:
                data[dst_index, 0] = float(x_tophead)
                data[dst_index, 1] = float(y_tophead)
                data[dst_index, 2] = local_score
            else:
                data.append({'id': [dst_index],
                             'x': [float(x_tophead)],
                             'y': [float(y_tophead)],
                             'score': [local_score]})

    return data
