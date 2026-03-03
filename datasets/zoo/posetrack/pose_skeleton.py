#!/usr/bin/python
# -*- coding:utf8 -*-

__all__ = ["PoseTrack_Official_Keypoint_Ordering",
           "PoseTrack_Keypoint_Pairs",
           "PoseTrack_Keypoint_Name_Colors",
           "PoseTrack_COCO_Keypoint_Ordering"]

#  PoseTrack Official Keypoint Ordering  - A total of 15
PoseTrack_Official_Keypoint_Ordering = [
    'right_ankle',  # 0
    'right_knee',   # 1
    'right_hip',    # 2
    'left_hip',    # 3  
    'left_knee',    # 4
    'left_ankle',   # 5
    'right_wrist',  # 6
    'right_elbow',  # 7
    'right_shoulder',   # 8
    'left_shoulder',    # 9
    'left_elbow',    # 10
    'left_wrist',   # 11  
    'head_bottom',  # 12
    'nose', # 13
    'head_top', # 14
]

# Endpoint1 , Endpoint2 , line_color
PoseTrack_Keypoint_Pairs = [
    ['head_top', 'head_bottom', 'Rosy'],
    ['head_bottom', 'right_shoulder', 'Yellow'],
    ['head_bottom', 'left_shoulder', 'Yellow'],
    ['right_shoulder', 'right_elbow', 'Blue'],
    ['right_elbow', 'right_wrist', 'Blue'],
    ['left_shoulder', 'left_elbow', 'Green'],
    ['left_elbow', 'left_wrist', 'Green'],
    ['right_shoulder', 'right_hip', 'Purple'],
    ['left_shoulder', 'left_hip', 'SkyBlue'],
    ['right_hip', 'right_knee', 'Purple'],
    ['right_knee', 'right_ankle', 'Purple'],
    ['left_hip', 'left_knee', 'SkyBlue'],
    ['left_knee', 'left_ankle', 'SkyBlue'],
]

# Facebook PoseTrack Keypoint Ordering (convert to COCO format)  -   A total of 17
PoseTrack_COCO_Keypoint_Ordering = [
    'nose',#1
    'head_bottom',#2
    'head_top',#0
    'left_ear',#0
    'right_ear',#0
    'left_shoulder',#3
    'right_shoulder',#6
    'left_elbow',#4
    'right_elbow',#7
    'left_wrist',#5
    'right_wrist',#8
    'left_hip',#9
    'right_hip',#12
    'left_knee',#10
    'right_knee',#13
    'left_ankle',#11
    'right_ankle',#14
    #[1,2,0,0,0,3,6,4,7,5,8,9,12,10,13,11,14]
    #0 'head_top',#0
    #1 'nose',#1
    #2 'head_bottom',#2
    #3 'left_shoulder',#3
    #4 'left_elbow',
    #5 'left_wrist',#
    #6 'right_shoulder',#
    #7 'right_elbow',#
    #8 'right_wrist',#
    #9 'left_hip',#
    #10 'left_knee',#
    #11 'left_ankle',#
    #12 'right_hip'',#
    #13 'right_knee,#
    #14 'right_ankle',
]

COCO_Keypoint_Ordering = [
    'nose',  # 0
    'left_eye',  # 1
    'right_eye',  # 2
    'left_ear',  # 3
    'right_ear',  # 4
    'left_shoulder',  # 5
    'right_shoulder',  # 6
    'left_elbow',  # 7
    'right_elbow',  # 8
    'left_wrist',  # 9
    'right_wrist',  # 10
    'left_hip',  # 11
    'right_hip',  # 12
    'left_knee',  # 13
    'right_knee',  # 14
    'left_ankle',  # 15
    'right_ankle'  # 16
]

JHMDB_Keypoint_Ordering = [
    'neck',  # 0
    'belly',  # 1 (midpoint between left_hip and right_hip)
    'head',  # 2 (can be computed as midpoint between nose and neck)
    'left_ear',  # 3
    'right_ear',  # 4
    'right_shoulder',  # 5
    'left_shoulder',  # 6
    'right_hip',  # 7
    'left_hip',  # 8
    'right_elbow',  # 9
    'left_elbow',  # 10
    'right_knee',  # 11
    'left_knee',  # 12
    'right_wrist',  # 13
    'left_wrist',  # 14
    'right_ankle',  # 15
    'left_ankle'  # 16
]

PoseTrack_Keypoint_Name_Colors = [['right_ankle', 'Gold'],
                                  ['right_knee', 'Orange'],
                                  ['right_hip', 'DarkOrange'],
                                  ['left_hip', 'Peru'],
                                  ['left_knee', 'LightSalmon'],
                                  ['left_ankle', 'OrangeRed'],
                                  ['right_wrist', 'LightGreen'],
                                  ['right_elbow', 'LimeGreen'],
                                  ['right_shoulder', 'ForestGreen'],
                                  ['left_shoulder', 'DarkTurquoise'],
                                  ['left_elbow', 'Cyan'],
                                  ['left_wrist', 'PaleTurquoise'],
                                  ['head_bottom', 'DoderBlue'],
                                  ['nose', 'HotPink'],
                                  ['head_top', 'SlateBlue']]
