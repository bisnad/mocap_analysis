"""
Things that are unclear at the moment:
    
How the travel distance is calculated (especially the mean, std, max, min), at the moment, I calculate only the mean
Same issue applies to the area covered

"""


"""
imports
"""

from common import utils
from common import bvh_tools as bvh
from common import mocap_tools as mocap
import motion_analysis as ma

import torch
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
import json

from matplotlib import pyplot as plt
import numpy as np

"""
mocap settings
"""

mocap_data_path = "../../../Data/mocap/stocos/solos/Muriel_Take4.bvh"
mocap_joint_weights_path = "configs/joint_weights_xsens_bvh.json"
mocap_fps = 50

"""
load mocap data
"""

bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

bvh_data = bvh_tools.load(mocap_data_path)
mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
mocap_data["motion"]["pos_world"], mocap_data["motion"]["rot_world"] = mocap_tools.local_to_world(mocap_data["motion"]["rot_local"], mocap_data["motion"]["pos_local"], mocap_data["skeleton"])

frame_count = mocap_data["motion"]["rot_local"].shape[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = mocap_data["motion"]["rot_local"].shape[2]
joint_names = mocap_data["skeleton"]["joints"]
joint_name_index_map = { jName : jI for jI, jName in enumerate(joint_names) }
offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

joint_weights = json.load(open(mocap_joint_weights_path))["jointWeights"]

"""
compute mocap features
"""

analysis_window_size = 16
smooth_window_size = 10

"""
# compute scalars and derivatives from joint positions
"""

mocap_data["motion"]["pos_world_m"] = mocap_data["motion"]["pos_world"] / 100.0 # from cm to meters
mocap_data["motion"]["pos_world_smooth"] = ma.smooth(mocap_data["motion"]["pos_world_m"], smooth_window_size)
mocap_data["motion"]["pos_scalar"] = ma.scalar(mocap_data["motion"]["pos_world_smooth"], "norm")
mocap_data["motion"]["vel_world"] = ma.derivative(mocap_data["motion"]["pos_world_smooth"], 1.0 / mocap_fps)
mocap_data["motion"]["vel_world_smooth"] = ma.smooth(mocap_data["motion"]["vel_world"], smooth_window_size)
mocap_data["motion"]["vel_world_scalar"] = ma.scalar(mocap_data["motion"]["vel_world_smooth"], "norm")
mocap_data["motion"]["accel_world"] = ma.derivative(mocap_data["motion"]["vel_world_smooth"], 1.0 / mocap_fps)
mocap_data["motion"]["accel_world_smooth"] = ma.smooth(mocap_data["motion"]["accel_world"], smooth_window_size)
mocap_data["motion"]["accel_world_scalar"] = ma.scalar(mocap_data["motion"]["accel_world_smooth"], "norm")
mocap_data["motion"]["jerk_world"] = ma.derivative(mocap_data["motion"]["accel_world_smooth"], 1.0 / mocap_fps)
mocap_data["motion"]["jerk_world_smooth"] = ma.smooth(mocap_data["motion"]["jerk_world"], smooth_window_size)
mocap_data["motion"]["jerk_world_scalar"] = ma.scalar(mocap_data["motion"]["jerk_world_smooth"], "norm")

"""
get derivatives of specific joints
TODO: check with Andreas if using the scalar values is the correct approach
"""

mocap_data["motion"]["hip_vel_scalar"] =  mocap_data["motion"]["vel_world_scalar"][:, joint_name_index_map["Hips"], :]
mocap_data["motion"]["hand_vel_scalar"] =  mocap_data["motion"]["vel_world_scalar"][:, joint_name_index_map["RightHand"], :] + mocap_data["motion"]["vel_world_scalar"][:, joint_name_index_map["LeftHand"], :]
mocap_data["motion"]["foot_vel_scalar"] =  mocap_data["motion"]["vel_world_scalar"][:, joint_name_index_map["RightFoot"], :] + mocap_data["motion"]["vel_world_scalar"][:, joint_name_index_map["LeftFoot"], :]

mocap_data["motion"]["hip_accel_scalar"] =  mocap_data["motion"]["accel_world_scalar"][:, joint_name_index_map["Hips"], :]
mocap_data["motion"]["hand_accel_scalar"] =  mocap_data["motion"]["accel_world_scalar"][:, joint_name_index_map["RightHand"], :] + mocap_data["motion"]["accel_world_scalar"][:, joint_name_index_map["LeftHand"], :]
mocap_data["motion"]["foot_accel_scalar"] =  mocap_data["motion"]["accel_world_scalar"][:, joint_name_index_map["RightFoot"], :] + mocap_data["motion"]["accel_world_scalar"][:, joint_name_index_map["LeftFoot"], :]

mocap_data["motion"]["hip_jerk_scalar"] =  mocap_data["motion"]["jerk_world_scalar"][:, joint_name_index_map["Hips"], :]


# find peaks for hip acceleration
# TODO: check with Andreas if this makes any sense

from scipy.signal import find_peaks

up_peaks, up_peak_properties = find_peaks(mocap_data["motion"]["hip_accel_scalar"][:, 0] , height=np.mean(mocap_data["motion"]["hip_accel_scalar"][:,0]))
down_peaks, down_peak_properties = find_peaks(-mocap_data["motion"]["hip_accel_scalar"][:, 0] , height=-np.mean(mocap_data["motion"]["hip_accel_scalar"][:,0]))

peaks = np.zeros(mocap_data["motion"]["hip_accel_scalar"].shape[0])
peaks[ up_peaks ] = up_peak_properties["peak_heights"]
peaks[ down_peaks ] = down_peak_properties["peak_heights"]
peaks = np.expand_dims(peaks, axis=1)

mocap_data["motion"]["hip_accel_peaks"] = peaks

"""
# compute distances between joints
"""

# feet hip distance
mocap_data["motion"]["rfoot_hip_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightFoot"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Hips"], :] )
mocap_data["motion"]["lfoot_hip_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftFoot"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Hips"], :] )
mocap_data["motion"]["foot_hip_dist"] = (mocap_data["motion"]["rfoot_hip_dist"] + mocap_data["motion"]["lfoot_hip_dist"]) / 2.0

# hand shoulder distance
mocap_data["motion"]["rhand_rshoulder_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightArm"], :] )
mocap_data["motion"]["lhand_lshoulder_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftArm"], :] )
mocap_data["motion"]["hand_shoulder_dist"] = (mocap_data["motion"]["rhand_rshoulder_dist"] + mocap_data["motion"]["lhand_lshoulder_dist"]) / 2.0

# hand distance
mocap_data["motion"]["hand_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftHand"], :] )

# feet distance
mocap_data["motion"]["feet_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightFoot"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftFoot"], :] )

# head hip distance
mocap_data["motion"]["head_hip_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Head"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Hips"], :] )

# hand head distance
mocap_data["motion"]["rhand_head_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Head"], :] )
mocap_data["motion"]["lhand_head_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Head"], :] )
mocap_data["motion"]["hand_head_dist"] = (mocap_data["motion"]["rhand_head_dist"] + mocap_data["motion"]["lhand_head_dist"]) / 2.0

# hand hip distance
mocap_data["motion"]["rhand_hip_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Hips"], :] )
mocap_data["motion"]["lhand_hip_dist"] = ma.joint_distance( mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Hips"], :] )
mocap_data["motion"]["hand_hip_dist"] = (mocap_data["motion"]["rhand_hip_dist"] + mocap_data["motion"]["lhand_hip_dist"]) / 2.0

"""
# calculate height of joints
"""

# hip height
mocap_data["motion"]["hip_height"] = mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Hips"], 1:2]

# hip height and foot hip distance combined
mocap_data["motion"]["hip_height_foot_hip_dist"] = mocap_data["motion"]["hip_height"] + mocap_data["motion"]["foot_hip_dist"] 

"""
# compute hand levels # 1: above head, 2: between head and spine 3: below spine
"""

mocap_data["motion"]["rhand_level"] = ma.joint_level(mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["RightHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Head"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Spine"], :])
mocap_data["motion"]["lhand_level"] = ma.joint_level(mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["LeftHand"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Head"], :], mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Spine"], :])
mocap_data["motion"]["hand_level_upper"] = (mocap_data["motion"]["rhand_level"] == 1).astype(int) + (mocap_data["motion"]["lhand_level"] == 1).astype(int) 
mocap_data["motion"]["hand_level_middle"] = (mocap_data["motion"]["rhand_level"] == 2).astype(int) + (mocap_data["motion"]["lhand_level"] == 2).astype(int) 
mocap_data["motion"]["hand_level_below"] = (mocap_data["motion"]["rhand_level"] == 3).astype(int) + (mocap_data["motion"]["lhand_level"] == 3).astype(int) 

"""
compute volumes
"""

# volume_5
volume_5_joint_names = ["Head", "RightHand", "LeftHand", "RightFoot", "LeftFoot"]
mocap_data["motion"]["volume_5"] = ma.joint_volumes( mocap_data["motion"]["pos_world_m"][:, [ joint_name_index_map[joint_name] for joint_name in volume_5_joint_names ], :] )

# volume_all
volume_all_joint_names = mocap_data["skeleton"]["joints"]
mocap_data["motion"]["volume_all"] = ma.joint_volumes( mocap_data["motion"]["pos_world_m"][:, [ joint_name_index_map[joint_name] for joint_name in volume_all_joint_names ], :] )

# volume_upper
volume_upper_joint_names = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_Nub", "RightArm", "RightForeArm", "RightHand", "RightHand_Nub"]
mocap_data["motion"]["volume_upper"] = ma.joint_volumes( mocap_data["motion"]["pos_world_m"][:, [ joint_name_index_map[joint_name] for joint_name in volume_upper_joint_names ], :] )

# volume_lower
volume_lower_joint_names = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Spine"]
mocap_data["motion"]["volume_lower"] = ma.joint_volumes( mocap_data["motion"]["pos_world_m"][:, [ joint_name_index_map[joint_name] for joint_name in volume_lower_joint_names ], :] )

# volume_right
volume_right_joint_names = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head", "RightArm", "RightForeArm", "RightHand", "RightHand_Nub"]
mocap_data["motion"]["volume_right"] = ma.joint_volumes( mocap_data["motion"]["pos_world_m"][:, [ joint_name_index_map[joint_name] for joint_name in volume_right_joint_names ], :] )

# volume_left
volume_left_joint_names = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_Nub"]
mocap_data["motion"]["volume_left"] = ma.joint_volumes( mocap_data["motion"]["pos_world_m"][:, [ joint_name_index_map[joint_name] for joint_name in volume_left_joint_names ], :] )

"""
compute travel area (area_covered)

TODO: check if this is correct
"""

ground_trajectory = mocap_data["motion"]["pos_world_m"][:, joint_name_index_map["Hips"], [0, 2]]

mocap_data["motion"]["area_covered"] = ma.windowed_joint_volume(ground_trajectory, analysis_window_size)

"""
travel_area = []
for fI in range(frame_count - analysis_window_size):
    travel_area.append(ma.joint_volume(ground_trajectory[fI:fI + analysis_window_size]))
    
travel_area = np.array(travel_area)
travel_area = np.reshape(travel_area, [-1, 1])
mocap_data["motion"]["area_covered"]  = travel_area
"""

"""
compute travel distance 

TODO: check if this is correct
"""

mocap_data["motion"]["travel_distance"] = ma.windowed_joint_travel_distance(ground_trajectory, analysis_window_size)

"""
travel_distance = []
for fI in range(frame_count - analysis_window_size):
    travel_distance.append(ma.joint_travel_distance(ground_trajectory[fI:fI + analysis_window_size]))
    
travel_distance = np.array(travel_distance)
travel_distance = np.reshape(travel_distance, [-1, 1])
mocap_data["motion"]["travel_distance"]  = travel_distance
"""

"""
compute statistics in analysis window size for all necessary features
"""

# (DecelerationPeaks)
mocap_data["motion"]["hip_accel_peaks_mean"] = ma.windowed_mean(mocap_data["motion"]["hip_accel_peaks"], analysis_window_size)

# foot hip distance (VFeetHip)
mocap_data["motion"]["foot_hip_dist_min"] = ma.windowed_min(mocap_data["motion"]["foot_hip_dist"], analysis_window_size)
mocap_data["motion"]["foot_hip_dist_max"] = ma.windowed_max(mocap_data["motion"]["foot_hip_dist"], analysis_window_size)
mocap_data["motion"]["foot_hip_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["foot_hip_dist"], analysis_window_size)
mocap_data["motion"]["foot_hip_dist_std"] = ma.windowed_std(mocap_data["motion"]["foot_hip_dist"], analysis_window_size)

# head hip distance (Vtorso)
mocap_data["motion"]["head_hip_dist_min"] = ma.windowed_min(mocap_data["motion"]["head_hip_dist"], analysis_window_size)
mocap_data["motion"]["head_hip_dist_max"] = ma.windowed_max(mocap_data["motion"]["head_hip_dist"], analysis_window_size)
mocap_data["motion"]["head_hip_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["head_hip_dist"], analysis_window_size)
mocap_data["motion"]["head_hip_dist_std"] = ma.windowed_std(mocap_data["motion"]["head_hip_dist"], analysis_window_size)

# hand shoulder distance (VHandsShoulder)
mocap_data["motion"]["hand_shoulder_dist_min"] = ma.windowed_min(mocap_data["motion"]["hand_shoulder_dist"], analysis_window_size)
mocap_data["motion"]["hand_shoulder_dist_max"] = ma.windowed_max(mocap_data["motion"]["hand_shoulder_dist"], analysis_window_size)
mocap_data["motion"]["hand_shoulder_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_shoulder_dist"], analysis_window_size)
mocap_data["motion"]["hand_shoulder_dist_std"] = ma.windowed_std(mocap_data["motion"]["hand_shoulder_dist"], analysis_window_size)

# hand distance (VHands)
mocap_data["motion"]["hand_dist_min"] = ma.windowed_min(mocap_data["motion"]["hand_dist"], analysis_window_size)
mocap_data["motion"]["hand_dist_max"] = ma.windowed_max(mocap_data["motion"]["hand_dist"], analysis_window_size)
mocap_data["motion"]["hand_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_dist"], analysis_window_size)
mocap_data["motion"]["hand_dist_std"] = ma.windowed_std(mocap_data["motion"]["hand_dist"], analysis_window_size)

# hand head distance (VHandsHead)
mocap_data["motion"]["hand_head_dist_min"] = ma.windowed_min(mocap_data["motion"]["hand_head_dist"], analysis_window_size)
mocap_data["motion"]["hand_head_dist_max"] = ma.windowed_max(mocap_data["motion"]["hand_head_dist"], analysis_window_size)
mocap_data["motion"]["hand_head_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_head_dist"], analysis_window_size)
mocap_data["motion"]["hand_head_dist_std"] = ma.windowed_std(mocap_data["motion"]["hand_head_dist"], analysis_window_size)

# hand hip distance (VHandsHip)
mocap_data["motion"]["hand_hip_dist_min"] = ma.windowed_min(mocap_data["motion"]["hand_hip_dist"], analysis_window_size)
mocap_data["motion"]["hand_hip_dist_max"] = ma.windowed_max(mocap_data["motion"]["hand_hip_dist"], analysis_window_size)
mocap_data["motion"]["hand_hip_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_hip_dist"], analysis_window_size)
mocap_data["motion"]["hand_hip_dist_std"] = ma.windowed_std(mocap_data["motion"]["hand_hip_dist"], analysis_window_size)

# hip height (Vhip)
mocap_data["motion"]["hip_height_min"] = ma.windowed_min(mocap_data["motion"]["hip_height"], analysis_window_size)
mocap_data["motion"]["hip_height_max"] = ma.windowed_max(mocap_data["motion"]["hip_height"], analysis_window_size)
mocap_data["motion"]["hip_height_mean"] = ma.windowed_mean(mocap_data["motion"]["hip_height"], analysis_window_size)
mocap_data["motion"]["hip_height_std"] = ma.windowed_std(mocap_data["motion"]["hip_height"], analysis_window_size)

# hip height and foot hip distance combined (VHipFeetHip)
mocap_data["motion"]["hip_height_foot_hip_dist_min"] = ma.windowed_min(mocap_data["motion"]["hip_height_foot_hip_dist"], analysis_window_size)
mocap_data["motion"]["hip_height_foot_hip_dist_max"] = ma.windowed_max(mocap_data["motion"]["hip_height_foot_hip_dist"], analysis_window_size)
mocap_data["motion"]["hip_height_foot_hip_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["hip_height_foot_hip_dist"], analysis_window_size)
mocap_data["motion"]["hip_height_foot_hip_dist_std"] = ma.windowed_std(mocap_data["motion"]["hip_height_foot_hip_dist"], analysis_window_size)

# feet distance (VFeet)
mocap_data["motion"]["feet_dist_min"] = ma.windowed_min(mocap_data["motion"]["feet_dist"], analysis_window_size)
mocap_data["motion"]["feet_dist_max"] = ma.windowed_max(mocap_data["motion"]["feet_dist"], analysis_window_size)
mocap_data["motion"]["feet_dist_mean"] = ma.windowed_mean(mocap_data["motion"]["feet_dist"], analysis_window_size)
mocap_data["motion"]["feet_dist_std"] = ma.windowed_std(mocap_data["motion"]["feet_dist"], analysis_window_size)

# hip velocity (Velocity)
mocap_data["motion"]["hip_vel_min"] = ma.windowed_min(mocap_data["motion"]["hip_vel_scalar"], analysis_window_size)
mocap_data["motion"]["hip_vel_max"] = ma.windowed_max(mocap_data["motion"]["hip_vel_scalar"], analysis_window_size)
mocap_data["motion"]["hip_vel_mean"] = ma.windowed_mean(mocap_data["motion"]["hip_vel_scalar"], analysis_window_size)
mocap_data["motion"]["hip_vel_std"] = ma.windowed_std(mocap_data["motion"]["hip_vel_scalar"], analysis_window_size)

# hands velocity (VelocityHand)
mocap_data["motion"]["hand_vel_min"] = ma.windowed_min(mocap_data["motion"]["hand_vel_scalar"], analysis_window_size)
mocap_data["motion"]["hand_vel_max"] = ma.windowed_max(mocap_data["motion"]["hand_vel_scalar"], analysis_window_size)
mocap_data["motion"]["hand_vel_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_vel_scalar"], analysis_window_size)
mocap_data["motion"]["hand_vel_std"] = ma.windowed_std(mocap_data["motion"]["hand_vel_scalar"], analysis_window_size)

# feet velocity (VelocityFeet)
mocap_data["motion"]["foot_vel_min"] = ma.windowed_min(mocap_data["motion"]["foot_vel_scalar"], analysis_window_size)
mocap_data["motion"]["foot_vel_max"] = ma.windowed_max(mocap_data["motion"]["foot_vel_scalar"], analysis_window_size)
mocap_data["motion"]["foot_vel_mean"] = ma.windowed_mean(mocap_data["motion"]["foot_vel_scalar"], analysis_window_size)
mocap_data["motion"]["foot_vel_std"] = ma.windowed_std(mocap_data["motion"]["foot_vel_scalar"], analysis_window_size)

# hip acceleration (Acceleration)
mocap_data["motion"]["hip_accel_min"] = ma.windowed_min(mocap_data["motion"]["hip_accel_scalar"], analysis_window_size)
mocap_data["motion"]["hip_accel_max"] = ma.windowed_max(mocap_data["motion"]["hip_accel_scalar"], analysis_window_size)
mocap_data["motion"]["hip_accel_mean"] = ma.windowed_mean(mocap_data["motion"]["hip_accel_scalar"], analysis_window_size)
mocap_data["motion"]["hip_accel_std"] = ma.windowed_std(mocap_data["motion"]["hip_accel_scalar"], analysis_window_size)

# hands acceleration (AccelerationHand)
mocap_data["motion"]["hand_accel_min"] = ma.windowed_min(mocap_data["motion"]["hand_accel_scalar"], analysis_window_size)
mocap_data["motion"]["hand_accel_max"] = ma.windowed_max(mocap_data["motion"]["hand_accel_scalar"], analysis_window_size)
mocap_data["motion"]["hand_accel_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_accel_scalar"], analysis_window_size)
mocap_data["motion"]["hand_accel_std"] = ma.windowed_std(mocap_data["motion"]["hand_accel_scalar"], analysis_window_size)

# feet acceleration (AccelerationFeet)
mocap_data["motion"]["foot_accel_min"] = ma.windowed_min(mocap_data["motion"]["foot_accel_scalar"], analysis_window_size)
mocap_data["motion"]["foot_accel_max"] = ma.windowed_max(mocap_data["motion"]["foot_accel_scalar"], analysis_window_size)
mocap_data["motion"]["foot_accel_mean"] = ma.windowed_mean(mocap_data["motion"]["foot_accel_scalar"], analysis_window_size)
mocap_data["motion"]["foot_accel_std"] = ma.windowed_std(mocap_data["motion"]["foot_accel_scalar"], analysis_window_size)

# hip jerk (Jerk)
mocap_data["motion"]["hip_jerk_min"] = ma.windowed_min(mocap_data["motion"]["hip_jerk_scalar"], analysis_window_size)
mocap_data["motion"]["hip_jerk_max"] = ma.windowed_max(mocap_data["motion"]["hip_jerk_scalar"], analysis_window_size)
mocap_data["motion"]["hip_jerk_mean"] = ma.windowed_mean(mocap_data["motion"]["hip_jerk_scalar"], analysis_window_size)
mocap_data["motion"]["hip_jerk_std"] = ma.windowed_std(mocap_data["motion"]["hip_jerk_scalar"], analysis_window_size)

# hand level upper (HandsLevelUpper)
mocap_data["motion"]["hand_level_upper_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_level_upper"], analysis_window_size)

# hand level middle (HandsLevelMiddle)
mocap_data["motion"]["hand_level_middle_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_level_middle"], analysis_window_size)

# hand level below (HandsLevelLow)
mocap_data["motion"]["hand_level_below_mean"] = ma.windowed_mean(mocap_data["motion"]["hand_level_below"], analysis_window_size)


# volume_5 (volume_5)
mocap_data["motion"]["volume_5_min"] = ma.windowed_min(mocap_data["motion"]["volume_5"], analysis_window_size)
mocap_data["motion"]["volume_5_max"] = ma.windowed_max(mocap_data["motion"]["volume_5"], analysis_window_size)
mocap_data["motion"]["volume_5_mean"] = ma.windowed_mean(mocap_data["motion"]["volume_5"], analysis_window_size)
mocap_data["motion"]["volume_5_std"] = ma.windowed_std(mocap_data["motion"]["volume_5"], analysis_window_size)

# volume_all (volume_all)
mocap_data["motion"]["volume_all_min"] = ma.windowed_min(mocap_data["motion"]["volume_all"], analysis_window_size)
mocap_data["motion"]["volume_all_max"] = ma.windowed_max(mocap_data["motion"]["volume_all"], analysis_window_size)
mocap_data["motion"]["volume_all_mean"] = ma.windowed_mean(mocap_data["motion"]["volume_all"], analysis_window_size)
mocap_data["motion"]["volume_all_std"] = ma.windowed_std(mocap_data["motion"]["volume_all"], analysis_window_size)

# volume_all (Vtorso)
mocap_data["motion"]["volume_all_min"] = ma.windowed_min(mocap_data["motion"]["volume_all"], analysis_window_size)
mocap_data["motion"]["volume_all_max"] = ma.windowed_max(mocap_data["motion"]["volume_all"], analysis_window_size)
mocap_data["motion"]["volume_all_mean"] = ma.windowed_mean(mocap_data["motion"]["volume_all"], analysis_window_size)
mocap_data["motion"]["volume_all_std"] = ma.windowed_std(mocap_data["motion"]["volume_all"], analysis_window_size)

# travel area (AreaCovered)
# this doesn't make any sense with the current implementation where the travel area is already windowed
"""
mocap_data["motion"]["area_covered_min"] = ma.windowed_min(mocap_data["motion"]["area_covered"], analysis_window_size)
mocap_data["motion"]["area_covered_max"] = ma.windowed_max(mocap_data["motion"]["area_covered"], analysis_window_size)
mocap_data["motion"]["area_covered_mean"] = ma.windowed_mean(mocap_data["motion"]["area_covered"], analysis_window_size)
mocap_data["motion"]["area_covered_std"] = ma.windowed_std(mocap_data["motion"]["area_covered"], analysis_window_size)
"""

# volume_upper (volume_upper)
mocap_data["motion"]["volume_upper_min"] = ma.windowed_min(mocap_data["motion"]["volume_upper"], analysis_window_size)
mocap_data["motion"]["volume_upper_max"] = ma.windowed_max(mocap_data["motion"]["volume_upper"], analysis_window_size)
mocap_data["motion"]["volume_upper_mean"] = ma.windowed_mean(mocap_data["motion"]["volume_upper"], analysis_window_size)
mocap_data["motion"]["volume_upper_std"] = ma.windowed_std(mocap_data["motion"]["volume_upper"], analysis_window_size)

# volume_lower (volume_lower)
mocap_data["motion"]["volume_lower_min"] = ma.windowed_min(mocap_data["motion"]["volume_lower"], analysis_window_size)
mocap_data["motion"]["volume_lower_max"] = ma.windowed_max(mocap_data["motion"]["volume_lower"], analysis_window_size)
mocap_data["motion"]["volume_lower_mean"] = ma.windowed_mean(mocap_data["motion"]["volume_lower"], analysis_window_size)
mocap_data["motion"]["volume_lower_std"] = ma.windowed_std(mocap_data["motion"]["volume_lower"], analysis_window_size)

# volume_right (volume_right)
mocap_data["motion"]["volume_right_min"] = ma.windowed_min(mocap_data["motion"]["volume_right"], analysis_window_size)
mocap_data["motion"]["volume_right_max"] = ma.windowed_max(mocap_data["motion"]["volume_right"], analysis_window_size)
mocap_data["motion"]["volume_right_mean"] = ma.windowed_mean(mocap_data["motion"]["volume_right"], analysis_window_size)
mocap_data["motion"]["volume_right_std"] = ma.windowed_std(mocap_data["motion"]["volume_right"], analysis_window_size)

# volume_left (volume_left)
mocap_data["motion"]["volume_left_min"] = ma.windowed_min(mocap_data["motion"]["volume_left"], analysis_window_size)
mocap_data["motion"]["volume_left_max"] = ma.windowed_max(mocap_data["motion"]["volume_left"], analysis_window_size)
mocap_data["motion"]["volume_left_mean"] = ma.windowed_mean(mocap_data["motion"]["volume_left"], analysis_window_size)
mocap_data["motion"]["volume_left_std"] = ma.windowed_std(mocap_data["motion"]["volume_left"], analysis_window_size)


"""
Gather all final mocap features
"""

mocap_features = {}

# max(VFeetHip)
mocap_features["foot_hip_dist_max"] = mocap_data["motion"]["foot_hip_dist_max"]

# min(VFeetHip)
mocap_features["foot_hip_dist_min"] = mocap_data["motion"]["foot_hip_dist_min"]

# std(VFeetHip)
mocap_features["foot_hip_dist_std"] = mocap_data["motion"]["foot_hip_dist_std"]

# mean(VFeetHip)
mocap_features["foot_hip_dist_mean"] = mocap_data["motion"]["foot_hip_dist_mean"]

# max(VHandsShoulder)
mocap_features["hand_shoulder_dist_max"] = mocap_data["motion"]["hand_shoulder_dist_max"]

# min(VHandsShoulder)
mocap_features["hand_shoulder_dist_min"] = mocap_data["motion"]["hand_shoulder_dist_min"]

# std(VHandsShoulder)
mocap_features["hand_shoulder_dist_std"] = mocap_data["motion"]["hand_shoulder_dist_std"]

# mean(VHandsShoulder)
mocap_features["hand_shoulder_dist_mean"] = mocap_data["motion"]["hand_shoulder_dist_mean"]

# max(VHands)
mocap_features["hand_dist_max"] = mocap_data["motion"]["hand_dist_max"]

# min(VHands)
mocap_features["hand_dist_min"] = mocap_data["motion"]["hand_dist_min"]

# std(VHands)
mocap_features["hand_dist_std"] = mocap_data["motion"]["hand_dist_std"]

# mean(VHands)
mocap_features["hand_dist_mean"] = mocap_data["motion"]["hand_dist_mean"]

# max(VHandsHead)
mocap_features["hand_head_dist_max"] = mocap_data["motion"]["hand_head_dist_max"]

# min(VHandsHead)
mocap_features["hand_head_dist_min"] = mocap_data["motion"]["hand_head_dist_min"]

# std(VHandsHead)
mocap_features["hand_head_dist_std"] = mocap_data["motion"]["hand_head_dist_std"]

# mean(VHandsHead)
mocap_features["hand_head_dist_mean"] = mocap_data["motion"]["hand_head_dist_mean"]

# max(VHandsHip)
mocap_features["hand_hip_dist_max"] = mocap_data["motion"]["hand_hip_dist_max"]

# min(VHandsHip)
mocap_features["hand_hip_dist_min"] = mocap_data["motion"]["hand_hip_dist_min"]

# std(VHandsHip)
mocap_features["hand_hip_dist_std"] = mocap_data["motion"]["hand_hip_dist_std"]

# mean(VHandsHip)
mocap_features["hand_hip_dist_mean"] = mocap_data["motion"]["hand_hip_dist_mean"]

# max(Vhip)
mocap_features["hip_height_max"] = mocap_data["motion"]["hip_height_max"]

# min(Vhip)
mocap_features["hip_height_min"] = mocap_data["motion"]["hip_height_min"]

# std(Vhip)
mocap_features["hip_height_std"] = mocap_data["motion"]["hip_height_std"]

# mean(Vhip)
mocap_features["hip_height_mean"] = mocap_data["motion"]["hip_height_mean"]

# max(VHipFeetHip)
mocap_features["hip_height_foot_hip_dist_max"] = mocap_data["motion"]["hip_height_foot_hip_dist_max"]

# min(VHipFeetHip)
mocap_features["hip_height_foot_hip_dist_min"] = mocap_data["motion"]["hip_height_foot_hip_dist_min"]

# std(VHipFeetHip)
mocap_features["hip_height_foot_hip_dist_std"] = mocap_data["motion"]["hip_height_foot_hip_dist_std"]

# mean(VHipFeetHip)
mocap_features["hip_height_foot_hip_dist_mean"] = mocap_data["motion"]["hip_height_foot_hip_dist_mean"]

# max(VFeet)
mocap_features["feet_dist_max"] = mocap_data["motion"]["feet_dist_max"]

# min(VFeet)
mocap_features["feet_dist_min"] = mocap_data["motion"]["feet_dist_min"]

# std(VFeet)
mocap_features["feet_dist_std"] = mocap_data["motion"]["feet_dist_std"]

# mean(VFeet)
mocap_features["feet_dist_mean"] = mocap_data["motion"]["feet_dist_mean"]

# DecelerationPeaks
mocap_features["hip_accel_peaks_mean"] = mocap_data["motion"]["hip_accel_peaks_mean"]

# max(Velocity)
mocap_features["hip_vel_max"] = mocap_data["motion"]["hip_vel_max"]

# std(Velocity)
mocap_features["hip_vel_std"] = mocap_data["motion"]["hip_vel_std"]

# mean(Velocity)
mocap_features["hip_vel_mean"] = mocap_data["motion"]["hip_vel_mean"]

# max(VelocityHand)
mocap_features["hand_vel_max"] = mocap_data["motion"]["hand_vel_max"]

# std(VelocityHand)
mocap_features["hand_vel_std"] = mocap_data["motion"]["hand_vel_std"]

# mean(VelocityHand)
mocap_features["hand_vel_mean"] = mocap_data["motion"]["hand_vel_mean"]

# max(VelocityFeet)
mocap_features["foot_vel_max"] = mocap_data["motion"]["foot_vel_max"]

# std(VelocityFeet)
mocap_features["foot_vel_std"] = mocap_data["motion"]["foot_vel_std"]

# mean(VelocityFeet)
mocap_features["foot_vel_mean"] = mocap_data["motion"]["foot_vel_mean"]

# max(abs(Acceleration))
mocap_features["hip_accel_max"] = mocap_data["motion"]["hip_accel_max"]

# std(Acceleration)
mocap_features["hip_accel_std"] = mocap_data["motion"]["hip_accel_std"]

# max(abs(AccelerationHand))
mocap_features["hand_accel_max"] = mocap_data["motion"]["hand_accel_max"]

# std(AccelerationHand)
mocap_features["hand_accel_std"] = mocap_data["motion"]["hand_accel_std"]

# max(abs(AccelerationFeet))
mocap_features["foot_accel_max"] = mocap_data["motion"]["foot_accel_max"]

# std(AccelerationFeet)
mocap_features["foot_accel_std"] = mocap_data["motion"]["foot_accel_std"]

# max(abs(Jerk))
mocap_features["hip_jerk_max"] = mocap_data["motion"]["hip_jerk_max"]

# std(Jerk)
mocap_features["hip_jerk_std"] = mocap_data["motion"]["hip_jerk_std"]

# max(volume_5)
mocap_features["volume_5_max"] = mocap_data["motion"]["volume_5_max"]

# min(volume_5)
mocap_features["volume_5_min"] = mocap_data["motion"]["volume_5_min"]

# std(volume_5)
mocap_features["volume_5_std"] = mocap_data["motion"]["volume_5_std"]

# mean(volume_5)
mocap_features["volume_5_mean"] = mocap_data["motion"]["volume_5_mean"]

# max(volume_all)
mocap_features["volume_all_max"] = mocap_data["motion"]["volume_all_max"]

# min(volume_all)
mocap_features["volume_all_min"] = mocap_data["motion"]["volume_all_min"]

# std(volume_all)
mocap_features["volume_all_std"] = mocap_data["motion"]["volume_all_std"]

# mean(volume_all)
mocap_features["volume_all_mean"] = mocap_data["motion"]["volume_all_mean"]

# max(Vtorso)
mocap_features["head_hip_dist_max"] = mocap_data["motion"]["head_hip_dist_max"]

# min(Vtorso)
mocap_features["head_hip_dist_min"] = mocap_data["motion"]["head_hip_dist_min"]

# std(Vtorso)
mocap_features["head_hip_dist_std"] = mocap_data["motion"]["head_hip_dist_std"]

# mean(Vtorso)
mocap_features["head_hip_dist_mean"] = mocap_data["motion"]["head_hip_dist_mean"]

# HandsLevelUpper
mocap_features["hand_level_upper_mean"] = mocap_data["motion"]["hand_level_upper_mean"]

# HandsLevelMiddle
mocap_features["hand_level_middle_mean"] = mocap_data["motion"]["hand_level_middle_mean"]

# HandsLevelLow
mocap_features["hand_level_below_mean"] = mocap_data["motion"]["hand_level_below_mean"]

# Distance
mocap_features["travel_distance"] = mocap_data["motion"]["travel_distance"]

# AreaCovered
mocap_features["area_covered"] = mocap_data["motion"]["area_covered"]

"""
# max(AreaCovered)
mocap_features["area_covered_max"] = mocap_data["motion"]["area_covered_max"]

# std(AreaCovered)
mocap_features["area_covered_std"] = mocap_data["motion"]["area_covered_std"]

# mean(AreaCovered)
mocap_features["area_covered_mean"] = mocap_data["motion"]["area_covered_mean"]
"""

# max(volume_upper)
mocap_features["volume_upper_max"] = mocap_data["motion"]["volume_upper_max"]

# min(volume_upper)
mocap_features["volume_upper_min"] = mocap_data["motion"]["volume_upper_min"]

# std(volume_upper)
mocap_features["volume_upper_std"] = mocap_data["motion"]["volume_upper_std"]

# mean(volume_upper)
mocap_features["volume_upper_mean"] = mocap_data["motion"]["volume_upper_mean"]

# max(volume_lower)
mocap_features["volume_lower_max"] = mocap_data["motion"]["volume_lower_max"]

# min(volume_lower)
mocap_features["volume_lower_min"] = mocap_data["motion"]["volume_lower_min"]

# std(volume_lower)
mocap_features["volume_lower_std"] = mocap_data["motion"]["volume_lower_std"]

# mean(volume_lower)
mocap_features["volume_lower_mean"] = mocap_data["motion"]["volume_lower_mean"]

# max(volume_right)
mocap_features["volume_right_max"] = mocap_data["motion"]["volume_right_max"]

# min(volume_right)
mocap_features["volume_right_min"] = mocap_data["motion"]["volume_right_min"]

# std(volume_right)
mocap_features["volume_right_std"] = mocap_data["motion"]["volume_right_std"]

# mean(volume_right)
mocap_features["volume_right_mean"] = mocap_data["motion"]["volume_right_mean"]

# max(volume_left)
mocap_features["volume_left_max"] = mocap_data["motion"]["volume_left_max"]

# min(volume_left)
mocap_features["volume_left_min"] = mocap_data["motion"]["volume_left_min"]

# std(volume_left)
mocap_features["volume_left_std"] = mocap_data["motion"]["volume_left_std"]

# mean(volume_left)
mocap_features["volume_left_mean"] = mocap_data["motion"]["volume_left_mean"]



"""
Save results
"""

np.save('mocap_features.npy', mocap_features) 


"""
Plot Results
"""

frame_count = mocap_features["foot_hip_dist_max"].shape[0]
feature_count = len(list(mocap_features.keys()))

plot_x = np.arange(frame_count)

for feature_name, feature_values in mocap_features.items():
    
    plot_y = feature_values
    
    plt.plot(plot_x, plot_y)
    plt.xlabel("frames")
    plt.ylabel(feature_name)
    plt.show() 
    

