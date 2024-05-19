import glob
import os
import numpy as np
import torch
from tqdm import tqdm
from dataset.quaternion import ax_to_6v




def slice_and_process_motion(joint_orientations, root_translation, frame_rate, target_frame_rate, stride, length, out_dir, file_name):
    start_idx = 0
    window = int(length * frame_rate)
    stride_step = int(stride * frame_rate)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= joint_orientations.shape[0] - window:
        joint_orientations_slice, root_translation_slice = (
            joint_orientations[start_idx : start_idx + window],
            root_translation[start_idx : start_idx + window],
        )
        joint_orientations_slice = torch.tensor(joint_orientations_slice).reshape(joint_orientations_slice.shape[0],-1,3)[:,:22,:]
        joint_orientations_slice = ax_to_6v(joint_orientations_slice).numpy()
        if joint_orientations_slice.min() < -1.01 or joint_orientations_slice.max() > 1.01:
            print('joint_orientations_slice out of range:: ' + file_name)

        # downsample joint_orientations_slice and root_translation_slice from the frame_rate to 20Hz using linear interpolation without using numpy.interp
        # TODO: implement with SLERP instead of linear interpolation (should not matter all that much)
        joint_orientations_slice_resampled = np.zeros((int(length * target_frame_rate), joint_orientations_slice.shape[1], joint_orientations_slice.shape[2]))
        root_translation_slice_resampled = np.zeros((int(length * target_frame_rate), root_translation_slice.shape[1]))
        for i in range(joint_orientations_slice.shape[1]):
            for j in range(joint_orientations_slice.shape[2]):
                joint_orientations_slice_resampled[:,i,j] = np.interp(np.linspace(0, length, int(length * target_frame_rate)), np.linspace(0, length, window), joint_orientations_slice[:,i,j])
        for i in range(root_translation_slice.shape[1]):
            root_translation_slice_resampled[:,i] = np.interp(np.linspace(0, length, int(length * target_frame_rate)), np.linspace(0, length, window), root_translation_slice[:,i])

        joint_orientations_slice_resampled = torch.tensor(joint_orientations_slice_resampled.reshape(joint_orientations_slice_resampled.shape[0],-1))
        root_translation_slice_resampled = torch.tensor(root_translation_slice_resampled)


        pose = torch.cat((root_translation_slice_resampled, joint_orientations_slice_resampled), axis=1).float()
        if pose[...,3:].min() < -1.01 or pose[...,3:].max() > 1.01:
            print('joint_orientations_slice out of range:: ' + file_name)

        # shift root position to start in (x,y) = (0,0)
        pose[:,0] = pose[:,0].clone() - pose[0,0].clone()
        pose[:,1] = pose[:,1].clone() - pose[0,1].clone()


        torch.save(pose, f"{out_dir}/{file_name}_slice{slice_count}.pt")
        start_idx += stride_step
        slice_count += 1
    return slice_count

def slice_AMASS(motion_dir, stride=1, length=3, target_frame_rate=20):
    """
    motion_dir: directory containing AMASS motion files
    stride: stride of the slicing window every time a slice is taken
    length: length of slicing window in seconds
    """
    out_dir = motion_dir + "_sliced"
    os.makedirs(out_dir, exist_ok=True)
    motions = sorted(glob.glob(f"{motion_dir}/*.npz"))
    motion_out = motion_dir + "_sliced"
    os.makedirs(motion_out, exist_ok=True)
    total_original_frames = 0
    total_sliced_frames = 0
    for motion in tqdm(motions):
        data = np.load(motion)
        # check if poses exists in data
        if 'poses' not in data.keys():
            print('No poses in data:: ' + motion)
        else:
            joint_orientations = data['poses']
            root_translation = data['trans']
            if 'mocap_framerate' in data.files:
                frame_rate = data['mocap_framerate']
            elif 'mocap_frame_rate' in data.files:
                frame_rate = data['mocap_frame_rate']
            else:
                # throw error
                raise ValueError('No frame rate in data:: ' + motion)
            number_of_slices = slice_and_process_motion(joint_orientations, root_translation, frame_rate, target_frame_rate, stride, length, out_dir, motion.split('/')[-1][:-4])
            total_original_frames += joint_orientations.shape[0]
            total_sliced_frames += number_of_slices * 60
    print(f"Total original frames: {total_original_frames}")
    print(f"Total original time [s]: {total_original_frames/frame_rate}")
    print(f"Total sliced frames [s]: {total_sliced_frames/target_frame_rate}")