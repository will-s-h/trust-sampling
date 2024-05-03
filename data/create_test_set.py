import argparse
import os
from pathlib import Path

from slice import *
from aggregate import *
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def create_dataset():
    # agreggate AMASS dataset
    path_AMASS = os.path.join(os.path.dirname(__file__),'AMASS_test')
    path_AMASS_agreggated = os.path.join(os.path.dirname(__file__), 'AMASS_test_aggregated')
    Path(path_AMASS_agreggated).mkdir(parents=True, exist_ok=True)
    aggregate(path_AMASS, path_AMASS_agreggated)

    # slice motions into sliding windows to create training dataset
    print("Slicing train data")
    slice_AMASS(path_AMASS_agreggated, stride=3, length=3, target_frame_rate=20) # no overlapping slices

def visualize_n_samples(n, path_AMASS_agreggated_sliced, render_dir):
    # Specific imports
    from model.motion_wrapper import MotionWrapper
    from demo_motion.motion_args import parse_test_opt
    from visualization.render_motion import SMPLSkeleton, just_render_simple_long_clip
    import random

    opt = parse_test_opt()
    opt.motion_save_dir = "./motions"
    opt.render_dir = render_dir

    opt.save_motions = False
    opt.no_render = False
    opt.predict_contact = True
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smpl = SMPLSkeleton(device)

    # select n random files in path_AMASS_agreggated_sliced
    files = os.listdir(path_AMASS_agreggated_sliced)
    samples = []
    for i in range(n):
        file = random.choice(files)
        samples.append(torch.load(os.path.join(path_AMASS_agreggated_sliced, file)))
    samples = torch.stack(samples)
    just_render_simple_long_clip(smpl, samples, normalizer=None, render_out=opt.render_dir, constraint=None)

def samples_to_poses(samples):
    from visualization.render_motion import SMPLSkeleton
    from dataset.quaternion import ax_from_6v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPLSkeleton(device)
    # discard contact
    if samples.shape[2] != 135:
        _, samples = torch.split(
            samples, (4, samples.shape[2] - 4), dim=2
        )

    # do the FK all at once
    b, s, c = samples.shape
    pos = samples[:, :, :3].to('cuda')
    q = samples[:, :, 3:].reshape(b, s, 22, 6)
    # go 6d to ax
    q = ax_from_6v(q).to('cuda')
    poses = smpl.forward(q, pos)
    return poses

def energy_analysis(path_AMASS_agreggated_sliced, plot=False):

    # select n random files in path_AMASS_agreggated_sliced
    files = os.listdir(path_AMASS_agreggated_sliced)
    samples = []
    for file in files:
        samples.append(torch.load(os.path.join(path_AMASS_agreggated_sliced, file)))
    samples = torch.stack(samples)

    # do some analysis on the energy of the motions
    poses = samples_to_poses(samples)
    energy = torch.sum((poses[:, 1: ,...] - poses[:, :-1 ,...]) ** 2, dim=(1,2,3))
    if plot:
        import matplotlib.pyplot as plt
        # histogram
        plt.hist(energy.cpu().detach().numpy(), bins=50)
        plt.show()

    slice_energy_dict = {}
    for i, file in enumerate(files):
        slice_energy_dict[file] = energy[i].item()

    return slice_energy_dict

def split_dataset_based_on_energy(slice_energy_dict, path_AMASS_agreggated_sliced, percentage=0.8):
    # split dataset based on energy
    # sort slice_energy_dict by energy
    sorted_slice_energy_dict = {k: v for k, v in sorted(slice_energy_dict.items(), key=lambda item: item[1])}

    # split dataset based on energy
    split_idx = int(len(sorted_slice_energy_dict) * percentage)
    low_energy_files = list(sorted_slice_energy_dict.keys())[:split_idx]
    high_energy_files = list(sorted_slice_energy_dict.keys())[split_idx:]

    # move files to new directories
    low_energy_dir = os.path.join(os.path.dirname(__file__), 'AMASS_test_aggregated_sliced_low_energy')
    high_energy_dir = os.path.join(os.path.dirname(__file__), 'AMASS_test_aggregated_sliced_high_energy')
    Path(low_energy_dir).mkdir(parents=True, exist_ok=True)
    Path(high_energy_dir).mkdir(parents=True, exist_ok=True)
    for file in low_energy_files:
        shutil.copy(os.path.join(path_AMASS_agreggated_sliced, file), os.path.join(low_energy_dir, file))
    for file in high_energy_files:
        shutil.copy(os.path.join(path_AMASS_agreggated_sliced, file), os.path.join(high_energy_dir, file))


    return


if __name__ == "__main__":
    create_dataset()
    # path_AMASS_agreggated_sliced = os.path.join(os.path.dirname(__file__), 'AMASS_test_aggregated_sliced')
    # #
    # # # # split dataset based on energy
    # slice_energy_dict = energy_analysis(path_AMASS_agreggated_sliced, plot=False)
    # split_dataset_based_on_energy(slice_energy_dict, path_AMASS_agreggated_sliced)
    #
    #
    # path_AMASS_agreggated_sliced_high_energy = os.path.join(os.path.dirname(__file__), 'AMASS_test_aggregated_sliced_high_energy')
    # n = 10
    # render_dir = "renders/examples_gt_high_energy_1"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_high_energy, render_dir)
    # render_dir = "renders/examples_gt_high_energy_2"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_high_energy, render_dir)
    # render_dir = "renders/examples_gt_high_energy_3"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_high_energy, render_dir)
    # render_dir = "renders/examples_gt_high_energy_4"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_high_energy, render_dir)
    # render_dir = "renders/examples_gt_high_energy_5"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_high_energy, render_dir)
    #
    #
    # path_AMASS_agreggated_sliced_low_energy = os.path.join(os.path.dirname(__file__), 'AMASS_test_aggregated_sliced_low_energy')
    # n = 10
    # render_dir = "renders/examples_gt_low_energy_1"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_low_energy, render_dir)
    # render_dir = "renders/examples_gt_low_energy_2"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_low_energy, render_dir)
    # render_dir = "renders/examples_gt_low_energy_3"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_low_energy, render_dir)
    # render_dir = "renders/examples_gt_low_energy_4"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_low_energy, render_dir)
    # render_dir = "renders/examples_gt_low_energy_5"
    # visualize_n_samples(n, path_AMASS_agreggated_sliced_low_energy, render_dir)