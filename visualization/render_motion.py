import os
from pathlib import Path
from tempfile import TemporaryDirectory
# import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
# import soundfile as sf
import torch
from matplotlib import cm
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_apply, quaternion_multiply
# from p_tqdm import p_map

from dataset.quaternion import ax_from_6v


smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact, text, colors):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    # plot frame number in animation
    text.set_text(f'Frame {num + 1}')
    if num in colors:
        text.set_color(colors[num])

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


def skeleton_render(
    poses,
    out="renders",
    contact=None,
    colors=None,
    constraint=None,
):
    # generate the pose with FK
    Path(out).mkdir(parents=True, exist_ok=True)
    num_steps = poses.shape[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", computed_zorder=False)  # computed_zorder kwarg requires matplotlib >= 3.5.0
    
    point = np.array([0, 0, 1])
    normal = np.array([0, 0, 1])
    d = -point.dot(normal)
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
    # plot the plane
    ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
    # Create lines initially without data
    lines = [
        ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
        for _ in smpl_parents
    ]
    scat = [
        # ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
        ax.scatter([], [], [], zorder=10, s=10)

        for _ in range(4)
    ]
    axrange = 3

    # add text, colors, constraint
    text = ax.text2D(0, 0, "Frame 0", transform=ax.transAxes, zorder=100)
    colors = colors if isinstance(colors, dict) else {}
    if constraint is not None and hasattr(constraint, "plot"):
        constraint.plot(fig, ax)

    # create contact labels
    if contact is None:
        # the current heuristic says that the feet are in contact 
        feet = poses[:, (7, 8, 10, 11)]
        feetv = np.zeros(feet.shape[:2])
        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
        HEURISTIC = 0.02  # in original code, was 0.01
        contact = feetv < HEURISTIC
    else:
        contact = contact > 0.95
    # Creating the Animation object
    anim = animation.FuncAnimation(
        fig,
        plot_single_pose,
        num_steps,
        fargs=(poses, lines, ax, axrange, scat, contact, text, colors),
        interval=1000 // 20,
    )

    # actually save the gif
    path = os.path.normpath(out)
    pathparts = path.split(os.sep)
    # previously, this was done with writer = animation.HTMLWriter(fps=20), and with .motion.html
    # ffmpeg may require updates; to do so, run `conda update ffmpeg` in command line
    gifname = os.path.join(out, f"{pathparts[-1]}" + ".mp4")
    anim.save(gifname, writer='ffmpeg', fps=20)
    plt.close()


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        # transform from axis angle to quaternion
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

### just_render_simple: most abstracted function, just renders the skeleton into an .mp4

def just_render_simple(
    smpl: SMPLSkeleton, 
    model_output,
    normalizer,
    render_out,
    colors=None,
    constraint=None):
    
    samples = model_output
    samples = normalizer.unnormalize(samples)

    if samples.shape[2] != 135:
        sample_contact, samples = torch.split(
            samples, (4, samples.shape[2] - 4), dim=2
        )
        sample_contact = sample_contact.detach().cpu()
    else:
        sample_contact = None
    
    # do the FK all at once
    b, s, c = samples.shape
    pos = samples[:, :, :3].to('cuda')  # np.zeros((sample.shape[0], 3))
    q = samples[:, :, 3:].reshape(b, s, 22, 6)
    # go 6d to ax
    q = ax_from_6v(q).to('cuda')

    poses = smpl.forward(q, pos).detach().cpu().numpy()
    
    def inner(xx):
        num, pose = xx

        contact = sample_contact[num] if sample_contact is not None else None
        skeleton_render(
            pose,
            out=render_out,
            contact=contact,
            colors=colors,
            constraint=constraint
        )

    for xx in enumerate(poses):
        inner(xx)
        
    # p_map(inner, enumerate(poses))
    

#### trajectory animation across diffusion time (see how trajectory changes over diffusion t)
def trajectory_animation(traj, desired_traj):
    # traj should be a list of tuples, each of format (torch.shape[60, 2], int time, text description)
    fig, ax = plt.subplots()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    line, = ax.plot([], [])
    ax.plot(desired_traj[:,0].cpu(), desired_traj[:,1].cpu())      # plot expected trajectory
    frame_text = ax.text(0.02, 1.03, '', transform=ax.transAxes)
    
    def update(frame):
        np_traj = traj[frame][0].numpy()
        x, y = np_traj[:, 0], np_traj[:, 1]
        line.set_ydata(y)
        line.set_xdata(x)
        frame_text.set_text(f'diffusion t: {traj[frame][1]}, {traj[frame][2]}')
        return line, frame_text
    
    ani = animation.FuncAnimation(fig, update, frames=len(traj), blit=True)
    return ani