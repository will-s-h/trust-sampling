import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh

from dataset.quaternion import ax_from_6v

from .body_model import SMPLH_PATH, BodyModel
from .mesh_viewer import MeshViewer, colors

SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
smpl_connections = [[11, 8], [8, 5], [5, 2], [2, 0], [10, 7], [7, 4], [4, 1], [1, 0], 
                [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [12, 13], [13, 16], [16, 18], 
                [18, 20], [12, 14], [14, 17], [17, 19], [19, 21]]
J = len(SMPL_JOINTS)   # 22

def c2c(tensor):
    return tensor.detach().cpu().numpy()


def just_render_smpl_seq(
    model_output,
    normalizer,
    render_out,
    colors=None,
    constraint=None):

    os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
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
    B, S, _ = samples.shape
    pos = samples[:, :, :3].to('cuda')  # (b, s, 3)
    q = samples[:, :, 3:].reshape(B, S, 22, 6)
    # go 6d to ax
    q = ax_from_6v(q).to('cuda')     # (b, s, 22, 3)
    root_orient = q[:, :, 0, :]         # (b, s, 3)
    pose_body = q[:, :, 1:, :].reshape(B, S, -1)  # (b, s, 21*3)


    bm_path = os.path.join(SMPLH_PATH, 'neutral/model.npz')
    bm_world = BodyModel(bm_path=bm_path, num_betas=16, batch_size=S).to(q)

    cur_offscreen = render_out is not None
    print("is cur_offscreen ", cur_offscreen)
    if cur_offscreen:
        Path(render_out).mkdir(parents=True, exist_ok=True)
        out_path_list = [render_out + '_samp%d' % (samp_idx) for samp_idx in range(B)]
    else:
        out_path_list = [None for _ in range(B)]

    for b in range(B):
        body_pred = bm_world(
            pose_body=pose_body[b],
            pose_hand=None,
            betas=torch.zeros((S, 16)).to(pose_body),
            root_orient=root_orient[b],
            trans=pos[b]
        )
        # viz_joints = pred_world_joints[b] + torch.tensor([-0.8, -0.8, 0.0]).repeat((T, J, 1)).to(pred_world_joints)

        body_alpha = 1.0
        viz_smpl_seq(
            body_pred,
            imw=1600, imh=1000, fps=20,
            render_body=True,
            render_joints=None,
            render_skeleton=None,
            render_ground=True,
            contacts=None,
            joints_seq=None,    # viz_joints
            body_alpha=body_alpha,
            use_offscreen=cur_offscreen,
            out_path=out_path_list[b],
            wireframe=False,
            RGBA=False,
            follow_camera=True,
            cam_offset=[0.0, 2.2, 0.9],
            joint_color=[0.0, 1.0, 0.0],
            point_color=[0.0, 0.0, 1.0],
            skel_color=[0.5, 0.5, 0.5],
            joint_rad=0.015,
            point_rad=0.015,
            # force_seq=force_seq,
            # extra_joint_seq=viz_extra_joints,
            # projectiles_seq=projectiles_seq,
            # other_viz_locs_seq=other_viz_locs_seq,
            # ground_plane_seq=plane_viz_seq
        )

        if cur_offscreen:
            create_video(out_path_list[b] + '/frame_%08d.' + '%s' % 'png', out_path_list[b] + '.mp4', 30)


def viz_smpl_seq(body, imw=1080, imh=1080, fps=30, contacts=None,
                render_body=True, render_joints=False, render_skeleton=False, render_ground=True, ground_plane_seq=None,
                use_offscreen=False, out_path=None, wireframe=False, RGBA=False,
                joints_seq=None, joints_vel=None, follow_camera=False, vtx_list=None, points_seq=None, points_vel=None,
                static_meshes=None, camera_intrinsics=None, img_seq=None, point_rad=0.015,
                skel_connections=smpl_connections, img_extn='png', ground_alpha=1.0, body_alpha=None, mask_seq=None,
                cam_offset=[0.0, 4.0, 1.25], ground_color0=[0.8, 0.9, 0.9], ground_color1=[0.6, 0.7, 0.7],
                skel_color=[0.0, 0.0, 1.0],
                joint_rad=0.015,
                point_color=[0.0, 1.0, 0.0],
                joint_color=[0.0, 1.0, 0.0],
                contact_color=[1.0, 0.0, 0.0],
                render_bodies_static=None,
                render_points_static=None,
                cam_rot=None,
                force_seq=None,
                force_color=[0.0, 0.0, 1.0],
                extra_joint_seq=None,
                extra_joint_color=[0.8, 0.0, 0.0],
                projectiles_seq=None,
                other_viz_locs_seq=None,
                second_body=None,
    ):
    '''
    Visualizes the body model output of a smpl sequence.
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    '''

    if contacts is not None and torch.is_tensor(contacts):
        contacts = c2c(contacts)

    if render_body or vtx_list is not None:
        nv = body.v.size(1)
        vertex_colors = np.tile(colors['grey'], (nv, 1))
        if body_alpha is not None:
            vtx_alpha = np.ones((vertex_colors.shape[0], 1))*body_alpha
            vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
        faces = c2c(body.f)
        body_mesh_seq = [trimesh.Trimesh(vertices=c2c(body.v[i]), faces=faces, vertex_colors=vertex_colors, process=False) for i in range(body.v.size(0))]

    if render_body and second_body is not None:
        nv = second_body.v.size(1)
        vertex_colors = np.tile(colors['grey'], (nv, 1))
        if body_alpha is not None:
            vtx_alpha = np.ones((vertex_colors.shape[0], 1)) * body_alpha
            vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
        faces = c2c(second_body.f)
        body_mesh_seq_2 = [
            trimesh.Trimesh(vertices=c2c(second_body.v[i]), faces=faces, vertex_colors=vertex_colors, process=False) for i in
            range(second_body.v.size(0))]

    if render_joints and joints_seq is None:
        # only body joints
        joints_seq = [c2c(body.Jtr[i, :22]) for i in range(body.Jtr.size(0))]
    elif render_joints and torch.is_tensor(joints_seq[0]):
        joints_seq = [c2c(joint_frame) for joint_frame in joints_seq]

    if render_joints and extra_joint_seq is not None:
        extra_joint_seq = [c2c(joint_frame) for joint_frame in extra_joint_seq]

    if joints_vel is not None and torch.is_tensor(joints_vel[0]):
        joints_vel = [c2c(joint_frame) for joint_frame in joints_vel]
    if points_vel is not None and torch.is_tensor(points_vel[0]):
        points_vel = [c2c(joint_frame) for joint_frame in points_vel]

    if projectiles_seq is not None:
        projectiles_seq = [c2c(balls_frame) for balls_frame in projectiles_seq]
    if other_viz_locs_seq is not None:
        other_viz_locs_seq = [c2c(balls_frame) for balls_frame in other_viz_locs_seq]

    mv = MeshViewer(width=imw, height=imh,
                    use_offscreen=use_offscreen, 
                    follow_camera=follow_camera and (second_body is None),
                    camera_intrinsics=camera_intrinsics,
                    img_extn=img_extn,
                    default_cam_offset=cam_offset,
                    default_cam_rot=cam_rot)
    if render_body and render_bodies_static is None:
        mv.add_mesh_seq(body_mesh_seq)
        if second_body is not None:
            mv.add_mesh_seq(body_mesh_seq_2)
    elif render_body and render_bodies_static is not None:
        mv.add_static_meshes([body_mesh_seq[i] for i in range(len(body_mesh_seq)) if i % render_bodies_static == 0])
    if render_joints and render_skeleton:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts,
                         connections=skel_connections, connect_color=skel_color, vel=joints_vel,
                         contact_color=contact_color, render_static=render_points_static)
        if extra_joint_seq is not None:
            mv.add_point_seq(extra_joint_seq, color=extra_joint_color, radius=joint_rad, contact_seq=None,
                             connections=skel_connections, connect_color=skel_color, vel=None,
                             contact_color=contact_color, render_static=render_points_static)
    elif render_joints:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts, vel=joints_vel, contact_color=contact_color,
                            render_static=render_points_static)

    if projectiles_seq is not None:
        # TODO: fixed radius for now
        print("projectiles", projectiles_seq[0].shape)
        mv.add_spheres_seq(projectiles_seq, color=extra_joint_color, render_static=render_points_static)

    if other_viz_locs_seq is not None:
        mv.add_point_seq(other_viz_locs_seq, color=[0.6, 0.0, 0.0], radius=0.05, contact_seq=None,
                         connections=None, connect_color=None, vel=None,
                         contact_color=contact_color, render_static=render_points_static)


    if force_seq is not None:
        force_seq_np = [c2c(f) for f in force_seq]
        # force_seq_np list of (K,2,3) numpy arrays #forces, (start, end)
        mv.add_forces_seq(force_seq_np, color=force_color, radius=joint_rad, render_static=render_points_static)

    if vtx_list is not None:
        mv.add_smpl_vtx_list_seq(body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015)

    if points_seq is not None:
        if torch.is_tensor(points_seq[0]):
            points_seq = [c2c(point_frame) for point_frame in points_seq]
        mv.add_point_seq(points_seq, color=point_color, radius=point_rad, vel=points_vel, render_static=render_points_static)

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        # xyz_orig = None
        # if ground_plane_seq is not None:
        #     if render_body:
        #         xyz_orig = body_mesh_seq[0].vertices[0, :]
        #     elif render_joints:
        #         xyz_orig = joints_seq[0][0, :]
        #     elif points_seq is not None:
        #         xyz_orig = points_seq[0][0, :]
        mv.add_ground(
            ground_plane_seq=ground_plane_seq, xyz_orig=None,
            color0=ground_color0, color1=ground_color1, alpha=ground_alpha,
            render_static=render_points_static
        )

    mv.set_render_settings(out_path=out_path, wireframe=wireframe, RGBA=RGBA,
                            single_frame=(render_points_static is not None or render_bodies_static is not None)) # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps)
    except RuntimeError as err:
        print('Could not render properly with the error: %s' % (str(err)))

    del mv

def create_video(img_path, out_path, fps):
    '''
    Creates a video from the frame format in the given directory and saves to out_path.
    '''
    command = ['ffmpeg', '-y', '-r', str(fps), '-i', img_path, \
                    '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', out_path]
    subprocess.run(command)