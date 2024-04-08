## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

## other imports
from motion_args import parse_test_opt
from model.motion_wrapper import MotionWrapper
import torch
import os
from visualization.render_motion import trajectory_animation, just_render_simple

X_START, Y_START = -0.107, -0.1545

class CustomConstraint:
    def __init__(self, traj, contact=True, device='cuda'):
        self.traj = traj # should be of shape [60, 2]
        self.root_slice = slice(4, 6) if contact else slice(0, 2)
        self.shape = (60, 139) if contact else (60, 135)
        self.traj = self.traj.to(device)
        
        assert self.traj.shape == (self.shape[0], 2)

    def traj_constraint(self, samples): # should be of shape [n, 60, 139]
        # if samples is passed in to have trajectory 0, this means that we wish to 
        if torch.equal(samples[..., self.root_slice], torch.zeros_like(samples[..., self.root_slice])):
            samples[..., self.root_slice.start] = X_START
            samples[..., self.root_slice.start+1] = Y_START
        
        if samples.dim() == 2: # of shape [60, 139]
            assert samples.shape == self.shape
            grad = torch.zeros_like(samples)
            grad[..., self.root_slice] = self.traj - samples[..., self.root_slice]
            return grad
        else:
            # otherwise, of shape [n, 60, 139]
            assert samples.shape[1:] == self.shape
            grad = torch.zeros_like(samples) 
            grad[..., self.root_slice] = self.traj.repeat(grad.shape[0], 1, 1) - samples[..., self.root_slice] 
            return grad
        
    def traj_constraint_backprop(self, samples, func):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        # sampels should be of shape [n, 60, 139]
        assert samples.dim() == 3 and samples.shape[1:] == self.shape
        traj = self.traj.repeat(samples.shape[0], 1, 1)
        with torch.enable_grad():
            traj.requires_grad_(True)
            samples.requires_grad_(True)
            next_sample = func(samples)[1]
            loss = -torch.nn.functional.mse_loss(next_sample[..., self.root_slice], traj)
            return (torch.autograd.grad(loss, samples)[0])
        
        
def main(opt):
    print('**********************')
    print('Loading model...')
    model = MotionWrapper(opt.checkpoint, predict_contact=opt.predict_contact)
    model.eval()
    print('Model loaded.')
    print('**********************\n')
    
    motion_dir = os.path.join(opt.motion_save_dir, f"{opt.model_name}")
    if not os.path.isdir(motion_dir): os.makedirs(motion_dir)
    samples_file = os.path.join(motion_dir, "normal_samples.pt")
    
    NUM = 1
    print(f'Generating {NUM} normal sample{"" if NUM == 1 else "s"}...')
    
    x_traj = torch.cat((torch.linspace(X_START, 0.3, 30), torch.linspace(0.3, 0.3, 30)))
    y_traj = torch.cat((torch.linspace(Y_START, Y_START, 30), torch.linspace(Y_START, 0.3, 30)))
    traj = torch.stack((x_traj, y_traj)).T
    const = CustomConstraint(traj=traj)
    
    shape = (NUM, model.horizon, model.repr_dim)
    cond = torch.ones(NUM).to(model.accelerator.device)  # throwaway values; artifacts of the original EDGE codebase
    samples, traj_found = model.diffusion.trust_sample(shape, cond, constraint_obj=const, debug=True)
    
    print(f'Finished generating trust samples.')
    if opt.save_motions: 
        torch.save(samples, samples_file)
        print(f'Saved in {motion_dir}')
    
    if not opt.no_render:
        print(f'Rendering {NUM} samples...')
        render_dir = os.path.join(opt.render_dir, f"{opt.model_name}")
        if not os.path.isdir(render_dir): os.makedirs(render_dir)
        just_render_simple(model.diffusion, samples[:NUM], model.normalizer, render_out=render_dir)
        
        print(f'Rendering trajectory changes...')
        ani = trajectory_animation(traj_found, traj)
        ani.save(os.path.join(render_dir, 'trajectory.mp4'), writer='ffmpeg')
        
        print('Finished rendering samples.\n')


if __name__ == "__main__":
    opt = parse_test_opt()
    opt.save_motions = True
    opt.no_render = False
    opt.predict_contact = True
    opt.data_dir   = f'./data/AMASS_{"plus_contact_new" if opt.predict_contact else "agreggated_sliced"}'
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"
    main(opt)