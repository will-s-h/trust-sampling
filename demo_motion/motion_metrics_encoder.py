import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from p_tqdm import p_map
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.AMASS_dataset import MotionDataset
from dataset.preprocess import increment_path
from dataset.quaternion import ax_from_6v
from visualization.render_motion import SMPLSkeleton, skeleton_render


class TemporalConvAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(TemporalConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, stride=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=3),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3, stride=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=256, out_channels=input_dim, kernel_size=3, stride=3),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = x.transpose(1, 2)       # (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        x = x.transpose(1, 2)       # (batch, input_dim, seq_len) -> (batch, seq_len, input_dim)
        # print(x.shape)
        return x
    
    def encode(self, x):
        x = x.transpose(1, 2)       # (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = self.encoder(x)      # (batch, 512, seq_len//9)
        x = x.mean(-1)          # (batch, 512)
        return x


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class MotionEncoder:
    def __init__(
        self,
        checkpoint_path="",
        normalizer=None,
        learning_rate=4e-4,
        weight_decay=0.02,
        predict_contact=False,
        use_masks=False,
        training_from_ckpt=False,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        # pos_dim = 3
        # rot_dim = 24 * 6  # 24 joints, 6dof
        # self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        pos_dim = 3
        rot_dim = 22 * 6  # 22 joints, 6dof
        self.predict_contact = predict_contact
        self.repr_dim = pos_dim + rot_dim + (4 if self.predict_contact else 0)

        horizon_seconds = 3
        FPS = 20
        self.horizon = horizon_seconds * FPS

        self.horizon = self.horizon - self.horizon % 9
        assert self.horizon > FPS, "horizon must be greater than 1 sec"
        assert self.horizon % 9 == 0, "horizon must be multiple of 9 for autoencoder"

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        # model = DanceDecoder(
        #     nfeats=repr_dim,
        #     seq_len=horizon,
        #     latent_dim=512,
        #     ff_size=1024,
        #     num_layers=8,
        #     num_heads=8,
        #     dropout=0.1,
        #     cond_feature_dim=feature_dim,
        #     activation=F.gelu,
        #     use_masks=use_masks
        # )

        model = TemporalConvAutoEncoder(input_dim=self.repr_dim)

        self.smpl = SMPLSkeleton(self.accelerator.device)

        # diffusion = GaussianDiffusion(
        #     model,
        #     horizon,
        #     repr_dim,
        #     smpl,
        #     schedule="cosine",
        #     n_timestep=1000,
        #     predict_epsilon=False,
        #     loss_type="l2",
        #     use_p2=False,
        #     cond_drop_prob=0.25,
        #     guidance_weight=2,
        #     predict_contact=self.predict_contact,
        # )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        # self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if training_from_ckpt:
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["model_state_dict"],
                    num_processes,
                )
            )
        print_dict = self.__dict__.copy()
        print_dict.pop("model")
        print(print_dict)

    def eval(self):
        self.model.eval()

    def encode(self, x):
        return self.model.encode(x)

    def train(self):
        self.model.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, "train_tensor_dataset.pkl"
        )
        print("Loading dataset...")
        print("Loading from", train_tensor_dataset_path)

        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
        ):
            train_eval_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
        else:
            train_eval_dataset = MotionDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=True,
                horizon_trim=self.horizon,          # truncate to horizon if not already
            )
        
        self.normalizer = train_eval_dataset.normalizer

        ll = len(train_eval_dataset)
        print("len(train_dataset)", ll)
        # Created using indices from 0 to train_size.
        train_dataset = torch.utils.data.Subset(train_eval_dataset, range(0, int(ll * 0.9)))
        # Created using indices from train_size to train_size + test_size.
        eval_dataset = torch.utils.data.Subset(train_eval_dataset, range(int(ll * 0.9), ll))

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 8),
            pin_memory=True,
            drop_last=True,
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            print(f"Saving to {wdir}")
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            # train
            self.train()
            # for step, (x, cond, filename, wavnames) in enumerate(
            #     load_loop(train_data_loader)
            # ):
            for step, x in enumerate(
                load_loop(train_data_loader)
            ):
                
                # print(x.shape)

                # total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                #     x, cond, t_override=None
                # )
                # total_loss, (loss, v_loss, fk_loss, foot_loss, inpaint_loss) = self.diffusion(
                #     x, None, t_override=None, normalizer=self.normalizer # NOTE: cond is None!!!
                # )

                reconstructed_x = self.model(x)

                # print(reconstructed_x.shape)
                total_loss = F.mse_loss(reconstructed_x, x)

                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                if self.accelerator.is_main_process:
                    avg_loss += total_loss.detach().cpu().numpy()

            # Save model
            # if (epoch % 1) == 0:
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()

                    avg_eval_loss = 0
                    eval_reconstructed_x = x = None
                    for data in eval_data_loader:
                        x = data.to(self.accelerator.device)
                        eval_reconstructed_x = self.model(x)
                        eval_loss = F.mse_loss(eval_reconstructed_x, x)
                        avg_eval_loss += eval_loss.detach().cpu().numpy()

                    # log
                    avg_eval_loss /= len(eval_data_loader)
                    avg_loss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "Eval Loss": avg_eval_loss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }

                    filename = f"{opt.exp_name}-train-{epoch}.pt" if opt.remote else f"train-{epoch}.pt"
                    torch.save(ckpt, os.path.join(wdir, filename))

                    # generate a sample
                    render_count = 20
                    print("Generating Sample")
                    
                    cond = torch.ones(render_count)     # dummy cond
                    cond = cond.to(self.accelerator.device)

                    render_out = os.path.join(opt.render_dir, "train_" + opt.exp_name)
                    max_render_count = 20
                    if not os.path.isdir(render_out):
                        os.makedirs(render_out)
                    self.just_render(
                        model_output=eval_reconstructed_x[:max_render_count],      # NOTE: last batch of eval data
                        cond=cond, 
                        normalizer=self.normalizer, 
                        epoch=epoch,        # TODO: used?
                        render_out=render_out, 
                        colors=None
                    )
                    self.just_render(
                        model_output=x[:max_render_count],      # NOTE: last batch of eval data
                        cond=cond, 
                        normalizer=self.normalizer, 
                        epoch=epoch,
                        render_out=render_out+"_GT", 
                        colors=None
                    )

                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            wandb.run.finish()

    # only renders a given output
    def just_render(
        self, 
        model_output,
        cond, 
        normalizer, 
        epoch, 
        render_out, 
        colors=None
    ):
        '''
        model_output should be the direct, normalized output of model.
        '''
        samples = model_output
        samples = normalizer.unnormalize(samples)

        if samples.shape[2] != 135:
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )
        else:
            sample_contact = None
        # do the FK all at once
        b, s, c = samples.shape
        pos = samples[:, :, :3].to(cond.device)  # np.zeros((sample.shape[0], 3))
        q = samples[:, :, 3:].reshape(b, s, 22, 6)
        # go 6d to ax
        q = ax_from_6v(q).to(cond.device)

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        # sample_contact = (
        #     sample_contact.detach().cpu().numpy()
        #     if sample_contact is not None
        #     else None
        # )

        def inner(xx):
            num, pose = xx

            contact = sample_contact[num] if sample_contact is not None else None
            skeleton_render(
                pose,
                name=f"e{epoch}_b{num}",
                out=render_out,
                sound=False,
                contact=contact,
                colors=colors
            )
        
        p_map(inner, enumerate(poses))

        # if fk_out is not None and mode != "long":
        #     Path(fk_out).mkdir(parents=True, exist_ok=True)
        #     for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
        #         path = os.path.normpath(filename)
        #         pathparts = path.split(os.sep)
        #         pathparts[-1] = pathparts[-1].replace("npy", "wav")
        #         # path is like "data/train/features/name"
        #         pathparts[2] = "wav_sliced"
        #         audioname = os.path.join(*pathparts)
        #         outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
        #         pickle.dump(
        #             {
        #                 "smpl_poses": qq.reshape((-1, 72)).cpu().numpy(),
        #                 "smpl_trans": pos_.cpu().numpy(),
        #                 "full_pose": pose,
        #             },
        #             open(f"{fk_out}/{outname}", "wb"),
        #         )
