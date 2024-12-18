import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path
from subprocess import run
from typing import Callable, Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from einops import rearrange, reduce
from tqdm import tqdm

from dataset.quaternion import ax_from_6v
from dataset.AMASS_dataset import MotionDataset
from dataset.preprocess import increment_path
from diffusion.adan import Adan
from diffusion.diffusion import GaussianDiffusion
from model.rotary_embedding_torch import RotaryEmbedding
from diffusion.utils import PositionalEncoding, SinusoidalPosEmb
from visualization.render_motion import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class MotionWrapper:
    def __init__(
        self,
        checkpoint_path="",
        EMA=True,
        loss_type="l2",
        learning_rate=4e-4,
        weight_decay=0.02,
        predict_contact=False,
        training_from_ckpt=False,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        pos_dim = 3
        rot_dim = 22*6 # 22 joints, 6dof
        self.predict_contact = predict_contact
        self.repr_dim = repr_dim = pos_dim + rot_dim + (4 if self.predict_contact else 0)

        horizon_seconds = 3
        FPS = 20
        self.horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        model = MotionDecoder(
            nfeats=repr_dim,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            activation=F.gelu
        )

        self.smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            schedule="cosine",
            n_timestep=1000,
        )
        
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if training_from_ckpt:
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            if training_from_ckpt:
                self.model.load_state_dict(
                    maybe_wrap(
                        checkpoint["model_state_dict"], num_processes,
                    )
                )

                self.diffusion.master_model.load_state_dict(
                    maybe_wrap(
                        checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                        num_processes,
                    )
                )
            else:
                self.model.load_state_dict(
                    maybe_wrap(
                        checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                        num_processes,
                    )
                )
            
    @torch.no_grad()
    def _get_normalized_loc(self, x, y, device='cuda'):
        X_COORD = 4 if self.predict_contact else 0
        frame = torch.zeros((1, 1, 135 + X_COORD))
        frame[0, 0, X_COORD] = x
        frame[0, 0, X_COORD + 1] = y
        return self.normalizer.normalize(frame)[0, 0, X_COORD : X_COORD + 2].to(device)

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)
    
    def p_losses(self, x_start, t):
        '''
        x_start : [batch_size x 60 x 135]
        cond : None
        t : [batch_size], randomly generated
        '''
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        x_recon = self.model(x_noisy, t)
        assert noise.shape == x_recon.shape
        model_out, target = x_recon, x_start        # NOTE: we assume MotionDecoder predicts x_start, not epsilon!!

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        
        # root motion extra loss
        loss_root = self.loss_fn(model_out[...,:3], target[...,:3], reduction="none") if not self.predict_contact else \
                    self.loss_fn(model_out[...,4:7], target[...,4:7], reduction="none")
        loss_root = reduce(loss_root, "b ... -> b (...)", "mean")

        if not self.predict_contact:
            losses = (
                0.636 * loss.mean(),
                1.000 * loss_root.mean(),
                torch.tensor(0.0,device='cuda'),
                torch.tensor(0.0,device='cuda'),
                torch.tensor(0.0,device='cuda')
            )
            return sum(losses), losses
        
        # split off contact from the rest
        model_contact, model_out = torch.split(
            model_out, (4, model_out.shape[2] - 4), dim=2
        )
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)
        
        # FK loss
        b, s, c = model_out.shape
        
        # unnormalize
        model_out, target = self.normalizer.unnormalize(model_out), self.normalizer.unnormalize(target)
        
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))  # should be b, s, 22, 6
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))  # should be b, s, 22, 6
        
        # perform FK
        model_xp = self.smpl.forward(model_q, model_x)
        target_xp = self.smpl.forward(target_q, target_x)

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        
        # foot skate loss
        foot_idx = [7, 8, 10, 11]
        
        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.loss_fn(
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        losses = (
            0.636 * loss.mean(),
            1.000 * loss_root.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),
            torch.tensor(0.0, device='cuda')
        )
        return sum(losses), losses

    def loss(self, x):
        batch_size = len(x)
        t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        return self.p_losses(x, t)
    
    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = MotionDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )

        # set normalizer
        self.normalizer = train_dataset.normalizer

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=8 if opt.remote else min(int(num_cpus * 0.75), 32),
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
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            avg_inpaintloss = 0
            
            # train
            self.train()
            for step, x in enumerate(load_loop(train_data_loader)):
                total_loss, (loss, v_loss, fk_loss, foot_loss, inpaint_loss) = self.loss(x)
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)
                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    avg_inpaintloss += inpaint_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
                        
            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    avg_inpaintloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                        "Inpaint Loss": avg_inpaintloss
                    }
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }

                    filename = f"{opt.exp_name}-train-{epoch}.pt" if opt.remote else f"train-{epoch}.pt"
                    torch.save(ckpt, os.path.join(wdir, filename))
                    if opt.remote:
                        # move it to local machine, then delete it from the remote server
                        # assuming `ssh-keygen -t` rsa was already run on the remote server

                        # since this is typically run on a random node, ssh to client just goes back to sc.stanford.edu, rather than the local folder
                        # run(f'ssh ${{SSH_CLIENT%% *}}', shell=True)
                        # run(f'mkdir -p {opt.local_save_dir}', shell=True)  # create the local output folder if it doesn't exist already
                        # run(f'exit', shell=True)
                        run(f"scp {os.path.join(wdir, filename)} ${{SSH_CLIENT%% *}}:{opt.local_save_dir}", shell=True)
                        run(f"rm {os.path.join(wdir, filename)}", shell=True)

                    # generate a sample
                    render_count = 20
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    # (x, cond, filename, wavnames) = next(iter(test_data_loader))
                    # (x, cond, filename, wavnames) = next(iter(train_data_loader))
                    cond = torch.ones(render_count)
                    wavnames = torch.ones(render_count)
                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            pass

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, colors=None
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render,
            colors=colors
        )

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        ) # nn.Mish is an activation function like ReLU or GELU

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")  # unsqueeze such that there are three dimensions instead of two
        scale_shift = pos_encoding.chunk(2, dim=-1)  # split embedding into two, along the last dimension
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, t
    def forward(
        self,
        tgt,
        t,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, t):
        for layer in self.stack:
            x = layer(x, t)
        return x


class MotionDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True
    ) -> None:

        super().__init__()
        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        
        # using nn.Sequential since previous saved models have these keys saved in their .pt file
        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.final_layer = nn.Linear(latent_dim, output_feats)

    
    def forward(
        self, x: Tensor, times: Tensor
    ):
        # project to latent space
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # sinusoidal position embedding - linear layer - non-linearity (output size is 4*latent_dim)
        t_hidden = self.time_mlp(times)
        t = self.to_time_cond(t_hidden) # linear from 4*latent_dim to latent_dim

        # Pass through the transformer decoder
        output = self.seqTransDecoder(x, t)

        output = self.final_layer(output)
        return output
