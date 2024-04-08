import argparse

def parse_train_demo_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="demo_train", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/AMASS_plus_contact_new", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="demo", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--remote", type=bool, default=False, help="training on remote server (default=True)"
    )
    parser.add_argument(
        "--local_save_dir", 
        type=str, 
        default="~/Documents/GitHub/TML_diffusion_AMASS/runs/train/remote_runs/", 
        help="model weights save directory on local machine (only used for remote training)"
    )
    parser.add_argument(
        "--predict_contact", type=bool, default=True, help="predict contact labels (default=true)"
    )
    parser.add_argument(
        "--use_masks", type=bool, default=False, help="use masks in training and inference (default=False)" 
    )
    opt = parser.parse_args()
    return opt

def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/AMASS_agreggated_sliced", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="MotionTrustSampling", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--remote", type=bool, default=True, help="training on remote server (default=True)"
    )
    parser.add_argument(
        "--local_save_dir", 
        type=str, 
        default="~/Documents/GitHub/TML_diffusion_AMASS/runs/train/remote_runs/", 
        help="model weights save directory on local machine (only used for remote training)"
    )
    parser.add_argument(
        "--predict_contact", type=bool, default=False, help="predict contact labels (default=False)"
    )
    parser.add_argument(
        "--use_masks", type=bool, default=False, help="use masks in training and inference (default=False)" 
    )
    opt = parser.parse_args()
    return opt

def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="model name to be used to save testing data"
    )
    parser.add_argument(
        "--predict_contact", type=bool, default=False, help="predict contact labels (default=False)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/AMASS_agreggated_sliced",
        help="directory in which training data can be found"
    )
    opt = parser.parse_args()
    return opt
