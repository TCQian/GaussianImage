import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms

class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-10):
        self.patience = patience  # Number of tolerated iterations with no improvement
        self.min_delta = min_delta  # Minimum improvement threshold
        self.best_loss = None  # Stores the best loss value
        self.counter = 0  # Tracks the number of iterations without improvement

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False  # Do not stop training

        # If the improvement over the previous best loss is less than min_delta, consider it no improvement
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # Reset counter
        else:
            self.counter += 1

        # If the counter exceeds patience, stop training
        if self.counter >= self.patience:
            return True  # Stop training

        return False  # Continue training

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.patience = 100
        self.min_delta = 1e-9

        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        
        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)

        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device) 

        elif model_name == "3DGS":
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            if model_name == "GaussianImage_Cholesky" and _is_3d_checkpoint(checkpoint):
                # Convert 3D attributes to 2D: first 2 from _xyz, _cholesky[0,1,3], _features_dc, opacity=1
                state_2d = load_3d_checkpoint_as_2d(checkpoint, model_dict, self.device)
                self.gaussian_model.load_state_dict(state_2d, strict=False)
                print("Loaded 3D checkpoint as 2D (xyz[:2], cholesky[0,1,3], features_dc, opacity=1).")
            else:
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            if iter == 1 or iter % 1000 == 0:
                self.gaussian_model.debug_mode = True
            else:
                self.gaussian_model.debug_mode = False
                
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            
            if early_stopping(loss):
                print(f"Early stopping at iteration {iter}")
                break

            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value

def _is_3d_checkpoint(checkpoint):
    """True if checkpoint has 3D attributes: _xyz (N,3), _cholesky, _features_dc, and optionally opacity."""
    if "_xyz" not in checkpoint or "_cholesky" not in checkpoint or "_features_dc" not in checkpoint:
        return False
    xyz = checkpoint["_xyz"]
    return xyz.dim() >= 2 and xyz.shape[1] >= 3


def load_3d_checkpoint_as_2d(checkpoint, model_dict, device):
    """
    Convert a 3D checkpoint (_xyz [N,3], _cholesky, _features_dc, opacity) into 2D
    attributes for GaussianImage_Cholesky: first 2 from _xyz, _cholesky[0,1,3],
    _features_dc as-is, and all opacity set to 1 (not loaded from checkpoint).
    """
    state_2d = {}
    n_ckpt = checkpoint["_xyz"].shape[0]
    n_model = model_dict["_xyz"].shape[0]
    n = min(n_ckpt, n_model)

    # _xyz: first 2 from 3D; 2D model uses atanh space (get_xyz = tanh(_xyz))
    xyz_3d = checkpoint["_xyz"].to(device).float()
    if xyz_3d.shape[1] >= 3:
        xy = xyz_3d[:n, :2].clone()
        xy = torch.clamp(xy, -1.0 + 1e-6, 1.0 - 1e-6)
        _xyz_2d = torch.atanh(xy)
    else:
        _xyz_2d = xyz_3d[:n].clone()
    _xyz_param = torch.empty_like(model_dict["_xyz"], device=device)
    _xyz_param[:n].copy_(_xyz_2d)
    if n < n_model:
        _xyz_param[n:].copy_(model_dict["_xyz"][n:].to(device))
    state_2d["_xyz"] = _xyz_param

    # _cholesky: take elements at indices 0, 1, 3 (2D block from 3D lower triangular)
    cholesky_3d = checkpoint["_cholesky"].to(device).float()
    if cholesky_3d.dim() == 2 and cholesky_3d.shape[1] >= 4:
        cholesky_2d = cholesky_3d[:n, [0, 1, 3]].clone()
    else:
        cholesky_2d = cholesky_3d[:n, :3].clone() if cholesky_3d.dim() == 2 else cholesky_3d[:n].clone()
    _cholesky_param = torch.empty_like(model_dict["_cholesky"], device=device)
    _cholesky_param[:n].copy_(cholesky_2d)
    if n < n_model:
        _cholesky_param[n:].copy_(model_dict["_cholesky"][n:].to(device))
    state_2d["_cholesky"] = _cholesky_param

    # _features_dc: keep; 3D may be (N, 1, 3) -> squeeze to (N, 3)
    fd = checkpoint["_features_dc"].to(device).float()
    if fd.dim() == 3:
        fd = fd.squeeze(1)
    _features_param = torch.empty_like(model_dict["_features_dc"], device=device)
    _features_param[:n].copy_(fd[:n])
    if n < n_model:
        _features_param[n:].copy_(model_dict["_features_dc"][n:].to(device))
    state_2d["_features_dc"] = _features_param

    # opacity: keep all as 1 (do not load from checkpoint)
    state_2d["_opacity"] = torch.ones_like(model_dict["_opacity"], device=device)

    return state_2d


def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=50,
        help="Number of frames (default: %(default)s)",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "kodak":
        image_length, start = 24, 0
    elif args.data_name == "DIV2K_valid_LRX2":
        image_length, start = 100, 800
    else:
        image_length, start = args.num_frames, args.start_frame

    for i in range(start, start+image_length):
        if args.data_name == "kodak":
            image_path = Path(args.dataset) / f'kodim{i+1:02}.png'
        elif args.data_name == "DIV2K_valid_LRX2":
            image_path = Path(args.dataset) /  f'{i+1:04}x2.png'
        else:
            image_path = Path(args.dataset) / f'frame_{i+1:04}.png'

        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])
