import argparse
import inspect
import logging
import math
import os
import gc
import shutil
from datetime import timedelta
from pathlib import Path
from functools import partial
from PIL import Image
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import Upsample2D
from torch.fx import symbolic_trace

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL, VQModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from src.standardizer import ChannelStandardize
from src.pipeline import *
from src.resampler import *  # create_named_schedule_sampler
from src.data import PathologyTrain, PathologyValidation

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# Change the info of your pretrained VAE model here
VAE_PRETRAINED_PATH = "stabilityai/stable-diffusion-3.5-large"
# VAE_PRETRAINED_PATH = "CompVis/ldm-celebahq-256"
# VAE_KWARGS = {"subfolder": "vqvae"}


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--blurr",
        action="store_true",
        help="Whether to use blurr upsampling. Defaults to bilinear interp otherwise",
    )
    parser.add_argument(
        "--vae_config",
        type=str,
        default="CompVis/ldm-celebahq-256"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples for the training set, used for debugging purposes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-256",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="Whether to use color jitter augmentation. Defaults to False otherwise",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--save_images_epochs", type=int, default=10,
                        help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0,
                        help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4,
                        help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999,
                        help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample", "v-prediction"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--use_snr", action="store_true",
                        help="Whether to use SNR weighting for the loss.")
    parser.add_argument("--snr_epochs", type=int, default=25,
                        help="Number of epochs to use SNR weighting.")
    parser.add_argument("--use_lpips", action="store_true",
                        help="Whether to use LPIPS loss.")
    parser.add_argument("--use_ssim", action="store_true",
                        help="Whether to use SSIM loss.")
    parser.add_argument("--use_resampling", action="store_true",
                        help="Whether to use resampling.")
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="The gamma value for SNR weighting.")
    parser.add_argument("--noise_scheduler_type", type=str,
                        default="ddpm", help="The noise scheduler type to use.")
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=50)
    parser.add_argument("--ddpm_beta_schedule", type=str,
                        default="squaredcos_cap_v2")
    parser.add_argument("--ddpm_beta_end", type=float, default=0.02)
    parser.add_argument("--ddpm_noise_timesteps", type=int, default=350)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--acc_seed",
        type=int,
        default=42,
        help="A seed to reproduce the training. If not set, the seed will be random.",
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        default=None,
        help=(
            "The files of the training data. The files must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.dataset_name is None and args.train_data_files is None and args.train_data_dir is None:
    #     raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def save_images(input: torch.Tensor):
    # todo
    pass
    # clean_images = output['clean_images'].permute(0, 2, 3, 1)
    # clean_images = (clean_images.to(torch.float32) / 2 +
    #                 0.5).clamp(0, 1).cpu().numpy()
    # clean_images = (clean_images * 255).round().astype("uint8")


def blurr_upsample(model):
    def preblur(x):
        k = torch.tensor([1., 2., 1.], device=x.device)
        k = (k[:, None] * k[None, :]) / 16.0     # 3x3 Gaussian-ish
        w = k.expand(x.size(1), 1, 3, 3)
        return F.conv2d(x, w, padding=1, groups=x.size(1))

    def up_forward_preblur(self, hidden_states, *args, **kwargs):
        x = preblur(hidden_states)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x

    for b, _ in enumerate(model.up_blocks):
        block = model.up_blocks[b]
        upsamplers = getattr(block, "upsamplers", [])
        if not upsamplers:
            continue
        for u, _ in enumerate(upsamplers):
            up = upsamplers[u]
            up.forward = up_forward_preblur.__get__(up, Upsample2D)

def swap_conv3x3_to_1x1(unet):
    for b, _ in enumerate(unet.up_blocks):
        block = unet.up_blocks[b]
        upsamplers = getattr(block, "upsamplers", [])
        if not upsamplers:
            continue
        for u, _ in enumerate(upsamplers):
            up = upsamplers[u]
            if up.conv is not None and isinstance(up.conv, nn.Conv2d):
                c_in, c_out = up.conv.in_channels, up.conv.out_channels
                # replace 3×3 with 1×1, keep bias
                new_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=(up.conv.bias is not None))
                # sensible init
                nn.init.kaiming_normal_(new_conv.weight, nonlinearity="linear")
                if new_conv.bias is not None: nn.init.zeros_(new_conv.bias)
                up.conv = new_conv

def set_bilinear_upblock(model):
    def aa_bilinear_upsample_forward(self, hidden_states, *args, **kwargs):
        # hidden_states: (B, C, H, W)
        # upsample by 2 with antialiasing, then 3x3 conv
        x = F.interpolate(
            hidden_states, scale_factor=2, mode="bilinear",
            align_corners=False, antialias=True
        )
        if self.conv is not None:
            x = self.conv(x)
        return x

    for b, _ in enumerate(model.up_blocks):
        block = model.up_blocks[b]
        upsamplers = getattr(block, "upsamplers", [])
        if not upsamplers:
            continue
        for u, _ in enumerate(upsamplers):
            up = upsamplers[u]
            up.forward = aa_bilinear_upsample_forward.__get__(up, Upsample2D)
            print("set bilinear upsample for ", up)

def attach_hook(model):
    def hook(m, x, y): print(m.__class__, tuple(y.shape))

    for b, _ in enumerate(model.up_blocks):
        block = model.up_blocks[b]
        upsamplers = getattr(block, "upsamplers", [])
        if not upsamplers:
            continue
        for u, _ in enumerate(upsamplers):
            up = upsamplers[u]
            up.forward = up.register_forward_hook(hook)
            print("hook set for ", up)
        resnet = getattr(block, "resnets", [])
        if not resnet:
            continue
        resnet[0].register_forward_hook(hook)
        print("hook set for ", resnet[0])


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    # a big number for high resolution or big dataset
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Set the random seed manually for reproducibility.
    if args.acc_seed is not None:
        set_seed(args.acc_seed)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(
                        os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    if model.__class__ == "ChannelStandardize":
                        continue
                    try:
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    except:
                        print("skip chan_std")
                        continue

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load pretrained VAE model
    if "celeb" in args.vae_config:
        VAE_KWARGS = {"subfolder": "vqvae"}
        vae = VQModel.from_pretrained(args.vae_config, **VAE_KWARGS)
    else:
        VAE_KWARGS = {"subfolder": "vae"}
        vae = AutoencoderKL.from_pretrained(args.vae_config, **VAE_KWARGS)
    
    print(vae)

    # Freeze the VAE model
    vae.requires_grad_(False)
    vae.eval()

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    print("scale factor: ", vae_scale_factor)
    vae_shift = vae.config.shift_factor if hasattr(
        vae.config, "shift_factor") else 0.0

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = UNet2DModel(sample_size=args.resolution // vae_scale_factor,
                            in_channels=vae.config.latent_channels, out_channels=vae.config.latent_channels,
                            # down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D'),
                            # up_block_types=('UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
                            )
        print("created model from scratch")
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)
    # attach_hook(model)
    
    chan_std_ = torch.load("src/chan_std.pth")
    mean, std = chan_std_["mean"], chan_std_["std"]
    chan_std = ChannelStandardize(mean, std)
    
    in_adapter = nn.Conv2d(vae.config.latent_channels, vae.config.latent_channels, 1)
    nn.init.dirac_(in_adapter.weight); nn.init.zeros_(in_adapter.bias)

    if args.blurr:
        blurr_upsample(model)
    else:           
        swap_conv3x3_to_1x1(model)
        set_bilinear_upblock(model)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = args.prediction_type in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type and args.noise_scheduler_type == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            beta_end=args.ddpm_beta_end,
            prediction_type=args.prediction_type,
        )
        print("schedulers created with prediction type ", args.prediction_type)
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                        beta_schedule=args.ddpm_beta_schedule, beta_end=args.ddpm_beta_end)

    resampler = create_named_schedule_sampler(
        'loss-second-moment', noise_scheduler)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = PathologyTrain(
        size=256, n_samples=args.n_samples, jitter=args.jitter)
    if args.debug:
        with open(args.output_dir + '/image_paths.txt', 'w') as f:
            for i in range(len(dataset)):
                f.write(f"{dataset.data[i]}\n")

    global_step = 0
    first_epoch = 0

    logger.info(f"Dataset size: {len(dataset)}")

    # dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, pin_memory=True
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler, in_adapter = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, in_adapter
    )
    if vae.config.latent_channels == mean.numel():
        chan_std = accelerator.prepare(chan_std)

    vae = vae.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    grads = 0.0
    # todo: 
    loss_fn_vgg = lpips.LPIPS(net='vgg').requires_grad_(False).to(torch.float32).to(accelerator.device)
    m = SSIM().to(accelerator.device)
    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            torch.cuda.reset_peak_memory_stats()

            clean_images = batch["input"].to(weight_dtype)
            posterior = vae.encode(clean_images)
            if hasattr(posterior, 'latent_dist'):
                latents = posterior.latent_dist.sample()
            else:
                latents = posterior.latents
            latents = (latents - vae_shift) * vae.config.scaling_factor

            if vae.config.latent_channels == mean.numel():
                latents = chan_std(latents)
            latents = in_adapter(latents)
            
            _temp = latents.permute(1,0,2,3).contiguous().view(vae.config.latent_channels,-1)
            print("per-channel std ~", _temp.std(dim=1).mean().item())
            
            # Sample noise that we'll add to the images
            noise = torch.randn(
                latents.shape, dtype=weight_dtype, device=latents.device)
            bsz = latents.shape[0]  # batch size
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_latents, timesteps).sample

                if args.prediction_type == "epsilon" and not args.use_snr and not args.use_resampling:
                    # this could have different weights!
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                elif args.prediction_type == "epsilon" and args.use_snr:
                    if epoch < args.snr_epochs:
                        bsz = noise_pred.shape[0]
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (
                                clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # https://medium.com/@wangdk93/min-snr-diffusion-training-289197810a9e
                        snr_weights = torch.min(torch.tensor([1.]).to(snr_weights.device), torch.tensor(
                            [float(args.gamma)]).to(snr_weights.device)/(snr_weights + 1e-9))
                        snr_weights[snr_weights == 0] = 1.0
                        loss = snr_weights * \
                            F.mse_loss(noise_pred.float(),
                                       noise.float(), reduction="none")
                        # importance sampling
                        loss = loss.mean(dim=[1, 2, 3])
                        indices, weights = resampler.sample(
                            bsz, noise_pred.device)
                        resampler.update_with_all_losses(timesteps, loss)
                        loss = loss * weights
                        loss = loss.mean()
                    else:
                        # this could have different weights!
                        loss = F.mse_loss(noise_pred.float(), noise.float())
                elif args.prediction_type == "epsilon" and args.use_resampling:
                    bsz = noise_pred.shape[0]
                    loss = F.mse_loss(noise_pred.float(
                    ), noise.float(), reduction="none")  # (B, C, H, W)
                    indices, weights = resampler.sample(
                        bsz, noise_pred.device)
                    resampler.update_with_all_losses(
                        timesteps, loss.mean(dim=[1, 2, 3]))
                    loss = loss.mean(dim=[1, 2, 3]) * weights
                    loss = loss.mean()
                elif args.prediction_type == "v-prediction":
                    # v   = √α_t * ε − √(1−α_t) * x0
                    # let the scheduler convert v_pred→ε_pred/x0_pred internally
                    # v prediction training objective from https://arxiv.org/abs/2202.09778
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (
                            clean_images.shape[0], 1, 1, 1)
                    )
                    sigma_t = torch.sqrt(1 - alpha_t)
                    target = torch.sqrt(alpha_t) * noise - sigma_t * latents
                    loss = F.mse_loss(noise_pred.float(),
                                      target.float(), reduction="none")
                    loss = loss.mean(dim=[1, 2, 3])
                    if args.use_snr:
                        # todo: refactor to use snr fn from diffusers
                        snr_weights = alpha_t / (1 - alpha_t)
                        snr_weights = torch.min(torch.tensor([1.]).to(snr_weights.device), torch.tensor(
                            [float(args.gamma)]).to(snr_weights.device)/(snr_weights + 1e-9))
                        snr_weights[snr_weights == 0] = 1.0
                        loss = snr_weights * loss
                    loss = loss.mean()
                    # loss += 1e-5 * 
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (
                            clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * \
                        F.mse_loss(noise_pred.float(),
                                   clean_images.float(), reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {args.prediction_type}")

                alpha_t = _extract_into_tensor(
                    noise_scheduler.alphas_cumprod, timesteps, (
                        clean_images.shape[0], 1, 1, 1)
                )
                if args.use_lpips and epoch >= args.snr_epochs:
                    # generate x_0
                    if args.prediction_type == "epsilon":
                        pred_x0 = (noisy_latents - torch.sqrt(1 - alpha_t)
                                   * noise_pred) / torch.sqrt(alpha_t)
                    elif args.prediction_type == "sample":
                        pred_x0 = noise_pred
                    lpips_loss = loss_fn_vgg(pred_x0, latents).mean()
                    loss += 0.2 * lpips_loss
                if args.use_ssim and epoch >= args.snr_epochs:
                    if args.prediction_type == "epsilon":
                        pred_x0 = (noisy_latents - torch.sqrt(1 - alpha_t)
                                   * noise_pred) / torch.sqrt(alpha_t)
                    elif args.prediction_type == "sample":
                        pred_x0 = noise_pred
                    pred_x0 = pred_x0.to(latents.dtype)
                    loss += 0.5 * (1 - m(pred_x0, latents))

                # pred_x0 = (noisy_latents - torch.sqrt(1 - alpha_t)
                #            * noise_pred) / torch.sqrt(alpha_t)
                # loss += tv_loss(pred_x0)

                accelerator.backward(loss)
                # raw_grads = accelerator.clip_grad_norm_(
                #     model.parameters(), max_norm=float('inf')).item()

                if accelerator.sync_gradients:
                    grads = accelerator.clip_grad_norm_(
                        model.parameters(), 1.0).item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            res_memory = torch.cuda.memory_reserved() / (1024**3)
            alloc_memory = torch.cuda.memory_allocated() / (1024**3)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch,
                    "grads": grads, "lpips": lpips_loss.item() if (args.use_lpips and epoch >= args.snr_epochs) else 0.0,
                    "peak memory": peak_memory, "reserved memory": res_memory, "allocated memory": alloc_memory}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            del clean_images, batch, noisy_latents, latents, noise, noise_pred, loss
            torch.cuda.empty_cache()
            gc.collect()

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        # sample image from last batch and reconstruct
        batch = {'input': torch.stack([dataset[i]['input'] for i in range(
            args.n_samples)]).to(weight_dtype).to(accelerator.device)}
        if accelerator.is_main_process:
            if (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = UncondLatentDiffusionPipeline(
                    vae=vae,
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(
                    device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                # todo: refactor pipeline to use latents
                output = pipeline(
                    # generator=generator,
                    batch_size=1,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                    batch=batch,
                    noise_timesteps=args.ddpm_noise_timesteps,
                    return_dict=True,
                )

                if args.use_ema:
                    ema_model.restore(unet.parameters())
                images = output['images'].permute(0, 2, 3, 1)
                # denormalize the images and save to tensorboard
                images = (images.to(torch.float32) / 2 +
                          0.5).clamp(0, 1).cpu().numpy()
                images_processed = (images * 255).round().astype("uint8")
                clean_images = output['clean_images'].permute(0, 2, 3, 1)
                clean_images = (clean_images.to(torch.float32) / 2 +
                                0.5).clamp(0, 1).cpu().numpy()
                clean_images = (clean_images * 255).round().astype("uint8")

                # os.makedirs(f"{args.output_dir}/images", exist_ok=True)
                # [Image.fromarray(clean_images[i]).save(
                #     f"{args.output_dir}/images/clean_{i}.png") for i in range(len(clean_images))]
                # input = batch['input'].permute(0, 2, 3, 1)
                # input = (input.to(torch.float32) / 2 +
                #          0.5).clamp(0, 1).cpu().numpy()
                # input = (input * 255).round().astype("uint8")
                # [Image.fromarray(input[i]).save(f"{args.output_dir}/images/input_{i}.png")
                #  for i in range(len(input))]

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker(
                            "tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images(
                        "test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(
                            img) for img in images_processed],
                         "epoch": epoch},
                        step=global_step,
                    )

            if (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1) and not args.debug:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = UncondLatentDiffusionPipeline(
                    vae=vae,
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
