import argparse
import os
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument(
            "--name", type=str, default="experiment_name", help="Experiment name"
        )
        parser.add_argument(
            "--checkpoints_dir", type=str, required=True, help="Models are saved here"
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2 use -1 for CPU",
        )
        parser.add_argument(
            "--num_threads", default=8, type=int, help="# threads for loading data"
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="input batch size"
        )

        parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
        parser.add_argument(
            "--lora_alpha", type=float, default=1.0, help="LoRA scaling factor"
        )

        parser.add_argument(
            "--real_image_dir", default=None, help="Path to real images"
        )
        parser.add_argument(
            "--vae_image_dir", default=None, help="Path to VAE reconstructed images"
        )

        parser.add_argument(
            "--cropSize", type=int, default=336, help="Crop to this size"
        )

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = f"\t[default: {default}]" if v != default else ""
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        with open(os.path.join(expr_dir, "opt.txt"), "wt") as opt_file:
            opt_file.write(message + "\n")

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        if print_options:
            self.print_options(opt)

        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = [int(x) for x in str_ids if int(x) >= 0]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.isTrain = True

        parser.add_argument("--niter", type=int, default=1, help="Total epochs")
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="Initial learning rate"
        )
        parser.add_argument("--optim", type=str, default="adam", help="[sgd, adam]")
        parser.add_argument(
            "--accumulation_steps",
            type=int,
            default=1,
            help="Gradient accumulation steps",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0.0, help="L2 regularization"
        )

        parser.add_argument("--p_pixelmix", type=float, default=0.2)
        parser.add_argument("--r_pixelmix", type=float, default=0.0)

        parser.add_argument("--p_freqmix", type=float, default=0.2)
        parser.add_argument("--r_freqmix", type=float, default=0.1)

        parser.add_argument(
            "--quality_json",
            default="MSCOCO_train2017.json",
            help="JSON for quality settings",
        )
        return parser
