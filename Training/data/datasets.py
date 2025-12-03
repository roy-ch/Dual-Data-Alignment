import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from PIL import Image
from PIL import ImageFile
import os
import torch
import json
from tqdm import tqdm

from .custom_transforms import (
    PadRandomCrop,
    RandomJPEGCompression,
    RandomGaussianNoise,
    RandomPepperNoise,
    MedianBlur,
    MotionBlur,
    RandomSharpen,
    ComposedTransforms,
    JPEG_Compression,
    pixel_blend_mix,
    freq_blend_mix,
    apply_resize,
    get_list
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}

STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def create_train_transforms(
    size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    is_crop=True,
):
    if is_crop:
        resize_func = PadRandomCrop(size)
    else:
        print(f"Using Resize to {size} x {size}")
        resize_func = transforms.Resize((size, size))

    transform_list = [
        RandomJPEGCompression(quality_lower=55, quality_upper=100, p=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomGaussianNoise(p=0.1),
        RandomPepperNoise(p=0.1),
        transforms.RandomApply(
            [
                transforms.RandomChoice(
                    [
                        transforms.GaussianBlur(kernel_size=3),
                        transforms.GaussianBlur(kernel_size=5),
                        MedianBlur(kernel_size=3),
                        MotionBlur(kernel_size=5),
                    ]
                )
            ],
            p=0.2,
        ),
        RandomSharpen(p=0.1),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.175
                )
            ],
            p=0.2,
        ),
        transforms.RandomGrayscale(p=0.1),
        resize_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transform_list)


class RealFakeDataset(Dataset):
    def __init__(self, opt):

        self.opt = opt

        self.data_list = []

        self.dataset_sources = {}

        with open(self.opt.quality_json, "rb") as file:
            self.real_quality_factor_mapping = json.load(file)

        primary_samples = self._load_dataset(
            real_image_dir=opt.real_image_dir,
            vae_models=[opt.vae_image_dir],
            source_name="primary",
        )
        self.data_list.extend(primary_samples)

        for source, count in self.dataset_sources.items():
            print(f"Loaded {count} samples from {source} dataset")

        random.shuffle(self.data_list)

        stat_from = "clip"
        print(f"Mean and std stats are from: {stat_from}")

        transform_list_png = create_train_transforms(
            size=opt.cropSize,
            mean=MEAN[stat_from],
            std=STD[stat_from],
            is_crop=True,
        )
        self.transform_png = ComposedTransforms(transform_list_png)

        transform_list_jpeg = create_train_transforms(
            size=opt.cropSize,
            mean=MEAN[stat_from],
            std=STD[stat_from],
            is_crop=True,
        )
        self.transform_jpeg = ComposedTransforms(transform_list_jpeg)

        self.p_freqmix = self.opt.p_freqmix

    def _load_dataset(self, real_image_dir, vae_models, source_name):

        samples = []

        self.dataset_sources[source_name] = 0

        real_list = get_list(real_image_dir)
        real_list.sort()

        for real_path in tqdm(real_list, desc=f"Loading {source_name} dataset..."):

            sample = {
                "real_path": real_path,
                "fake_paths": [],
                "source": source_name,
                "format": (
                    "jpeg"
                    if real_path.lower().endswith(".jpg")
                    or real_path.lower().endswith(".jpeg")
                    else "png"
                ),
                "jpeg_quality": self.real_quality_factor_mapping.get(
                    os.path.basename(real_path), 96
                ),
            }

            basename_real_path = os.path.basename(real_path)
            basename_real_path_without_suffix = os.path.splitext(basename_real_path)[0]
            basename_fake_path = basename_real_path_without_suffix + ".png"

            missing_path = False
            missing_vae_models = []

            for vae_rec_dir in vae_models:
                vae_rec_path = os.path.join(vae_rec_dir, basename_fake_path)
                if not os.path.exists(vae_rec_path):
                    missing_path = True
                    missing_vae_models.append(vae_rec_dir)

                else:
                    sample["fake_paths"].append(vae_rec_path)

            if not missing_path:
                samples.append(sample)
                self.dataset_sources[source_name] += 1
            else:
                continue

        return samples

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, idx):

        resampling_methods = [
            Image.NEAREST,
            Image.BOX,
            Image.BILINEAR,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
        ]

        img_dict = {
            "real": None,
            "fake": None,
            "real_resized": [],
            "fake_resized": [],
        }

        sample = self.data_list[idx]

        actual_format = sample["format"]

        real_img_path = sample["real_path"]

        real_img = Image.open(sample["real_path"]).convert("RGB")
        img_dict["real"] = real_img

        if len(sample["fake_paths"]) > 0:
            fake_path = random.choice(sample["fake_paths"])
            fake_img = Image.open(fake_path).convert("RGB")

            if random.random() < 0.5:
                jpeg_quality_factor = int(sample["jpeg_quality"])
                fake_img = JPEG_Compression(fake_img, jpeg_quality_factor)

            blending_operations = []

            if self.opt.p_pixelmix > 0 and random.random() < self.opt.p_pixelmix:
                blending_operations.append("pixel")

            if self.p_freqmix > 0 and random.random() < self.p_freqmix:
                blending_operations.append("frequency")

            if blending_operations:
                random.shuffle(blending_operations)

                for operation in blending_operations:
                    if operation == "pixel":

                        resize_method = random.choice(resampling_methods)
                        fake_img_resized = fake_img.resize(real_img.size, resize_method)
                        fake_img = pixel_blend_mix(
                            real_img=real_img,
                            fake_img=fake_img_resized,
                            ratios=[0, self.opt.r_pixelmix],
                        )

                    elif operation == "frequency":
                        fake_img = freq_blend_mix(
                            real_img=real_img,
                            fake_img=fake_img,
                            ratios=[0, self.opt.r_freqmix],
                            patch=-1,
                        )

            img_dict["fake"] = fake_img

        else:
            raise ValueError("No fake images could be loaded")

        down_resize_factor = random.uniform(0.2, 1.0)
        upper_resize_factor = random.uniform(1.0, 3.5)

        resize_methods = random.sample(resampling_methods, 2)

        for resize_factor, resize_method in zip(
            [down_resize_factor, upper_resize_factor], resize_methods
        ):

            real_resized = apply_resize(real_img, resize_factor, resize_method)
            fake_resized = apply_resize(fake_img, resize_factor, resize_method)

            img_dict["real_resized"].append(real_resized)
            img_dict["fake_resized"].append(fake_resized)

        transformed_dict = (
            self.transform_jpeg(img_dict)
            if actual_format == "jpeg"
            else self.transform_png(img_dict)
        )

        transformed_dict["source"] = sample["source"]

        return transformed_dict


def custom_collate_fn(batch):

    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return {
            "real": torch.tensor([]),
            "real_resized": torch.tensor([]),
            "fake": torch.tensor([]),
            "fake_resized": torch.tensor([]),
        }

    real_images = []
    fake_images = []
    real_resized_images = []
    fake_resized_images = []

    for item in batch:

        if "real" in item:
            if isinstance(item["real"], list):
                for real_img in item["real"]:
                    if real_img is not None:
                        real_images.append(real_img)
            else:
                if item["real"] is not None:
                    real_images.append(item["real"])

        if "fake" in item:
            if isinstance(item["fake"], list):
                for fake_img in item["fake"]:
                    if fake_img is not None:
                        fake_images.append(fake_img)
            else:
                if item["fake"] is not None:
                    fake_images.append(item["fake"])

        if "real_resized" in item:
            if isinstance(item["real_resized"], list):
                for real_resize_img in item["real_resized"]:
                    if real_resize_img is not None:
                        real_resized_images.append(real_resize_img)
            else:
                if item["real_resized"] is not None:
                    real_resized_images.append(item["real_resized"])

        if "fake_resized" in item:
            if isinstance(item["fake_resized"], list):
                for fake_resize_img in item["fake_resized"]:
                    if fake_resize_img is not None:
                        fake_resized_images.append(fake_resize_img)
            else:
                if item["fake_resized"] is not None:
                    fake_resized_images.append(item["fake_resized"])

    real_images_tensor = torch.stack(real_images) if real_images else torch.tensor([])
    real_resized_images_tensor = (
        torch.stack(real_resized_images) if real_resized_images else torch.tensor([])
    )
    fake_images_tensor = torch.stack(fake_images) if fake_images else torch.tensor([])
    fake_resized_images_tensor = (
        torch.stack(fake_resized_images) if fake_resized_images else torch.tensor([])
    )

    return {
        "real": real_images_tensor,
        "real_resized": real_resized_images_tensor,
        "fake": fake_images_tensor,
        "fake_resized": fake_resized_images_tensor,
    }
