import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from io import BytesIO
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import pickle
import os
from scipy.fftpack import dct, idct
import torch

# --- 基础辅助函数 ---

def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir, followlinks=True):
        for file in f:
            if (file.split(".")[1].lower() in exts) and (
                must_contain in os.path.join(r, file)
            ):
                out.append(os.path.join(r, file))
    return out

def get_list(path, must_contain=""):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

def apply_resize(image, resize_factor, resize_method=None):
    if resize_method is None:
        resampling_methods = [
            Image.NEAREST,
            Image.BOX,
            Image.BILINEAR,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
        ]
        resize_method = random.choice(resampling_methods)

    w, h = image.size
    new_w, new_h = int(w * resize_factor), int(h * resize_factor)
    resized = image.resize((new_w, new_h), resize_method)
    return resized

def JPEG_Compression(img, quality_factor):
    if quality_factor == 100:
        return img
    out = BytesIO()
    img.save(out, format="jpeg", quality=quality_factor)
    out.seek(0)
    img = Image.open(out)
    return img

# --- Transform 类 ---

class PadRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = F.pad(img, padding, fill=0)

        cropped = transforms.RandomCrop(self.size)(img)
        return cropped

class ComposedTransforms:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, images_dict):
        # 保存随机状态以确保对 pair 中的图片应用相同的变换
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()

        result = {}

        for key, val in images_dict.items():
            if val is None:
                result[key] = None
                continue

            if isinstance(val, list):
                transformed_imgs = []
                for i, single_img in enumerate(val):
                    # 重置随机种子
                    torch.set_rng_state(torch_state)
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)

                    transformed = self.transforms(single_img)
                    transformed_imgs.append(transformed)
                result[key] = transformed_imgs

            elif isinstance(val, Image.Image):
                torch.set_rng_state(torch_state)
                np.random.set_state(numpy_state)
                random.setstate(python_state)

                transformed = self.transforms(val)
                result[key] = transformed
            else:
                result[key] = val

        return result

class MedianBlur:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1

    def __call__(self, img):
        img_np = np.array(img)
        blurred = cv2.medianBlur(img_np, self.kernel_size)
        return Image.fromarray(blurred)

    def __repr__(self):
        return self.__class__.__name__ + f"(kernel_size={self.kernel_size})"

class MotionBlur:
    def __init__(self, kernel_size=5, angle=None):
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        self.angle = angle

    def __call__(self, img):
        img_np = np.array(img)

        angle = self.angle
        if angle is None:
            angle = np.random.uniform(0, 180)

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        angle_rad = np.deg2rad(angle)

        for i in range(self.kernel_size):
            offset = i - center
            x = int(center + np.round(offset * np.cos(angle_rad)))
            y = int(center + np.round(offset * np.sin(angle_rad)))

            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                kernel[y, x] = 1

        kernel = kernel / np.sum(kernel)

        if len(img_np.shape) == 3:
            blurred = np.zeros_like(img_np)
            for c in range(img_np.shape[2]):
                blurred[:, :, c] = cv2.filter2D(img_np[:, :, c], -1, kernel)
        else:
            blurred = cv2.filter2D(img_np, -1, kernel)

        return Image.fromarray(blurred)

class RandomSharpen:
    def __init__(self, p=0.1, factor=(1.0, 3.0)):
        self.p = p
        self.factor = factor
        if isinstance(factor, (list, tuple)):
            assert len(factor) == 2, "factor should be a tuple of (min, max)"
            self.min_factor, self.max_factor = factor
        else:
            self.min_factor = self.max_factor = factor

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_np = np.array(img).astype(np.float32)
        blurred = cv2.GaussianBlur(img_np, (0, 0), 3.0)
        factor = random.uniform(self.min_factor, self.max_factor)
        sharpened = img_np + factor * (img_np - blurred)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return Image.fromarray(sharpened)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p}, factor=({self.min_factor}, {self.max_factor}))"
        )

class RandomPepperNoise:
    def __init__(self, p=0.1, noise_ratio=(0.01, 0.10)):
        self.p = p
        self.noise_ratio = noise_ratio
        if isinstance(noise_ratio, (list, tuple)):
            assert len(noise_ratio) == 2, "noise_ratio should be a tuple of (min, max)"
            self.min_ratio, self.max_ratio = noise_ratio
        else:
            self.min_ratio = self.max_ratio = noise_ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_np = np.array(img)
        height, width = img_np.shape[:2]
        noise_ratio = random.uniform(self.min_ratio, self.max_ratio)
        n_pixels = int(height * width * noise_ratio)

        y_coords = np.random.randint(0, height, n_pixels)
        x_coords = np.random.randint(0, width, n_pixels)

        if len(img_np.shape) == 3:
            img_np[y_coords, x_coords, :] = 0
        else:
            img_np[y_coords, x_coords] = 0

        return Image.fromarray(img_np)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p}, noise_ratio=({self.min_ratio}, {self.max_ratio}))"
        )

class RandomGaussianNoise:
    def __init__(self, mean=0, std=55, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_array = np.array(img).astype(np.float32)
            noise = np.random.normal(self.mean, self.std, img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
        return img

class RandomJPEGCompression:
    def __init__(self, quality_lower=30, quality_upper=95, p=0.3):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_lower, self.quality_upper)
            img = JPEG_Compression(img, quality_factor=quality)
            return img
        return img

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(quality_lower={self.quality_lower}, quality_upper={self.quality_upper}, p={self.p})"
        )

# --- 混合函数 (Mix Functions) ---

def pixel_blend_mix(
    real_img, fake_img, ratios=[0.0, 1.0], resize_method="LANCZOS", random_seed=None
):
    RESIZE_METHODS = {
        "LANCZOS": Image.LANCZOS,
        "BILINEAR": Image.BILINEAR,
        "BICUBIC": Image.BICUBIC,
        "NEAREST": Image.NEAREST,
        "BOX": Image.BOX,
        "HAMMING": Image.HAMMING,
    }

    if not (0.0 <= ratios[0] <= 1.0 and 0.0 <= ratios[1] <= 1.0):
        raise ValueError("Ratios must be between 0.0 and 1.0")
    if ratios[0] > ratios[1]:
        raise ValueError("ratios[0] must be <= ratios[1]")

    resize_method_upper = resize_method.upper()
    if resize_method_upper not in RESIZE_METHODS:
        supported = ", ".join(RESIZE_METHODS.keys())
        raise ValueError(f"resize_method must be one of: {supported}")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if isinstance(real_img, str):
        real_img = Image.open(real_img)
    if isinstance(fake_img, str):
        fake_img = Image.open(fake_img)

    if real_img.mode != "RGB":
        real_img = real_img.convert("RGB")
    if fake_img.mode != "RGB":
        fake_img = fake_img.convert("RGB")

    if real_img.size != fake_img.size:
        resize_filter = RESIZE_METHODS[resize_method_upper]
        fake_img = fake_img.resize(real_img.size, resize_filter)

    real_color_np = np.array(real_img)
    fake_color_np = np.array(fake_img)

    real_arr = real_color_np.astype(np.float32)
    fake_arr = fake_color_np.astype(np.float32)

    blend_factor = random.uniform(ratios[0], ratios[1])
    blended = blend_factor * fake_arr + (1 - blend_factor) * real_arr

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    mixed_img = Image.fromarray(blended)

    return mixed_img

def freq_blend_mix(real_img, fake_img, ratios=[0.0, 1.0], patch=8, random_seed=None):
    if not (0.0 <= ratios[0] <= 1.0 and 0.0 <= ratios[1] <= 1.0):
        raise ValueError("Ratios must be between 0.0 and 1.0")
    if ratios[0] > ratios[1]:
        raise ValueError("ratios[0] must be <= ratios[1]")

    if patch != -1 and patch <= 0:
        raise ValueError("patch must be positive or -1")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if isinstance(real_img, str):
        real_img = Image.open(real_img)
    if isinstance(fake_img, str):
        fake_img = Image.open(fake_img)

    if real_img.size != fake_img.size:
        fake_img = fake_img.resize(real_img.size, Image.LANCZOS)

    if real_img.mode != "RGB":
        real_img = real_img.convert("RGB")
    if fake_img.mode != "RGB":
        fake_img = fake_img.convert("RGB")

    real_color_np = np.array(real_img)
    fake_color_np = np.array(fake_img)

    height, width, channels = real_color_np.shape

    if patch != -1:
        if patch > min(height, width):
            patch = -1

    fixed_blend_ratio = random.uniform(ratios[0], ratios[1])

    def apply_2d_dct(patch):
        return dct(dct(patch.T, norm="ortho").T, norm="ortho")

    def apply_2d_idct(patch):
        return idct(idct(patch.T, norm="ortho").T, norm="ortho")

    def process_patches_improved(real_channel, fake_channel, patch):
        height, width = real_channel.shape
        mixed_channel = np.zeros_like(real_channel, dtype=np.float32)
        step_size = patch // 2 if patch >= 16 else patch

        for i in range(0, height, step_size):
            for j in range(0, width, step_size):
                end_i = min(i + patch, height)
                end_j = min(j + patch, width)

                real_patch = real_channel[i:end_i, j:end_j].astype(np.float32)
                fake_patch = fake_channel[i:end_i, j:end_j].astype(np.float32)

                if real_patch.shape != (patch, patch):
                    pad_h = patch - real_patch.shape[0]
                    pad_w = patch - real_patch.shape[1]
                    real_patch = np.pad(real_patch, ((0, pad_h), (0, pad_w)), mode="edge")
                    fake_patch = np.pad(fake_patch, ((0, pad_h), (0, pad_w)), mode="edge")

                real_dct = apply_2d_dct(real_patch)
                fake_dct = apply_2d_dct(fake_patch)
                mixed_dct = fixed_blend_ratio * real_dct + (1 - fixed_blend_ratio) * fake_dct
                mixed_patch = apply_2d_idct(mixed_dct)

                orig_h = min(patch, end_i - i)
                orig_w = min(patch, end_j - j)
                mixed_patch = mixed_patch[:orig_h, :orig_w]

                if step_size < patch:
                    weight = np.ones((orig_h, orig_w))
                    mixed_channel[i:end_i, j:end_j] += mixed_patch * weight
                else:
                    mixed_channel[i:end_i, j:end_j] = mixed_patch

        if step_size < patch:
            mixed_channel = mixed_channel / 2

        return np.round(np.clip(mixed_channel, 0, 255)).astype(np.uint8)

    mixed_channels = []
    for channel_idx in range(3):
        real_channel = real_color_np[:, :, channel_idx]
        fake_channel = fake_color_np[:, :, channel_idx]

        if patch == -1:
            real_channel_f = real_channel.astype(np.float32)
            fake_channel_f = fake_channel.astype(np.float32)
            real_dct = apply_2d_dct(real_channel_f)
            fake_dct = apply_2d_dct(fake_channel_f)
            mixed_dct = fixed_blend_ratio * real_dct + (1 - fixed_blend_ratio) * fake_dct
            mixed_channel = apply_2d_idct(mixed_dct)
            mixed_channel = np.round(np.clip(mixed_channel, 0, 255)).astype(np.uint8)
        else:
            mixed_channel = process_patches_improved(real_channel, fake_channel, patch)
        mixed_channels.append(mixed_channel)

    mixed_np = np.stack(mixed_channels, axis=2)
    return Image.fromarray(mixed_np)