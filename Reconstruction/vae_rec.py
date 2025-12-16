import os
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from diffusers import AutoencoderKL


class VAERebilder:
    def __init__(self, vae_model_path, use_fp16=True, device=None):
        self.vae_model_path = vae_model_path
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16 = use_fp16 and "cuda" in self.device
        self.vae = None
        self.mse_threshold = 2e-2

    def _load_vae(self):
        if self.vae is None:
            print(f"Loading VAE from '{self.vae_model_path}'")
            start_time = time.time()

            model_dtype = torch.float16 if self.use_fp16 else torch.float32

            try:
                self.vae = AutoencoderKL.from_pretrained(
                    self.vae_model_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=False,
                )
            except OSError:
                print("Try to load VAE from complete model")
                self.vae = AutoencoderKL.from_pretrained(
                    self.vae_model_path,
                    subfolder="vae",
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                )

            self.vae.to(self.device)
            self.vae.eval()

            print(f"VAE loaded! Time Cost: {time.time() - start_time:.2f}s")

    def rebuild_image(self, input_path, output_path, real_output_path=None):
        self._load_vae()

        try:
            img = Image.open(input_path).convert("RGB")

            width, height = img.size
            new_width = (width // 8) * 8
            new_height = (height // 8) * 8

            if new_width != width or new_height != height:
                left = (width - new_height) // 2
                top = (height - new_height) // 2
                right = left + new_width
                bottom = top + new_height
                img = img.crop((left, top, right, bottom))

            x = torch.from_numpy(np.array(img)).float() / 255.0
            x = x.permute(2, 0, 1).unsqueeze(0)

            model_dtype = next(self.vae.parameters()).dtype
            x = x.to(device=self.device, dtype=model_dtype)

            x = 2.0 * x - 1.0

            with torch.no_grad():
                latents = self.vae.encode(x).latent_dist.sample()

                scaling_factor = (
                    self.vae.config.scaling_factor
                    if hasattr(self.vae.config, "scaling_factor")
                    else 0.18215
                )
                latents = latents * scaling_factor

                decoded = self.vae.decode(latents / scaling_factor).sample

                decoded = (decoded + 1.0) / 2.0

            decoded = (
                decoded.to(dtype=torch.float32)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            decoded = (decoded * 255).clip(0, 255).astype(np.uint8)

            rebuilt_image = Image.fromarray(decoded)

            if real_output_path:
                os.makedirs(os.path.dirname(real_output_path), exist_ok=True)
                img.save(real_output_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            rebuilt_image.save(output_path)

            return True

        except Exception as e:
            print(f"Error while processing img {input_path}: {e}")
            return False


def read_image_list(list_file):
    if not os.path.exists(list_file):
        print(f"{list_file} does not exist!")
        return []
    with open(list_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def process_images(
    image_paths, output_folder, vae_path, use_fp16=True, num_threads=4, device=None
):
    if not image_paths:
        print("Can't find the input folder")
        return

    output_folder = os.path.abspath(output_folder)
    real_folder = os.path.join(output_folder, "real")
    vae_folder = os.path.join(output_folder, "sd2.0")

    os.makedirs(real_folder, exist_ok=True)
    os.makedirs(vae_folder, exist_ok=True)

    rebuilder = VAERebilder(vae_model_path=vae_path, use_fp16=use_fp16, device=device)

    tasks = []
    for path in image_paths:
        basename = os.path.basename(path)
        filename_without_ext = os.path.splitext(basename)[0]

        real_output = os.path.join(real_folder, f"{filename_without_ext}.png")
        vae_output = os.path.join(vae_folder, f"{filename_without_ext}.png")

        if os.path.exists(real_output) and os.path.exists(vae_output):
            continue

        tasks.append((path, real_output, vae_output))

    print(f"All {len(image_paths)} images, {len(tasks)} images will be rebuilt")

    def worker(args):
        in_p, real_p, vae_p = args
        return rebuilder.rebuild_image(in_p, vae_p, real_p)

    start_time = time.time()
    success_count = 0

    if tasks:
        with ThreadPoolExecutor(max_workers=min(num_threads, len(tasks))) as executor:
            for res in tqdm(
                executor.map(worker, tasks), total=len(tasks), desc="Processing"
            ):
                if res:
                    success_count += 1

    print(f"\nRebuild Success: {success_count}/len(tasks)")
    print(f"Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    LIST_FILE = "path/to/yout/image_list"

    OUTPUT_DIR = "path/to/output/dir"

    VAE_MODEL_PATH = "stabilityai/stable-diffusion-2-1"

    GPU_ID = 0
    NUM_THREADS = 4
    USE_FP16 = True

    if not os.path.exists(LIST_FILE):
        print("Please set the correct LIST_FILE file.")
    else:
        device_str = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"

        img_paths = read_image_list(LIST_FILE)

        process_images(
            image_paths=img_paths,
            output_folder=OUTPUT_DIR,
            vae_path=VAE_MODEL_PATH,
            use_fp16=USE_FP16,
            num_threads=NUM_THREADS,
            device=device_str,
        )
