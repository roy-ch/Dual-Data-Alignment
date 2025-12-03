import time
import random
import numpy as np
import torch
from data import create_dataloader
from networks.trainer import Trainer
from options import TrainOptions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    opt = TrainOptions().parse()

    model = Trainer(opt)
    data_loader = create_dataloader(opt)

    start_time = time.time()
    print(f"Length of data loader: {len(data_loader)}")

    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % 5000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Step: {model.total_steps} | Loss: {model.loss:.4f} | Avg Time/Step: {elapsed/model.total_steps:.4f}s"
                )
                model.save_networks(f"model_iters_{model.total_steps}.pth")

        model.finalize_epoch()
        print(f"Saving model at end of epoch {epoch}")
        model.save_networks(f"model_epoch_{epoch}.pth")
