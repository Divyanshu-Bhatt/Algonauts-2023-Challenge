import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataloader import get_roi_indexes
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm import tqdm
from utils import argObj
from annealing import CyclicAnnealing
import argparse
from models.VAE_encoded_loss import VariationalAutoEncoder

ROI_NAMES = {
    "prf-visualrois": 0,
    "floc-bodies": 1,
    "floc-faces": 2,
    "floc-places": 3,
    "floc-words": 4,
    "streams": 5,
    "unknown": 6,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--fmri", type=str, default="lh")
    parser.add_argument("--roi", type=str, required=True)
    args = parser.parse_args()

    subj = args.subj
    epochs = 200
    patience = 25
    annealing_iterations = 500
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    file_paths = argObj("./data", "./submission", subj=subj)

    if args.fmri == "lh":
        fmri = np.load(
            os.path.join(
                "all_extracted_features", f"subj0{subj}", "train_lh_target.npy"
            )
        )
        val_fmri = np.load(
            os.path.join("all_extracted_features", f"subj0{subj}", "val_lh_target.npy")
        )
        print("LH FMRI: {}".format(fmri.shape))
        indexes, _ = get_roi_indexes(file_paths)
    elif args.fmri == "rh":
        fmri = np.load(
            os.path.join(
                "all_extracted_features", f"subj0{subj}", "train_rh_target.npy"
            )
        )
        val_fmri = np.load(
            os.path.join("all_extracted_features", f"subj0{subj}", "val_rh_target.npy")
        )
        print("RH FMRI: {}".format(fmri.shape))
        _, indexes = get_roi_indexes(file_paths)

    fmri = fmri[:, indexes[ROI_NAMES[args.roi]]]
    val_fmri = val_fmri[:, indexes[ROI_NAMES[args.roi]]]
    print("ROI FMRI: {}| ROI: {}".format(fmri.shape, args.roi))

    fmri = torch.from_numpy(fmri).to(torch.float32)
    dataset = TensorDataset(fmri)
    trainloader = DataLoader(
        dataset, batch_size=1024, shuffle=True, num_workers=32, pin_memory=True
    )

    val_fmri = torch.from_numpy(val_fmri).to(torch.float32)
    val_dataset = TensorDataset(val_fmri)
    valloader = DataLoader(
        val_dataset, batch_size=1024, shuffle=True, num_workers=32, pin_memory=True
    )

    in_shape = fmri.shape[1]
    print("in_shape: {}".format(in_shape))

    model = VariationalAutoEncoder(in_shape).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=1e-7
    )

    save_path = os.path.join(
        "seriously_dim_reduction_models",
        f"subj0{subj}",
        f"VariationalAutoEncoder_{args.fmri}_{args.roi}",
    )
    print("save_path: {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)

    best_loss = np.inf
    last_save = 0

    annealing = CyclicAnnealing(annealing_iterations)
    annealing_factor = 0.0

    for epoch in range(epochs):
        last_save += 1
        total_loss = 0.0

        model.train()
        for data in trainloader:
            data = data[0].to(device)
            optimizer.zero_grad()
            mse_loss, kld_loss = model.get_loss(data, annealing_factor)
            loss = mse_loss + kld_loss
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += mse_loss.item()
            annealing_factor = annealing()

        scheduler.step()
        total_loss /= len(trainloader.dataset)

        with torch.no_grad():
            val_loss = 0.0
            for data in valloader:
                data = data[0].to(device)
                mse_loss, kld_loss = model.get_loss(data, annealing_factor)
                loss = mse_loss + kld_loss
                val_loss += mse_loss.item()

            val_loss /= len(valloader.dataset)

        print(
            "Epoch: {} | Train Loss: {:.4f} | Val Loss: {:.4f} | Best Loss {:.4f} |Learning Rate: {:.4f}x10-4 | Annealing Factor: {:.4f}".format(
                epoch,
                total_loss,
                val_loss,
                best_loss,
                scheduler.get_last_lr()[0] * 1e4,
                annealing_factor,
            )
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "model.pt"),
            )
            print("Model saved!")
            last_save = 0

        if last_save > patience:
            print("Early Stopping!")
            break

    print("Training Finished!")

    config = {
        "best_loss": best_loss,
        "epochs": epochs,
        "patience": patience,
        "annealing_iterations": annealing_iterations,
        "lr": lr,
        "annealing_factor": annealing_factor,
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f)
