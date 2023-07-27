from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
from models import ResNetRegress
from utils import argObj
from dataloader import FeatureDataset, get_roi_indexes
from torch.utils.data import DataLoader
from metrics import prediction_metric
from models.LeNet import *

ROI_NAMES = {
    "prf-visualrois": 0,
    "floc-bodies": 1,
    "floc-faces": 2,
    "floc-places": 3,
    "floc-words": 4,
    "streams": 5,
    "unknown": 6,
}


def get_loaders(subj, roi=None, lenet=False, layer=3):
    data_path = f"./all_extracted_features/subj0{subj}"

    if not lenet:
        train_data = np.load(os.path.join(data_path, "train_clip_emb.npy"))
        val_data = np.load(os.path.join(data_path, "val_clip_emb.npy"))
    else:
        train_data = np.load(os.path.join(data_path, f"train_layer{layer}.npy"))
        val_data = np.load(os.path.join(data_path, f"val_layer{layer}.npy"))

    train_lh_target = np.load(os.path.join(data_path, "train_lh_target.npy"))
    train_rh_target = np.load(os.path.join(data_path, "train_rh_target.npy"))

    val_lh_target = np.load(os.path.join(data_path, "val_lh_target.npy"))
    val_rh_target = np.load(os.path.join(data_path, "val_rh_target.npy"))

    if roi is not None:
        file_paths = argObj("./data", "./submission", subj)
        lh_roi_indexes, rh_roi_indexes = get_roi_indexes(file_paths)
        lh_roi_indexes = lh_roi_indexes[ROI_NAMES[roi]]
        rh_roi_indexes = rh_roi_indexes[ROI_NAMES[roi]]

        train_lh_target = train_lh_target[:, lh_roi_indexes]
        train_rh_target = train_rh_target[:, rh_roi_indexes]
        val_lh_target = val_lh_target[:, lh_roi_indexes]
        val_rh_target = val_rh_target[:, rh_roi_indexes]

    out_dim_lh = train_lh_target.shape[1]
    out_dim_rh = train_rh_target.shape[1]

    print("Training Data Shape: ", train_data.shape)
    print("Validation Data Shape: ", val_data.shape)
    print("Training LH Target Shape: ", train_lh_target.shape)
    print("Training RH Target Shape: ", train_rh_target.shape)
    print("Validation LH Target Shape: ", val_lh_target.shape)
    print("Validation RH Target Shape: ", val_rh_target.shape)

    trainset = FeatureDataset(train_data, train_lh_target, train_rh_target)
    valset = FeatureDataset(val_data, val_lh_target, val_rh_target)

    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=32, pin_memory=True
    )
    valloader = DataLoader(
        valset, batch_size=128, shuffle=False, num_workers=32, pin_memory=True
    )

    return trainloader, valloader, out_dim_lh, out_dim_rh


def train_epoch(model, optimizer, train_loader, args):
    model_lh, model_rh = model
    optimizer_lh, optimizer_rh = optimizer
    train_loss = 0.0

    for data, target in tqdm(train_loader, desc="Train Loader"):
        data = data.to(args.device)
        target_lh, target_rh = target
        target_lh, target_rh = target_lh.to(args.device), target_rh.to(args.device)

        optimizer_lh.zero_grad()
        optimizer_rh.zero_grad()

        output_lh = model_lh(data)
        output_rh = model_rh(data)

        loss_lh = F.mse_loss(output_lh, target_lh)
        loss_rh = F.mse_loss(output_rh, target_rh)

        loss_lh.backward()
        loss_rh.backward()
        optimizer_lh.step()
        optimizer_rh.step()

        train_loss += loss_lh.item() + loss_rh.item()

    return train_loss / len(train_loader.dataset)


@torch.no_grad()
def val_epoch(model, val_loader, args):
    model_lh, model_rh = model
    model_lh.eval()
    model_rh.eval()
    val_loss = 0.0

    prediction_lh, prediction_rh = [], []
    val_target_lh, val_target_rh = [], []

    for data, target in tqdm(val_loader, desc="Val Loader"):
        data = data.to(args.device)
        target_lh, target_rh = target
        target_lh, target_rh = target_lh.to(args.device), target_rh.to(args.device)

        output_lh = model_lh(data)
        output_rh = model_rh(data)

        loss_lh = F.mse_loss(output_lh, target_lh)
        loss_rh = F.mse_loss(output_rh, target_rh)

        val_loss += loss_lh.item() + loss_rh.item()

        prediction_lh.append(output_lh.cpu().numpy())
        prediction_rh.append(output_rh.cpu().numpy())
        val_target_lh.append(target_lh.cpu().numpy())
        val_target_rh.append(target_rh.cpu().numpy())

    prediction_lh = np.concatenate(prediction_lh)
    prediction_rh = np.concatenate(prediction_rh)
    val_target_lh = np.concatenate(val_target_lh)
    val_target_rh = np.concatenate(val_target_rh)

    val_correlation_lh, val_correlation_rh = prediction_metric(
        (prediction_lh, prediction_rh), (val_target_lh, val_target_rh)
    )

    return (
        val_loss / len(val_loader.dataset),
        np.mean(val_correlation_lh),
        np.mean(val_correlation_rh),
    )


def train_loop(args):
    trainloader, valloader, out_dim_lh, out_dim_rh = get_loaders(
        args.subj, args.roi, args.lenet, args.layer
    )

    if args.roi is None:
        args.roi = "all"

    if args.lenet:
        if args.layer == 1:
            model_lh = LeNet_1((256, 56, 56), out_dim_lh).to(args.device)
            model_rh = LeNet_1((256, 56, 56), out_dim_rh).to(args.device)
        elif args.layer == 2:
            model_lh = LeNet_2((512, 28, 28), out_dim_lh).to(args.device)
            model_rh = LeNet_2((512, 28, 28), out_dim_rh).to(args.device)
        elif args.layer == 3:
            model_lh = LeNet_3((1024, 14, 14), out_dim_lh).to(args.device)
            model_rh = LeNet_3((1024, 14, 14), out_dim_rh).to(args.device)
        else:
            model_lh = LeNet_4((2048, 7, 7), out_dim_lh).to(args.device)
            model_rh = LeNet_4((2048, 7, 7), out_dim_rh).to(args.device)
            
    else:
        model_lh = ResNetRegress.ClipRegress(out_dim_lh).to(args.device)
        model_rh = ResNetRegress.ClipRegress(out_dim_rh).to(args.device)

    optimizer_lh = torch.optim.AdamW(
        model_lh.parameters(), lr=args.lr, weight_decay=1e-4
    )
    optimizer_rh = torch.optim.AdamW(
        model_rh.parameters(), lr=args.lr, weight_decay=1e-4
    )

    # scheduler_lh = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_lh, T_max=(3 * args.epochs / 4), eta_min=1e-8
    # )
    # scheduler_rh = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_rh, T_max=(3 * args.epochs / 4), eta_min=1e-8
    # )
    scheduler_lh = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_lh, patience=5, factor=0.5, verbose=True
    )
    scheduler_rh = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_rh, patience=5, factor=0.5, verbose=True, min_lr=1e-8
    )

    model = (model_lh, model_rh)
    optimizer = (optimizer_lh, optimizer_rh)

    best_loss = np.inf
    best_correlation_lh = 0
    best_correlation_rh = 0
    best_epoch_lh = 0
    best_epoch_rh = 0

    last_save = 0

    os.makedirs(args.model_dir + f"/subj0{args.subj}" + f"/{args.roi}", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        last_save += 1

        train_loss = train_epoch(model, optimizer, trainloader, args)
        val_loss, val_correlation_lh, val_correlation_rh = val_epoch(
            model, valloader, args
        )
        # scheduler_lh.step()
        # scheduler_rh.step()
        scheduler_lh.step(val_loss)
        scheduler_rh.step(val_loss)

        if best_correlation_lh < val_correlation_lh:
            best_correlation_lh = val_correlation_lh
            best_epoch_lh = epoch
            torch.save(
                model_lh.state_dict(),
                os.path.join(
                    args.model_dir,
                    f"subj0{args.subj}/{args.roi}",
                    f"best_lh_model_{args.roi}.pt",
                ),
            )
            last_save = 0
            print(
                os.path.join(
                    args.model_dir,
                    f"subj0{args.subj}/{args.roi}",
                    f"best_lh_model_{args.roi}.pt",
                )
            )
            print("Saved best lh model")

        if best_correlation_rh < val_correlation_rh:
            best_correlation_rh = val_correlation_rh
            best_epoch_rh = epoch
            torch.save(
                model_rh.state_dict(),
                os.path.join(
                    args.model_dir,
                    f"subj0{args.subj}/{args.roi}",
                    f"best_rh_model_{args.roi}.pt",
                ),
            )
            print(
                os.path.join(
                    args.model_dir,
                    f"subj0{args.subj}/{args.roi}",
                    f"best_rh_model_{args.roi}.pt",
                )
            )
            last_save = 0
            print("Saved best rh model")

        if val_loss < best_loss:
            best_loss = val_loss

        print(
            "Epoch: {} Train Loss: {:.4f} Val Loss: {:.4f} Val Correlation LH: {:.4f} Val Correlation RH: {:.4f}".format(
                epoch, train_loss, val_loss, val_correlation_lh, val_correlation_rh
            )
        )

        if args.log_comet:
            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)
            experiment.log_metric("val_correlation_lh", val_correlation_lh, step=epoch)
            experiment.log_metric("val_correlation_rh", val_correlation_rh, step=epoch)
            experiment.log_metric(
                "best_correlation_lh", best_correlation_lh, step=epoch
            )
            experiment.log_metric(
                "best_correlation_rh", best_correlation_rh, step=epoch
            )
            experiment.log_metric("best_epoch_lh", best_epoch_lh, step=epoch)
            experiment.log_metric("best_epoch_rh", best_epoch_rh, step=epoch)
            experiment.log_metric("best_loss", best_loss, step=epoch)
            experiment.log_metric("lr", optimizer_lh.param_groups[0]["lr"], step=epoch)

        if last_save > args.patience:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, required=True)
    parser.add_argument("--roi", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--log_comet", action="store_true", default=False)
    parser.add_argument(
        "--model_dir", type=str, default="./training_models/models_lenet"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lenet", action="store_true", default=False)
    parser.add_argument("--layer", type=int, default=3)

    args = parser.parse_args()

    if args.log_comet:
        experiment = Experiment(
            api_key="URVDBfbqnTFso6fkonsUc20tW",
            project_name="algonauts23",
            workspace="divyanshu-bhatt",
        )
        experiment = experiment
        experiment.log_parameters(vars(args))
        experiment.set_name(f"LeNet_{args.roi}")

    train_loop(args)
