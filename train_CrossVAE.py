from comet_ml import Experiment
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CrossVAE
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from utils import argObj, get_logger
from metrics import prediction_metric
from dataloader import get_roi_indexes, FeatureDataset_with_Clip
from annealing import LinearAnnealing, CyclicAnnealing
import gc
from tqdm import tqdm

ROI_NAMES = {
    "prf-visualrois": 0,
    "floc-bodies": 1,
    "floc-faces": 2,
    "floc-places": 3,
    "floc-words": 4,
    "streams": 5,
    "unknown": 6,
}


def train_epoch(model, train_loader, optimizer, epoch, args):
    model_lh, model_rh = model
    optimizer_lh, optimizer_rh = optimizer

    model_lh.train()
    model_rh.train()
    print("Annealing Factor: ", args.annealing_factor)
    if args.log_in_comet:
        args.experiment.log_metric(
            "Annealing Factor", args.annealing_factor, epoch=epoch, step=epoch
        )
    for batch_idx, (data, target) in enumerate(train_loader):
        data, clip_data = data
        data = data.to(args.device)
        clip_data = clip_data.to(args.device)

        target_lh, target_rh = target
        target_lh = target_lh.to(args.device)
        target_rh = target_rh.to(args.device)

        optimizer_lh.zero_grad()
        optimizer_rh.zero_grad()

        loss_lh = model_lh.get_loss(
            data,
            clip_data,
            target_lh,
            annealing_factor=args.annealing_factor,
        )
        loss_rh = model_rh.get_loss(
            data,
            clip_data,
            target_rh,
            annealing_factor=args.annealing_factor,
        )

        clip_grad_norm_(model_lh.parameters(), max_norm=1.0)
        clip_grad_norm_(model_rh.parameters(), max_norm=1.0)

        loss_lh.backward()
        optimizer_lh.step()

        loss_rh.backward()
        optimizer_rh.step()

        loss = loss_lh.detach().item() + loss_rh.detach().item()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )

        args.annealing_factor = args.annealing()

    if args.log_in_comet:
        args.experiment.log_metric(
            "train_loss_lh" + args.name_roi, loss_lh.item(), epoch=epoch, step=epoch
        )
        args.experiment.log_metric(
            "train_loss_rh" + args.name_roi, loss_rh.item(), epoch=epoch, step=epoch
        )
        args.experiment.log_metric(
            "train_loss" + args.name_roi, loss, epoch=epoch, step=epoch
        )


def eval_epoch(model, val_loader, epoch, args):
    model_lh, model_rh = model

    model_lh.eval()
    model_rh.eval()
    val_loss = 0

    criterion = nn.MSELoss(reduction="sum")

    predictions_lh, predictions_rh = [], []
    targets_lh, targets_rh = [], []

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, clip_data = data
            data = data.to(args.device)
            clip_data = clip_data.to(args.device)

            lh_target, rh_target = target
            lh_target = lh_target.to(args.device)
            rh_target = rh_target.to(args.device)

            output_lh = model_lh(data, clip_data)
            output_rh = model_rh(data, clip_data)

            val_loss += criterion(output_lh, lh_target).item()
            val_loss += criterion(output_rh, rh_target).item()

            predictions_lh.append((output_lh).cpu().numpy())
            predictions_rh.append((output_rh).cpu().numpy())

            targets_lh.append((lh_target).cpu().numpy())
            targets_rh.append((rh_target).cpu().numpy())

    val_loss /= len(val_loader.dataset)
    print("\nValidation set: Average loss: {:.4f}\n".format(val_loss))

    if args.log_in_comet:
        args.experiment.log_metric("val_loss" + args.name_roi, val_loss, epoch=epoch)

    fmri_lh_val_pred = np.concatenate(predictions_lh)
    fmri_rh_val_pred = np.concatenate(predictions_rh)
    fmri_val_pred = (fmri_lh_val_pred, fmri_rh_val_pred)
    target_lh = np.concatenate(targets_lh)
    target_rh = np.concatenate(targets_rh)
    fmri_val_true = (target_lh, target_rh)

    lh_correlation, rh_correlation = prediction_metric(
        fmri_val_pred, fmri=fmri_val_true
    )

    lh_mean_correlation = np.mean(lh_correlation)
    rh_mean_correlation = np.mean(rh_correlation)
    lh_median_correlation = np.median(lh_correlation)
    rh_median_correlation = np.median(rh_correlation)

    print(
        "Validation set (mean):  lh_correlation: {:.4f} ".format(lh_mean_correlation),
        "rh_correlation: {:.4f} ".format(rh_mean_correlation),
    )
    print(
        "Validation set (median):  lh_correlation: {:.4f} ".format(
            lh_median_correlation
        ),
        "rh_correlation: {:.4f} ".format(rh_median_correlation),
    )

    if args.log_in_comet:
        args.experiment.log_metric(
            "val_corr_lh (mean)" + args.name_roi, lh_mean_correlation, epoch=epoch
        )
        args.experiment.log_metric(
            "val_corr_rh (mean)" + args.name_roi, rh_mean_correlation, epoch=epoch
        )
        args.experiment.log_metric(
            "val_corr_lh (median)" + args.name_roi, lh_median_correlation, epoch=epoch
        )
        args.experiment.log_metric(
            "val_corr_rh (median)" + args.name_roi, rh_median_correlation, epoch=epoch
        )

    return val_loss, lh_mean_correlation, rh_mean_correlation


def train_loop(args):
    file_paths = argObj(args.data_dir, args.submission_dir, args.subj)
    print("Loading data...")

    if args.annealing_how == "linear":
        args.annealing = LinearAnnealing(args.annealing_iteration)
    elif args.annealing_how == "cyclic":
        args.annealing = CyclicAnnealing(args.annealing_iteration)

    args.annealing_factor = 0.0

    if args.layer != -1:
        args.extracted_features = os.path.join(
            args.extracted_features, f"subj{file_paths.subj}"
        )
    else:
        args.extracted_features = os.path.join(
            args.extracted_features, f"subj{file_paths.subj}_last_layer"
        )

    train_resent_features = np.load(
        os.path.join(
            args.extracted_features,
            f"train_layer{args.layer}.npy",
        )
    )
    train_lh_target = np.load(
        os.path.join(args.extracted_features, f"train_lh_target.npy")
    )
    train_rh_target = np.load(
        os.path.join(args.extracted_features, f"train_rh_target.npy")
    )

    val_resnet_features = np.load(
        os.path.join(
            args.extracted_features,
            f"val_layer{args.layer}.npy",
        )
    )
    val_lh_target = np.load(os.path.join(args.extracted_features, f"val_lh_target.npy"))
    val_rh_target = np.load(os.path.join(args.extracted_features, f"val_rh_target.npy"))

    print("Creating Dataloader...")
    if args.roi_data:
        lh_roi_idxs, rh_roi_idxs = get_roi_indexes(file_paths)

        lh_roi_idxs = lh_roi_idxs[ROI_NAMES[args.name_roi]]
        rh_roi_idxs = rh_roi_idxs[ROI_NAMES[args.name_roi]]

        print(lh_roi_idxs.shape, rh_roi_idxs.shape)

        train_lh_target = train_lh_target[:, lh_roi_idxs]
        train_rh_target = train_rh_target[:, rh_roi_idxs]

        train_clip_features = np.load(
            os.path.join(
                args.extracted_features,
                f"train_clip_emb.npy",
            )
        )

        trainset = FeatureDataset_with_Clip(
            train_resent_features,
            train_clip_features,
            train_lh_target,
            train_rh_target,
        )

        val_lh_target = val_lh_target[:, lh_roi_idxs]
        val_rh_target = val_rh_target[:, rh_roi_idxs]

        val_clip_features = np.load(
            os.path.join(
                args.extracted_features,
                f"val_clip_emb.npy",
            )
        )

        valset = FeatureDataset_with_Clip(
            val_resnet_features,
            val_clip_features,
            val_lh_target,
            val_rh_target,
        )

        args.feature_dim_lh = len(lh_roi_idxs)
        args.feature_dim_rh = len(rh_roi_idxs)
        del lh_roi_idxs, rh_roi_idxs

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    del trainset, valset
    del train_resent_features, val_resnet_features
    gc.collect()

    if args.roi_data:
        args.name_roi = "_" + args.name_roi
        args.out_shape_lh = args.feature_dim_lh
        args.out_shape_rh = args.feature_dim_rh
        print("Training on ROI: ", args.name_roi)
    else:
        args.name_roi = ""
        args.out_shape_lh = args.feature_dim_lh
        args.out_shape_rh = args.feature_dim_rh
        print("Training on whole brain")

    if args.encoded_loss_path is not None:
        variational_auto_encoder_path_lh = os.path.join(
            args.encoded_loss_path,
            f"subj0{args.subj}",
            "VariationalAutoEncoder_lh" + args.name_roi,
            "model.pt",
        )
        variational_auto_encoder_path_rh = os.path.join(
            "seriously_dim_reduction_models",
            f"subj0{args.subj}",
            "VariationalAutoEncoder_rh" + args.name_roi,
            "model.pt",
        )
        model_lh = CrossVAE.CrossVAE(
            in_shape=args.in_shape,
            out_dim=args.out_shape_lh,
            encoded_loss=True,
            fmri_encoder_path=variational_auto_encoder_path_lh,
        )
        model_rh = CrossVAE.CrossVAE(
            in_shape=args.in_shape,
            out_dim=args.out_shape_rh,
            encoded_loss=True,
            fmri_encoder_path=variational_auto_encoder_path_rh,
        )
    else:
        model_lh = CrossVAE.CrossVAE(
            in_shape=args.in_shape,
            out_dim=args.out_shape_lh,
            encoded_loss=False,
        )
        model_rh = CrossVAE.CrossVAE(
            in_shape=args.in_shape,
            out_dim=args.out_shape_rh,
            encoded_loss=False,
        )

    model_lh.to(args.device)
    model_rh.to(args.device)

    optimizer_lh = torch.optim.AdamW(
        model_lh.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_rh = torch.optim.AdamW(
        model_rh.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler_lh = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_lh, T_max=(3 * args.epochs / 4), eta_min=1e-7
    )
    scheduler_rh = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rh, T_max=(3 * args.epochs / 4), eta_min=1e-7
    )

    model = (model_lh, model_rh)
    optimizer = (optimizer_lh, optimizer_rh)

    best_corr_mean_lh = -np.inf
    best_lh_epoch = 0
    best_rh_epoch = 0
    best_corr_mean_rh = -np.inf
    last_save = 0

    for epoch in range(1, args.epochs + 1):
        last_save += 1
        train_epoch(model, train_loader, optimizer, epoch, args)
        scheduler_lh.step()
        scheduler_rh.step()
        _, corr_mean_lh, corr_mean_rh = eval_epoch(model, val_loader, epoch, args)

        if corr_mean_lh >= best_corr_mean_lh:
            best_corr_mean_lh = corr_mean_lh
            print(os.path.join(args.model_dir, f"best_lh_model{args.name_roi}.pt"))
            torch.save(
                model_lh.state_dict(),
                os.path.join(args.model_dir, f"best_lh_model{args.name_roi}.pt"),
            )
            last_save = 0
            best_lh_epoch = epoch
        if corr_mean_rh >= best_corr_mean_rh:
            best_corr_mean_rh = corr_mean_rh
            torch.save(
                model_rh.state_dict(),
                os.path.join(args.model_dir, f"best_rh_model{args.name_roi}.pt"),
            )
            last_save = 0
            best_rh_epoch = epoch

        if args.log_in_comet:
            args.experiment.log_metric(
                "best_lh_corr", best_corr_mean_lh, epoch=epoch, step=epoch
            )
            args.experiment.log_metric(
                "best_rh_corr", best_corr_mean_rh, epoch=epoch, step=epoch
            )
            args.experiment.log_metric(
                "best_lh_epoch", best_lh_epoch, epoch=epoch, step=epoch
            )
            args.experiment.log_metric(
                "best_rh_epoch", best_rh_epoch, epoch=epoch, step=epoch
            )

            args.experiment.log_metric(
                "learning_rate_lh", scheduler_lh.get_lr()[0], epoch=epoch, step=epoch
            )
            args.experiment.log_metric(
                "learning_rate_rh", scheduler_rh.get_lr()[0], epoch=epoch, step=epoch
            )

        if epoch % 10 == 0:
            print("Checking for overfitting")
            _, corr_mean_lh, corr_mean_rh = eval_epoch(model, train_loader, epoch, args)

            args.experiment.log_metric(
                "train_lh_corr", corr_mean_lh, epoch=epoch, step=epoch
            )
            args.experiment.log_metric(
                "train_rh_corr", corr_mean_rh, epoch=epoch, step=epoch
            )

        if last_save > args.patience:
            break
    del model, model_lh, model_rh
    del optimizer, optimizer_lh, optimizer_rh
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_in_comet", action="store_true", help="Log in comet.ml.")
    parser.add_argument("--tag", type=str, default="default", help="Tag for comet.ml.")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--annealing_how", default="linear", help="annealing_how")
    parser.add_argument("--name_roi", type=str, default="", help="Name of ROI.")
    parser.add_argument("--layer", type=int, default=0, help="Choosing data")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval.")

    parser.add_argument(
        "--extracted_features", type=str, default="./extracted_features"
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--weight_decay", default=1e-5, type=float, help="Weight decay."
    )
    parser.add_argument(
        "--roi_data", action="store_true", default=False, help="Use ROI data."
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping."
    )

    parser.add_argument(
        "--annealing_iteration", type=float, default=1000, help="annealing_iteration"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="dataset directory path",
    )
    parser.add_argument(
        "--subj",
        type=int,
        required=True,
        default=1,
        help="subject number",
    )
    parser.add_argument(
        "--submission_dir",
        type=str,
        default="submission",
        help="submission directory path",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./seriously_training",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--encoded_loss_path",
        type=str,
        default=None,
        help="Directory to save model.",
    )

    args = parser.parse_args()

    log = get_logger()
    log.debug(args)

    args.model_dir = os.path.join(args.model_dir, f"subj0{args.subj}")

    if args.log_in_comet:
        experiment = Experiment(
            api_key="URVDBfbqnTFso6fkonsUc20tW",
            project_name="algonauts23",
            workspace="divyanshu-bhatt",
        )
        args.experiment = experiment
        args.experiment.add_tag(args.tag)
        args.experiment.log_parameters(vars(args))

        args.experiment.set_name(
            f"CrossVAE_subj_{args.subj}_Layer-{args.layer}_ROI-{args.name_roi}_{args.annealing_how}_iteration_encoded_loss"
        )
        args.model_dir = os.path.join(args.model_dir, args.experiment.get_name())
    else:
        args.model_dir = os.path.join(
            args.model_dir, f"non_comet_run_{args.annealing_how}"
        )

    os.mkdir(args.model_dir)
    args.submission_dir = os.path.join(args.model_dir, args.submission_dir)
    os.mkdir(args.submission_dir)

    if args.subj == 8:
        args.feature_dim_lh = 18981
        args.feature_dim_rh = 20530
    elif args.subj == 6:
        args.feature_dim_lh = 18978
        args.feature_dim_rh = 20220
    else:
        args.feature_dim_lh = 19004
        args.feature_dim_rh = 20544

    if args.layer == 1:
        args.in_shape = (256, 56, 56)
    elif args.layer == 2:
        args.in_shape = (512, 28, 28)
    elif args.layer == 3:
        args.in_shape = (1024, 14, 14)
    elif args.layer == 4:
        args.in_shape = (2048, 7, 7)

    train_loop(args)
