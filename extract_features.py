import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from models import ResNetRegress
from dataloader import dataloader_generator
from utils import argObj
from tqdm import tqdm
import clip

features = {}

def get_features(name):
    """
    Hook for extracting Resnet Features
    """
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


def extract_resnet_features(model, loaders, file_paths, args):
    """
    Extract Layer 1, 2, 3, 4 features from Resnet
    """
    model.eval()
    train_loader, val_loader, test_loader = loaders

    train_roi_features = {}
    for i in range(1, 5):
        train_roi_features[f"layer{i}"] = []
    train_roi_features[f"lh_target"] = []
    train_roi_features[f"rh_target"] = []

    with torch.no_grad():
        # for loader in [train_loader, val_loader]:
        for data, target in tqdm(train_loader, desc="Train Loader"):
            data = data.to(args.device)
            lh_target, rh_target = target
            _ = model(data)

            for i in range(1, 5):
                train_roi_features[f"layer{i}"].append(
                    features[f"layer{i}"].cpu().numpy()
                )

            train_roi_features["lh_target"].append(lh_target.cpu().numpy())
            train_roi_features["rh_target"].append(rh_target.cpu().numpy())

    save_path = args.save_path
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = save_path + f"/subj{file_paths.subj}"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i in range(1, 5):
        train_roi_features[f"layer{i}"] = np.concatenate(
            train_roi_features[f"layer{i}"]
        )
        print("layer{} shape: {}".format(i, train_roi_features[f"layer{i}"].shape))
        np.save(save_path + f"/train_layer{i}.npy", train_roi_features[f"layer{i}"])
        print("Saved")
        del train_roi_features[f"layer{i}"]

    train_roi_features["lh_target"] = np.concatenate(train_roi_features["lh_target"])
    print("lh_target shape: {}".format(train_roi_features["lh_target"].shape))
    np.save(save_path + "/train_lh_target.npy", train_roi_features["lh_target"])
    del train_roi_features[f"lh_target"]

    train_roi_features["rh_target"] = np.concatenate(train_roi_features["rh_target"])
    print("rh_target shape: {}".format(train_roi_features["rh_target"].shape))
    np.save(save_path + "/train_rh_target.npy", train_roi_features["rh_target"])
    del train_roi_features[f"rh_target"]

    del train_roi_features

    val_roi_features = {}
    for i in range(1, 5):
        val_roi_features[f"layer{i}"] = []
    val_roi_features[f"lh_target"] = []
    val_roi_features[f"rh_target"] = []

    with torch.no_grad():
        # for loader in [train_loader, val_loader]:
        for data, target in tqdm(val_loader, desc="Val Loader"):
            data = data.to(args.device)
            lh_target, rh_target = target
            _ = model(data)

            for i in range(1, 5):
                val_roi_features[f"layer{i}"].append(
                    features[f"layer{i}"].cpu().numpy()
                )

            val_roi_features["lh_target"].append(lh_target.cpu().numpy())
            val_roi_features["rh_target"].append(rh_target.cpu().numpy())

    for i in range(1, 5):
        val_roi_features[f"layer{i}"] = np.concatenate(val_roi_features[f"layer{i}"])
        print("layer{} shape: {}".format(i, val_roi_features[f"layer{i}"].shape))
        np.save(save_path + f"/val_layer{i}.npy", val_roi_features[f"layer{i}"])
        print("Saved")
        del val_roi_features[f"layer{i}"]

    val_roi_features["lh_target"] = np.concatenate(val_roi_features["lh_target"])
    print("lh_target shape: {}".format(val_roi_features["lh_target"].shape))
    np.save(save_path + "/val_lh_target.npy", val_roi_features["lh_target"])
    del val_roi_features[f"lh_target"]

    val_roi_features["rh_target"] = np.concatenate(val_roi_features["rh_target"])
    print("rh_target shape: {}".format(val_roi_features["rh_target"].shape))
    np.save(save_path + "/val_rh_target.npy", val_roi_features["rh_target"])
    del val_roi_features[f"rh_target"]

    del val_roi_features

    test_roi_features = {}
    for i in range(1, 5):
        test_roi_features[f"layer{i}"] = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Test Loader"):
            data = data.to(args.device)
            _ = model(data)

            for i in range(1, 5):
                test_roi_features[f"layer{i}"].append(
                    features[f"layer{i}"].cpu().numpy()
                )

    for i in range(1, 5):
        test_roi_features[f"layer{i}"] = np.concatenate(test_roi_features[f"layer{i}"])
        print("layer{} shape: {}".format(i, test_roi_features[f"layer{i}"].shape))
        np.save(save_path + f"/test_layer{i}.npy", test_roi_features[f"layer{i}"])

    return None


def clip_embeddings(loaders, file_paths, args):
    """
    A pre-trained ViT is used to extract image features
    """
    device = args.device
    save_path = "./all_extracted_features"
    save_path = save_path + f"/subj{file_paths.subj}"

    train_loader, val_loader, test_loader = loaders

    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    train_image_features = []
    with torch.no_grad():
        for images, target in tqdm(train_loader, desc="Train Dataset"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features.cpu().numpy()
            train_image_features.append(image_features)

    train_image_features = np.concatenate(train_image_features, axis=0)
    np.save(save_path + "/train_clip_emb.npy", train_image_features)

    val_image_features = []
    with torch.no_grad():
        for images, target in tqdm(val_loader, desc="Val Dataset"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features.cpu().numpy()
            val_image_features.append(image_features)

    val_image_features = np.concatenate(val_image_features, axis=0)
    np.save(save_path + "/val_clip_emb.npy", val_image_features)

    test_image_features = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Test Dataset"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features.cpu().numpy()
            test_image_features.append(image_features)

    test_image_features = np.concatenate(test_image_features, axis=0)
    np.save(save_path + "/test_clip_emb.npy", test_image_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default="152", help="resnet config")
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
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--save_path", type=str, default="./extracted_features")
    args = parser.parse_args()

    print("Loading")
    file_paths = argObj(args.data_dir, args.submission_dir, args.subj)
    train_loader, val_loader, test_loader = dataloader_generator(
        file_paths, 32, 0.9, 4, False
    )
    loaders = (train_loader, val_loader, test_loader)

    model = ResNetRegress.ResNetRegress(out_dim=1, config=args.model_config)
    model.to(args.device)

    model.resnet.layer1[-1].relu.register_forward_hook(get_features("layer1"))
    model.resnet.layer2[-1].relu.register_forward_hook(get_features("layer2"))
    model.resnet.layer3[-1].relu.register_forward_hook(get_features("layer3"))
    model.resnet.layer4[-1].relu.register_forward_hook(get_features("layer4"))

    extract_resnet_features(model, loaders, file_paths, args)
    clip_embeddings(loaders, file_paths, args)