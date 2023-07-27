import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torch


class FeatureDataset_with_Clip(Dataset):
    """
    Both ResNet Extracted Features and Clip Features as inputs and both LH and RH fmri as outputs
    """
    def __init__(self, input_features, clip_features, lh_fmri=None, rh_fmri=None):
        self.input_features = input_features
        self.clip_features = clip_features
        self.test_flag = False

        if lh_fmri is not None:
            self.lh_fmri = lh_fmri
            self.rh_fmri = rh_fmri
        else:
            self.test_flag = True

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        input_features = torch.from_numpy(self.input_features[idx]).to(torch.float32)
        clip_features = torch.from_numpy(self.clip_features[idx]).to(torch.float32)

        if self.test_flag:
            return input_features, clip_features

        return (input_features, clip_features), (
            torch.from_numpy(self.lh_fmri[idx]).to(torch.float32),
            torch.from_numpy(self.rh_fmri[idx]).to(torch.float32),
        )


class FeatureDataset(Dataset):
    """
    Either ResNet Extracted Features or Clip Features as inputs and both LH and RH fmri as outputs
    """
    def __init__(self, input_features, lh_fmri=None, rh_fmri=None):
        self.input_features = input_features
        self.test_flag = False

        if lh_fmri is not None:
            self.lh_fmri = lh_fmri
            self.rh_fmri = rh_fmri
        else:
            self.test_flag = True

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        input_features = torch.from_numpy(self.input_features[idx]).to(torch.float32)
        if self.test_flag:
            return input_features

        return input_features, (
            torch.from_numpy(self.lh_fmri[idx]).to(torch.float32),
            torch.from_numpy(self.rh_fmri[idx]).to(torch.float32),
        )


class ImageDataset(Dataset):
    """
    Input Features are images
    """
    def __init__(self, imgs_paths, idxs, transform, lh_fmri=None, rh_fmri=None):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.test_flag = False

        if lh_fmri is not None:
            self.lh_fmri = lh_fmri[idxs]
            self.rh_fmri = rh_fmri[idxs]
        else:
            self.test_flag = True

        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.test_flag:
            return img

        return img, (
            torch.from_numpy(self.lh_fmri[idx]),
            torch.from_numpy(self.rh_fmri[idx]),
        )


def get_roi_indexes(file_paths, unique=False):
    """
    Get ROI specific indexes from the output vector
    """
    lh_challenge_roi_files = [
        "lh.prf-visualrois_challenge_space.npy",
        "lh.floc-bodies_challenge_space.npy",
        "lh.floc-faces_challenge_space.npy",
        "lh.floc-places_challenge_space.npy",
        "lh.floc-words_challenge_space.npy",
        "lh.streams_challenge_space.npy",
    ]
    rh_challenge_roi_files = [
        "rh.prf-visualrois_challenge_space.npy",
        "rh.floc-bodies_challenge_space.npy",
        "rh.floc-faces_challenge_space.npy",
        "rh.floc-places_challenge_space.npy",
        "rh.floc-words_challenge_space.npy",
        "rh.streams_challenge_space.npy",
    ]

    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(
            np.load(
                os.path.join(
                    file_paths.data_dir, "roi_masks", lh_challenge_roi_files[r]
                )
            )
        )
        rh_challenge_rois.append(
            np.load(
                os.path.join(
                    file_paths.data_dir, "roi_masks", rh_challenge_roi_files[r]
                )
            )
        )

    lh_roi_idxs = []
    rh_roi_idxs = []

    for r in range(len(lh_challenge_rois)):
        lh_roi_idxs.append(np.where(lh_challenge_rois[r] != 0)[0])
        rh_roi_idxs.append(np.where(rh_challenge_rois[r] != 0)[0])

    lh_challenge_rois = np.array(lh_challenge_rois).sum(axis=0)
    rh_challenge_rois = np.array(rh_challenge_rois).sum(axis=0)

    lh_unknown_idxs = np.where(lh_challenge_rois == 0)[0]
    rh_unknown_idxs = np.where(rh_challenge_rois == 0)[0]

    lh_roi_idxs.append(lh_unknown_idxs)
    rh_roi_idxs.append(rh_unknown_idxs)

    if unique:
        # If ROIs have an overlap, remove the overlap. The index is given to the ROI which has smaller number of voxels.
        lengths = []
        for i in range(len(lh_roi_idxs)):
            lengths.append((len(lh_roi_idxs[i]), i))

        lengths.sort()
        _, lh_permute = zip(*lengths)
        print("LH Permute:", lh_permute)
        new_lh_roi_idxs = []
        for i in lh_permute:
            new_lh_roi_idxs.append(lh_roi_idxs[i])

        lh_roi_idxs = []
        print("For LH ROIs...", end=" ")
        for i in range(len(new_lh_roi_idxs)):
            print(f"{i}/{len(new_lh_roi_idxs)}", end=" ")
            dummy = new_lh_roi_idxs[i]
            for idx in lh_roi_idxs:
                dummy = [k for k in dummy if k not in idx]
            dummy = np.array(dummy)
            lh_roi_idxs.append(dummy)

        print()

        lengths = []
        for i in range(len(rh_roi_idxs)):
            lengths.append((len(rh_roi_idxs[i]), i))

        lengths.sort()
        _, rh_permute = zip(*lengths)
        print("RH Permute:", rh_permute)

        new_rh_roi_idxs = []
        for i in rh_permute:
            new_rh_roi_idxs.append(rh_roi_idxs[i])

        rh_roi_idxs = []
        print("For RH ROIs...", end=" ")
        for i in range(len(new_rh_roi_idxs)):
            print(f"{i}/{len(new_rh_roi_idxs)}", end=" ")
            dummy = new_rh_roi_idxs[i]
            for idx in rh_roi_idxs:
                dummy = [k for k in dummy if k not in idx]
            dummy = np.array(dummy)
            rh_roi_idxs.append(dummy)
        print()

        cache_lh_roi_indexes = lh_roi_idxs.copy()
        cache_rh_roi_indexes = rh_roi_idxs.copy()

        for i in range(len(lh_roi_idxs)):
            lh_roi_idxs[lh_permute[i]] = cache_lh_roi_indexes[i]
            rh_roi_idxs[rh_permute[i]] = cache_rh_roi_indexes[i]

    return lh_roi_idxs, rh_roi_idxs

def dataloader_generator(
    file_paths, batch_size, train_split, num_workers, roi_data=False
):
    train_img_list = os.listdir(file_paths.train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(file_paths.test_img_dir)
    test_img_list.sort()

    # Output coordinates
    lh_fmri = np.load(os.path.join(file_paths.fmri_dir, "lh_training_fmri.npy"))
    rh_fmri = np.load(os.path.join(file_paths.fmri_dir, "rh_training_fmri.npy"))

    # split
    num_train = int(np.round(len(train_img_list) * train_split))

    # shuffling and splitting
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

    idxs_test = np.arange(len(test_img_list))

    train_imgs_paths = sorted(list(Path(file_paths.train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(file_paths.test_img_dir).iterdir()))

    # transformation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not roi_data:
        train_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_train, transform, lh_fmri, rh_fmri),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        val_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_val, transform, lh_fmri, rh_fmri),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_imgs_dataloader = DataLoader(
            ImageDataset(test_imgs_paths, idxs_test, transform),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        return [train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader]

    else:
        lh_roi_idxs, rh_roi_idxs = get_roi_indexes(file_paths)

        train_loaders = []
        val_loaders = []

        for i in range(len(lh_roi_idxs)):
            target_lh = lh_fmri[:, lh_roi_idxs[i]]
            target_rh = rh_fmri[:, rh_roi_idxs[i]]

            train_imgs_dataloader = DataLoader(
                ImageDataset(
                    train_imgs_paths, idxs_train, transform, target_lh, target_rh
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )
            val_imgs_dataloader = DataLoader(
                ImageDataset(
                    train_imgs_paths, idxs_val, transform, target_lh, target_rh
                ),
                batch_size=batch_size,
                num_workers=num_workers,
            )

            train_loaders.append(train_imgs_dataloader)
            val_loaders.append(val_imgs_dataloader)

        test_imgs_dataloader = DataLoader(
            ImageDataset(test_imgs_paths, idxs_test, transform),
            batch_size=batch_size,
            num_workers=num_workers,
        )

        total_val_targets = DataLoader(
            ImageDataset(train_imgs_paths, idxs_val, transform, lh_fmri, rh_fmri)
        )
        val_loaders.append(total_val_targets)

        return [
            train_loaders,
            val_loaders,
            test_imgs_dataloader,
            lh_roi_idxs,
            rh_roi_idxs,
        ]
