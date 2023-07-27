import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from dataloader import get_roi_indexes, FeatureDataset_with_Clip
from utils import argObj
from models import CrossVAE_encoded_loss, MLP
from scipy.stats import pearsonr as corr
import argparse


def inference(models, dataloader, device, vae_model=False, test_flag=False):
    model_lh, model_rh = models
    model_lh.eval()
    model_rh.eval()
    model_lh.to(device)
    model_rh.to(device)

    if test_flag == False:
        preds_lh = []
        preds_rh = []
        targets_lh = []
        targets_rh = []
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="VAE Inference"):
                if vae_model:
                    data, clip_data = data
                    data = data.to(device)
                    clip_data = clip_data.to(device)
                    output_lh = model_lh(data, clip_data)
                
                else:
                    data = data.to(device)
                    output_lh = model_lh(data)
                target_lh, target_rh = target

                preds_lh.append(output_lh.cpu().numpy())
                targets_lh.append(target_lh.cpu().numpy())

                output_rh = model_rh(data, clip_data)
                preds_rh.append(output_rh.cpu().numpy())
                targets_rh.append(target_rh.cpu().numpy())

        preds_lh = np.concatenate(preds_lh, axis=0)
        preds_rh = np.concatenate(preds_rh, axis=0)
        targets_lh = np.concatenate(targets_lh, axis=0)
        targets_rh = np.concatenate(targets_rh, axis=0)

        return (preds_lh, preds_rh), (targets_lh, targets_rh)

    else:
        preds_lh = []
        preds_rh = []
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Inference"):
                data, clip_data = data
                data = data.to(device)
                clip_data = clip_data.to(device)

                output_lh = model_lh(data, clip_data)
                preds_lh.append(output_lh.cpu().numpy())

                output_rh = model_rh(data, clip_data)
                preds_rh.append(output_rh.cpu().numpy())

        preds_lh = np.concatenate(preds_lh, axis=0)
        preds_rh = np.concatenate(preds_rh, axis=0)

        return (preds_lh, preds_rh)