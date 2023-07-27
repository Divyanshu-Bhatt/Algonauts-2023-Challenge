import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr as corr
import os


def prediction_metric(fmri_pred, fmri):
    lh_fmri_pred, rh_fmri_pred = fmri_pred
    lh_fmri, rh_fmri = fmri

    # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(lh_fmri_pred.shape[1])

    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh_fmri_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_pred[:, v], lh_fmri[:, v])[0]

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(rh_fmri_pred.shape[1])
    
    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh_fmri_pred.shape[1])):
        rh_correlation[v] = corr(rh_fmri_pred[:, v], rh_fmri[:, v])[0]

    return (lh_correlation, rh_correlation)


def visualising_metrics(file_paths, correlation, args, name=None):
    lh_correlation, rh_correlation = correlation

    # Load the ROI classes mapping dictionaries
    roi_mapping_files = [
        "mapping_prf-visualrois.npy",
        "mapping_floc-bodies.npy",
        "mapping_floc-faces.npy",
        "mapping_floc-places.npy",
        "mapping_floc-words.npy",
        "mapping_streams.npy",
    ]
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(
            np.load(os.path.join(file_paths.data_dir, "roi_masks", r),
                    allow_pickle=True).item())

    # Load the ROI brain surface maps
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
                os.path.join(file_paths.data_dir, "roi_masks",
                             lh_challenge_roi_files[r])))
        rh_challenge_rois.append(
            np.load(
                os.path.join(file_paths.data_dir, "roi_masks",
                             rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if (
                    r2[0] != 0
            ):  # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append("All vertices")
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)

    # Create the plot
    lh_median_roi_correlation = [
        np.median(lh_roi_correlation[r])
        for r in range(len(lh_roi_correlation))
    ]
    rh_median_roi_correlation = [
        np.median(rh_roi_correlation[r])
        for r in range(len(rh_roi_correlation))
    ]
    
    x = np.arange(len(roi_names))
    width = 0.30

    plt.figure(figsize=(18, 6))
    plt.bar(x - width / 2,
            lh_median_roi_correlation,
            width,
            label="Left Hemisphere")
    plt.bar(x + width / 2,
            rh_median_roi_correlation,
            width,
            label="Right Hemishpere")
    
    plt.xlim(left=min(x) - 0.5, right=max(x) + 0.5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel("ROIs")
    plt.ylabel("Median Pearson's $r$")
    
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.legend(frameon=True, loc=1)
    
    plt.title(
        "Median Pearson's for each ROI: Subject " + str(args.subj) +
        " median: " +
        str(np.nanmedian(lh_median_roi_correlation +
                         rh_median_roi_correlation)))
    
    if not os.path.isdir("plots"):
        os.makedirs("plots")

    if name is not None:
        # plt.savefig(f"plots/Subj-" + str(args.subj) + "-" + name + ".png",
        #             dpi=300,
        #             bbox_inches="tight")
        print(args.model_dir + "/Subj-" + str(args.subj)+ name + ".png")
        plt.savefig(args.model_dir + "/Subj-" + str(args.subj) + name + ".png",
                    dpi=300,
                    bbox_inches="tight")
    else:
        # plt.savefig(f"plots/Subj-" + str(args.subj) + ".png",
        #             dpi=300,
        #             bbox_inches="tight")
        print(args.model_dir + "/Subj-" + str(args.subj)+ ".png")
        plt.savefig(args.model_dir + "/Subj-" + str(args.subj) + "-" + ".png",
                    dpi=300,
                    bbox_inches="tight")
