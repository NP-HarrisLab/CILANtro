import numpy as np
import pandas as pd
from scipy.ndimage._filters import gaussian_filter1d
from tqdm import tqdm


def calc_amplitude_cutoff(cluster_metrics, num_bins=500, smoothing_factor=3):
    # adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/blob/archive/ecephys_spike_sorting/modules/quality_metrics/metrics.py
    # add column for amplitude cutoff if it doesn't exist
    if "amplitude_cutoff" not in cluster_metrics.columns:
        cluster_metrics["amplitude_cutoff"] = pd.NA

    # calculate amplitude cutoff for each cluster if it is not already calculated
    for cluster in tqdm(cluster_metrics.index, desc="Calculating amplitude cutoffs"):
        if pd.isna(cluster_metrics.loc[cluster, "amplitude_cutoff"]):
            amplitudes = cluster_metrics.loc[cluster, "amplitude"]
            hist, bin_edges = np.histogram(amplitudes, num_bins, density=True)
            pdf = gaussian_filter1d(hist, smoothing_factor)
            support = bin_edges[:-1]
            peak_idx = np.argmax(pdf)
            G = np.argmin(np.abs(pdf[peak_idx:] - pdf[0])) + peak_idx
            bin_size = np.mean(np.diff(support))
            fraction_missing = np.sum(pdf[G:]) * bin_size
            fraction_missing = np.min([fraction_missing, 0.5])
            cluster_metrics.loc[cluster, "amplitude_cutoff"] = fraction_missing
    return cluster_metrics


def calc_presence_ratio(cluster_metrics, num_bins=100):
    # adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/blob/archive/ecephys_spike_sorting/modules/quality_metrics/metrics.py
    # add column for presence ratio if it doesn't exist
    if "presence_ratio" not in cluster_metrics.columns:
        cluster_metrics["presence_ratio"] = pd.NA

    for cluster in tqdm(cluster_metrics.index, desc="Calculating presence ratios"):
        if pd.isna(cluster_metrics.loc[cluster, "presence_ratio"]):
            spike_times = cluster_metrics.loc[cluster, "spike_times"]
            min_time = np.min(spike_times)
            max_time = np.max(spike_times)
            hist, _ = np.histogram(
                spike_times, np.linspace(min_time, max_time, num_bins)
            )
            pr = np.sum(hist > 0) / num_bins
            cluster_metrics.loc[cluster, "presence_ratio"] = pr
    return cluster_metrics
