import os
from typing import Tuple

import cupy as cp
import npx_utils as npx
import numpy as np
import pandas as pd
import slay
from marshmallow import EXCLUDE
from numpy.typing import NDArray
from scipy import signal
from slay.schemas import CustomMetricsParams
from tqdm import tqdm


def extract_noise(data, times, pre_samples, post_samples, max_snippets=-1):
    """
    Extract snippets of noise from the data.
    Args:
        data (NDArray): The data to extract noise from.
        times (NDArray): The spike times.
        post_samples (int): The number of samples after the spike time to include.
        pre_samples (int): The number of samples before the spike time to include.
        max_snippets (int, optional): The maximum number of snippets to extract. Defaults to 300000.
    Returns:
        NDArray: The extracted noise snippets. Shape (max_snippets, n_channels).
    """
    total_samples = len(data)
    signal_mask = cp.zeros(total_samples, dtype=bool)
    for time in times:
        start_idx = max(0, time - pre_samples)
        end_idx = min(total_samples, time + post_samples + 1)
        signal_mask[start_idx:end_idx] = True
    noise_indices = cp.where(~signal_mask)[0]

    # Get min between max_snippets and available memory
    snippet_size_bytes = data.shape[1] * np.dtype(data.dtype).itemsize
    available_memory = cp.cuda.runtime.memGetInfo()[0] * 0.1  # add buffer
    max_snippets_by_memory = int(available_memory / snippet_size_bytes)
    if max_snippets == -1:
        max_snippets = max_snippets_by_memory
    else:
        max_snippets = min(max_snippets, max_snippets_by_memory)
    if max_snippets <= 0:
        raise MemoryError("Insufficient memory to extract any noise snippets.")

    if max_snippets < len(noise_indices):
        noise_indices = cp.random.choice(noise_indices, max_snippets, replace=False)
    noise_samples = data[noise_indices.get(), :]
    return noise_samples


def custom_metrics(args: dict = None) -> None:
    """
    Calculate various metrics for spike sorting.
    """
    args = slay.parse_kilosort_params(args)
    schema = CustomMetricsParams(unknown=EXCLUDE)
    params = schema.load(args)

    ks_folder = params["KS_folder"]
    data_filepath = params["data_filepath"]
    n_chan = params["n_chan"]

    # load stuff
    times = np.load(os.path.join(ks_folder, "spike_times.npy")).flatten()
    clusters = np.load(os.path.join(ks_folder, "spike_clusters.npy")).flatten()
    n_clust = clusters.max() + 1
    channel_pos = np.load(os.path.join(ks_folder, "channel_positions.npy"))

    rawData = np.memmap(data_filepath, dtype=np.int16, mode="r")
    data = np.reshape(rawData, (int(rawData.size / n_chan), n_chan))

    times_multi = npx.find_times_multi(
        times,
        clusters,
        np.arange(clusters.max() + 1),
        data,
        params["pre_samples"],
        params["post_samples"],
    )

    # skip empty ids
    good_ids = np.unique(clusters)
    cl_good = np.zeros(n_clust, dtype=bool)
    cl_good[good_ids] = True

    mean_wf = npx.calc_mean_wf(params, n_clust, good_ids, times_multi, data)

    tqdm.write("Calculating background standard deviation...")
    noise = extract_noise(data, times, params["pre_samples"], params["post_samples"])
    noise_stds = np.std(noise, axis=1)

    snrs = calc_SNR(mean_wf, noise_stds, good_ids)
    slid_rp_viols = npx.calc_sliding_RP_viol(
        times_multi, good_ids, n_clust, params["sample_rate"]
    )
    num_peaks, num_troughs, wf_durs, spat_decays = calc_wf_shape_metrics(
        mean_wf, good_ids, channel_pos
    )

    # make dataframes
    cl_ids = np.arange(n_clust)

    snr_df = pd.DataFrame({"cluster_id": cl_ids, "SNR_good": snrs})
    srv_df = pd.DataFrame({"cluster_id": cl_ids, "slid_RP_viol": slid_rp_viols})
    wf_df = pd.DataFrame(
        {
            "cluster_id": cl_ids,
            "num_peaks": num_peaks,
            "num_troughs": num_troughs,
            "wf_dur": wf_durs,
            "spat_decays": spat_decays,
        }
    )

    # write tsv
    snr_df.to_csv(os.path.join(ks_folder, "cluster_SNR_good.tsv"), sep="\t")
    srv_df.to_csv(os.path.join(ks_folder, "cluster_RP_conf.tsv"), sep="\t")
    wf_df.to_csv(os.path.join(ks_folder, "cluster_wf_shape.tsv"), sep="\t")


def calc_SNR(
    mean_wf: NDArray[np.float64],
    noise_stds: NDArray[np.float64],
    clust_ids: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Calculates the signal-to-noise ratio (SNR) for each waveform.
    Parameters:
    - mean_wf (NDArray): Array of shape (n_waveforms, n_channels, n_samples) representing the mean waveforms.
    - noise_stds (NDArray): Array of shape (n_channels,) representing the standard deviation of the noise for each channel.
    - clust_ids (NDArray): Cluster ids to calculate SNR for. Rest will be zeros.
    Returns:
    - snrs (NDArray): Array of shape (n_waveforms,) representing the SNR for each waveform.
    """

    tqdm.write("Calculating peak channels and amplitudes")
    # calculate peak chans, amplitudes
    n_chans = mean_wf.shape[1]
    peak_chans = np.argmax(np.ptp(mean_wf, axis=-1), axis=-1)
    peak_chans[peak_chans >= n_chans] = n_chans - 1
    amps = np.max(np.ptp(mean_wf, axis=-1), axis=-1)

    # calculate snrs
    peak_noise = noise_stds[peak_chans]
    snrs = amps / (2 * peak_noise)

    return snrs


def calc_wf_shape_metrics(
    mean_wf: NDArray[np.float64],
    clust_ids: NDArray[np.int_],
    channel_pos: NDArray[np.float64],
    minThreshDetectPeaksTroughs: float = 0.2,
) -> Tuple[
    NDArray[np.int_], NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Calculate waveform shape metrics.
    Args:
        mean_wf (NDArray[np.float64]): Array of mean waveforms.
        clust_ids (NDArray[np.int_]): Array of cluster id's to calcualte waveform metrics for, typically "good" clusters.
        channel_pos (NDArray[np.float64]): Array of channel positions.
        minThreshDetectPeaksTroughs (float, optional): Minimum threshold to detect peaks and troughs. Defaults to 0.2.
    Returns:
        Tuple[NDArray[int], NDArray[int], NDArray[float], NDArray[float]]: A tuple containing the following metrics:
            - num_peaks: Array of the number of peaks for each waveform.
            - num_troughs: Array of the number of troughs for each waveform.
            - wf_durs: Array of waveform durations for each waveform.
            - spat_decays: Array of spatial decay values for each waveform.
    """
    peak_chans = np.argmax(np.max(np.abs(mean_wf), axis=-1), axis=-1)
    peak_chans[peak_chans >= channel_pos.shape[0]] = (
        channel_pos.shape[0] - 1
    )  # TODO hacky fix

    num_peaks = np.zeros(mean_wf.shape[0], dtype="int8")
    num_troughs = np.zeros(mean_wf.shape[0], dtype="int8")
    wf_durs = np.zeros(mean_wf.shape[0], dtype=np.float32)
    spat_decays = np.zeros(mean_wf.shape[0], dtype=np.float32)

    for i in clust_ids:
        peak_wf = mean_wf[i, peak_chans[i], :]

        # count peaks and troughs
        minProminence = minThreshDetectPeaksTroughs * np.max(np.abs(peak_chans))
        peak_locs, _ = signal.find_peaks(peak_wf, prominence=minProminence)
        trough_locs, _ = signal.find_peaks(-1 * peak_wf, prominence=minProminence)
        num_peaks[i] = max(peak_locs.shape[0], 1)
        num_troughs[i] = max(trough_locs.shape[0], 1)

        # calculate wf width
        peak_loc = np.argmax(peak_wf)
        trough_loc = np.argmax(-1 * peak_wf)
        wf_dur = np.abs(peak_loc - trough_loc) / 30
        wf_durs[i] = wf_dur

        # calculate spatial decay
        channels_with_same_x = np.squeeze(
            np.argwhere(np.abs(channel_pos[:, 0] - channel_pos[peak_chans[i], 0]) <= 33)
        )
        if channels_with_same_x.shape[0] > 5:
            peak_idx = np.squeeze(np.argwhere(channels_with_same_x == peak_chans[i]))

            if peak_idx > 5:
                channels_for_decay_fit = channels_with_same_x[
                    peak_idx : peak_idx - 5 : -1
                ]
            else:
                channels_for_decay_fit = channels_with_same_x[peak_idx : peak_idx + 5]

            spatialDecayPoints = np.max(
                np.abs(mean_wf[i, channels_for_decay_fit, :]), axis=0
            )
            estimatedUnitXY = channel_pos[peak_chans[i], :]
            relativePositionsXY = (
                channel_pos[channels_for_decay_fit, :] - estimatedUnitXY
            )
            channelDists_relative = np.sqrt(np.nansum(relativePositionsXY**2, axis=1))

            indSort = np.argsort(channelDists_relative)
            spatialDecayPoints_norm = spatialDecayPoints[indSort]
            spatialDecayFit = np.polyfit(
                channelDists_relative[indSort], spatialDecayPoints_norm, 1
            )

            spat_decays[i] = spatialDecayFit[0]

    return num_peaks, num_troughs, wf_durs, spat_decays


def calculate_noise_cutoff(spikes, peak, cluster_ids, total_units):
    cutoff = np.zeros((total_units,))

    for cluster_id in tqdm(cluster_ids, desc="Calculating noise cutoffs"):
        # get amplitudes for spike_times
        try:
            cl_spikes = spikes[cluster_id][:, peak[cluster_id], :]
        except KeyError:
            cutoff[cluster_id] = np.nan
            continue
        spike_amps = np.ptp(cl_spikes, axis=1)
        cutoff[cluster_id] = noise_cutoff(amps=spike_amps)[
            1
        ]  # TODO can return others if we want

    return cutoff


def noise_cutoff(
    amps, quantile_length=0.25, n_bins=100, nc_threshold=5, percent_threshold=0.10
):
    cutoff = np.float64(np.nan)
    first_low_quantile = np.float64(np.nan)
    fail_criteria = np.ones(1).astype(bool)[0]

    if (
        not np.all(np.isnan(amps)) and amps.size > 1
    ):  # ensure there are amplitudes available to analyze
        bins_list = np.linspace(
            0, np.max(amps), n_bins
        )  # list of bins to compute the amplitude histogram
        n, _ = np.histogram(amps, bins=bins_list)  # construct amplitude histogram
        idx_peak = np.argmax(n)  # peak of amplitude distribution
        # don't count zeros #len(n) - idx_peak, compute the length of the top half of the distribution -- ignoring zero bins
        length_top_half = len(np.where(n[idx_peak:-1] > 0)[0])
        # the remaining part of the distribution, which we will compare the low quantile to
        high_quantile = 2 * quantile_length
        # the first bin (index) of the high quantile part of the distribution
        high_quantile_start_ind = int(
            np.ceil(high_quantile * length_top_half + idx_peak)
        )
        # bins to consider in the high quantile (of all non-zero bins)
        indices_bins_high_quantile = np.arange(high_quantile_start_ind, len(n))
        idx_use = np.where(n[indices_bins_high_quantile] >= 1)[0]

        if (
            len(n[indices_bins_high_quantile]) > 0
        ):  # ensure there are amplitudes in these bins
            # mean of all amp values in high quantile bins
            mean_high_quantile = np.mean(n[indices_bins_high_quantile][idx_use])
            std_high_quantile = np.std(n[indices_bins_high_quantile][idx_use])
            if std_high_quantile > 0:
                first_low_quantile = n[(n != 0)][1]  # take the second bin
                cutoff = (first_low_quantile - mean_high_quantile) / std_high_quantile
                peak_bin_height = np.max(n)
                percent_of_peak = percent_threshold * peak_bin_height

                fail_criteria = (cutoff > nc_threshold) & (
                    first_low_quantile > percent_of_peak
                )

    nc_pass = ~fail_criteria
    return nc_pass, cutoff, first_low_quantile


def calc_presence_ratio(
    cluster_metrics, times_multi, recording_duration_s, bin_size_s=20
):
    # adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/blob/archive/ecephys_spike_sorting/modules/quality_metrics/metrics.py
    # add column for presence ratio if it doesn't exist
    if "presence_ratio" not in cluster_metrics.columns:
        cluster_metrics["presence_ratio"] = pd.NA

    # calculate number of bins
    for cluster in tqdm(cluster_metrics.index, desc="Calculating presence ratios"):
        if pd.isna(cluster_metrics.loc[cluster, "presence_ratio"]):
            spike_times = times_multi[cluster]
            pr = presence_ratio(spike_times, recording_duration_s, bin_size_s)
            cluster_metrics.loc[cluster, "presence_ratio"] = pr
    return cluster_metrics


def presence_ratio(spike_times, recording_duration_s, bin_size_s=20):
    num_bins = int(recording_duration_s / bin_size_s)
    if not np.all(np.isnan(spike_times)) and len(spike_times) > 0:
        hist, _ = np.histogram(
            spike_times, np.linspace(0, recording_duration_s * 1000, num_bins)
        )
        pr = np.sum(hist > 0) / num_bins
        return pr
    return np.nan
