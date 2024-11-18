import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import slay
from marshmallow import EXCLUDE
from numpy.typing import NDArray
from scipy import signal, stats
from scipy.ndimage._filters import gaussian_filter1d
from slay.schemas import CustomMetricsParams
from tqdm import tqdm

logger = logging.getLogger("burst-detector")


# --------------------------------------------- DATA HANDLING HELPERS -------------------------------------------------
def extract_noise(
    data: NDArray,
    times: NDArray,
    pre_samples: int,
    post_samples: int,
    n_chan: int,
) -> NDArray:
    """
    Extracts noise snippets from the given data based on the provided spike times.
    Args:
        data (NDArray): The input data array.
        times (NDArray): The spike times array.
        pre_samples (int): The number of samples to include before each spike time.
        post_samples (int): The number of samples to include after each spike time.
        n_chan (int): The number of channels.
    Returns:
        NDArray: The noise snippets array.
    """
    # extraction loop
    # TODO make 2000 a parameter?
    noise = np.zeros((n_chan, 2000 * (pre_samples + post_samples)))

    ind = 0
    for i in range(1, times.shape[0]):
        # portions where no spikes occur
        if (
            (times[i] - pre_samples) - (times[i - 1] + post_samples)
        ) > pre_samples + post_samples:

            # randomly select 10 times and channels in range
            noise_times = np.random.choice(
                range(int(times[i - 1] + post_samples), int(times[i] - pre_samples)),
                10,
                replace=False,
            )

            for j in range(10):
                start = noise_times[j] - pre_samples
                end = noise_times[j] + post_samples
                snip_len = pre_samples + post_samples
                noise[:, ind * snip_len : (ind + 1) * snip_len] = data[start:end, :].T
                noise[:, ind * snip_len : (ind + 1) * snip_len] = np.nan_to_num(
                    noise[:, ind : ind + pre_samples + post_samples]
                )
                ind += 1

                if ind >= 2000:
                    return noise

    if ind < 2000 - 1:
        noise = noise[:, : (ind - 1) * (post_samples + pre_samples)]
    return noise


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

    times_multi = slay.find_times_multi(
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

    mean_wf, _, _ = slay.calc_mean_and_std_wf(
        params,
        n_clust,
        good_ids,
        times_multi,
        data,
        return_std=False,
        return_spikes=False,
    )

    logger.info("Calculating background standard deviation...")
    noise = extract_noise(
        data, times, params["pre_samples"], params["post_samples"], n_chan
    )
    noise_stds = np.std(noise, axis=1)

    snrs = calc_SNR(mean_wf, noise_stds, good_ids)
    slid_rp_viols = calc_sliding_RP_viol(times_multi, good_ids, n_clust)
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
    mean_wf: NDArray[np.float_],
    noise_stds: NDArray[np.float_],
    clust_ids: NDArray[np.bool_],
) -> NDArray[np.float_]:
    """
    Calculates the signal-to-noise ratio (SNR) for each waveform.
    Parameters:
    - mean_wf (NDArray): Array of shape (n_waveforms, n_samples, n_channels) representing the mean waveforms.
    - noise_stds (NDArray): Array of shape (n_channels,) representing the standard deviation of the noise for each channel.
    - clust_ids (NDArray): Cluster ids to calculate SNR for. Rest will be zeros.
    Returns:
    - snrs (NDArray): Array of shape (n_waveforms,) representing the SNR for each waveform.
    """

    logger.info("Calculating peak channels and amplitudes")
    # calculate peak chans, amplitudes
    peak_chans = np.argmax(np.max(np.abs(mean_wf), axis=-1), axis=-1)
    peak_chans[peak_chans == 384] = 383  # TODO hacky fix
    amps = np.max(np.max(np.abs(mean_wf), axis=-1), axis=-1)

    # calculate snrs
    snrs = np.zeros(mean_wf.shape[0])
    for i in clust_ids:
        snrs[i] = amps[i] / noise_stds[int(peak_chans[i])]

    return snrs


def max_cont(fr: float, rp: float, rec_dur: float, acc_cont: float) -> float:
    """
    Calculates the maximum number of continuous events expected based on the given parameters.
    Args:
        fr (float): The firing rate of the events.
        rp (float): The refractory period between events.
        rec_dur (float): The recording duration.
        acc_cont (float): The acceptable level of continuity.
    Returns:
        cnt_exp (float): The maximum number of continuous events expected.
    """
    time_for_viol = rp * fr * rec_dur * 2
    cnt_exp = acc_cont * time_for_viol

    return cnt_exp


def calc_sliding_RP_viol(
    times_multi: list[NDArray[np.float_]],
    clust_ids: NDArray[np.int_],
    n_clust: int,
    bin_size: float = 0.25,
    acceptThresh: float = 0.25,
) -> NDArray[np.float32]:
    """
    Calculate the sliding refractory period violation confidence for each cluster.
    Args:
        times_multi (list[NDArray[np.float_]]): A list of arrays containing spike times for each cluster.
        clust_ids (NDArray[np.int_]): An array indicating cluster_ids to process. Should be "good" clusters.
        n_clust (int): The total number of clusters (shape of mean_wf or max_clust_id + 1).
        bin_size (float, optional): The size of each bin in milliseconds. Defaults to 0.25.
        acceptThresh (float, optional): The threshold for accepting refractory period violations. Defaults to 0.25.
    Returns:
        NDArray[np.float32]: An array containing the refractory period violation confidence for each cluster.
    """
    b = np.arange(0, 10.25, bin_size) / 1000
    bTestIdx = np.array([1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40], dtype="int8")
    bTest = [b[i] for i in bTestIdx]  # -1 bc 0th bin corresponds to 0-0.5 ms

    RP_conf = np.zeros(n_clust, dtype=np.float32)

    for i in tqdm(range(len(clust_ids)), desc="Calculating RP viol confs"):
        times = times_multi[clust_ids[i]] / 30000
        if times.shape[0] > 1:
            # calculate and avg halves of acg
            acg = slay.auto_correlogram(times, 2, bin_size / 1000, 5 / 30000)
            half_len = int(acg.shape[0] / 2)
            acg[half_len:] = (acg[half_len:] + acg[:half_len][::-1]) / 2
            acg = acg[half_len:]

            acg_cumsum = np.cumsum(acg)
            sum_res = acg_cumsum[bTestIdx - 1]

            # calculate max violations per refractory period size
            num_bins_2s = acg.shape[0]
            num_bins_1s = int(num_bins_2s / 2)
            bin_rate = np.mean(acg[num_bins_1s:num_bins_2s])
            max_conts = np.array(bTest) / bin_size * 1000 * bin_rate * acceptThresh

            # compute confidence of less than acceptThresh contamination at each refractory period
            confs = []
            for j, cnt in enumerate(sum_res):
                confs.append(1 - stats.poisson.cdf(cnt, max_conts[j]))
            RP_conf[i] = 1 - max(confs)

    return RP_conf


def calc_wf_shape_metrics(
    mean_wf: NDArray[np.float_],
    clust_ids: NDArray[np.int_],
    channel_pos: NDArray[np.float_],
    minThreshDetectPeaksTroughs: float = 0.2,
) -> Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float_], NDArray[np.float_]]:
    """
    Calculate waveform shape metrics.
    Args:
        mean_wf (NDArray[np.float_]): Array of mean waveforms.
        clust_ids (NDArray[np.int_]): Array of cluster id's to calcualte waveform metrics for, typically "good" clusters.
        channel_pos (NDArray[np.float_]): Array of channel positions.
        minThreshDetectPeaksTroughs (float, optional): Minimum threshold to detect peaks and troughs. Defaults to 0.2.
    Returns:
        Tuple[NDArray[int], NDArray[int], NDArray[float], NDArray[float]]: A tuple containing the following metrics:
            - num_peaks: Array of the number of peaks for each waveform.
            - num_troughs: Array of the number of troughs for each waveform.
            - wf_durs: Array of waveform durations for each waveform.
            - spat_decays: Array of spatial decay values for each waveform.
    """
    peak_chans = np.argmax(np.max(np.abs(mean_wf), axis=-1), axis=-1)
    peak_chans[peak_chans >= 383] = 382  # TODO hacky fix

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


def calc_noise_cutoff(
    cluster_metrics,
    quantile_length=0.25,
    n_bins=100,
    nc_threshold=5,
    percent_threshold=0.10,
):
    """
    Adapted from J. Colonell's ecephys_spike_sorting code.
    A new metric to determine whether a unit's amplitude distribution is cut off
    (at floor), without assuming a Gaussian distribution.
    This metric takes the amplitude distribution, computes the mean and std
    of an upper quartile of the distribution, and determines how many standard
    deviations away from that mean a lower quartile lies.
    Parameters
    ----------
    custom_metrics : dataframe
        The amplitudes (in uV) of the spikes.
    quantile_length : float
        The size of the upper quartile of the amplitude distribution.
    n_bins : int
        The number of bins used to compute a histogram of the amplitude
        distribution.
    n_low_bins : int
        The number of bins used in the lower part of the distribution (where
        cutoff is determined).
     nc_threshold: float
        the noise cutoff result has to be lower than this for a neuron to fail
    percent_threshold: float
        the first bin has to be greater than percent_threshold for neuron the to fail
    Returns
    -------
    cutoff : float
        Number of standard deviations that the lower mean is outside of the
        mean of the upper quartile.
    See Also
    --------
    missed_spikes_est
    Examples
    --------
    1) Compute whether a unit's amplitude distribution is cut off
        >>> amps = spks_b['amps'][unit_idxs]
        >>> cutoff = bb.metrics.noise_cutoff(amps, quantile_length=.25, n_bins=100)
    """
    if "noise_cutoff" not in cluster_metrics.columns:
        cluster_metrics["noise_cutoff"] = pd.NA

    for cluster in tqdm(cluster_metrics.index, desc="Calculating noise cutoffs"):
        if pd.isna(cluster_metrics.loc[cluster, "noise_cutoff"]):
            _, cutoff, _ = noise_cutoff(
                cluster_metrics.loc[cluster, "amplitude"],
                quantile_length,
                n_bins,
                nc_threshold,
                percent_threshold,
            )
            cluster_metrics.loc[cluster, "noise_cutoff"] = cutoff


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


def calc_presence_ratio(cluster_metrics, times_multi, num_bins=100):
    # adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/blob/archive/ecephys_spike_sorting/modules/quality_metrics/metrics.py
    # add column for presence ratio if it doesn't exist
    if "presence_ratio" not in cluster_metrics.columns:
        cluster_metrics["presence_ratio"] = pd.NA

    for cluster in tqdm(cluster_metrics.index, desc="Calculating presence ratios"):
        if pd.isna(cluster_metrics.loc[cluster, "presence_ratio"]):
            spike_times = times_multi[cluster]
            pr = presence_ratio(spike_times, num_bins)
            cluster_metrics.loc[cluster, "presence_ratio"] = pr
    return cluster_metrics


def presence_ratio(spike_times, num_bins=100):
    if not np.all(np.isnan(spike_times)) and len(spike_times) > 0:
        min_time = np.min(spike_times)
        max_time = np.max(spike_times)
        hist, _ = np.histogram(spike_times, np.linspace(min_time, max_time, num_bins))
        pr = np.sum(hist > 0) / num_bins
        return pr
    return np.nan
