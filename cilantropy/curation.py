import json
import os

import cupy as cp
import numpy as np
import pandas as pd
import slay
from numpy.typing import NDArray
from tqdm import tqdm

from cilantropy.custom_metrics import (
    calc_presence_ratio,
    calc_sliding_RP_viol,
    calc_SNR,
    calc_wf_shape_metrics,
    calculate_noise_cutoff,
    extract_noise,
    noise_cutoff,
    presence_ratio,
)
from cilantropy.params import AutoCurateParams, CuratorParams
from cilantropy.rawdata import RawData


class Curator(object):
    """
    Class for curating spike sorted data.

    Attributes:
        cluster_metrics: pd.DataFrame
            Cluster metrics.
        spike_clusters: NDArray
            cluster_id for each spike. Shape (n_spikes,)
        n_clusters: int
            Number of clusters.
        cluster_ids: NDArray
            Unique cluster_ids.
        ks_folder: str
            Path to kilosort folder.
        raw_data: RawData
            RawData object.
        spike_times: NDArray
            Spike times. Shape (n_spikes,)
        params: CuratorParams
            Parameters for curation.
        cluster_metrics: pd.DataFrame
            Cluster metrics.
        mean_wf: NDArray
            Mean waveforms for each cluster. Shape (n_clusters, n_channels, pre_samples + post_samples)
        channel_pos: NDArray
            Channel positions. Shape (n_channels, 2)
        times_multi: NDArray
            Spike times for each cluster. Shape (n_clusters, n_spikes)
    """

    # TODO fix parameters
    def __init__(self, ks_folder: str, **kwargs) -> None:
        self.ks_folder = ks_folder
        self.params: dict
        self.n_clusters: int
        self.cluster_ids: NDArray
        self.raw_data: RawData
        self.cluster_metrics: pd.DataFrame
        self.mean_wf: NDArray
        self.channel_pos: NDArray
        self.spikes: NDArray
        self._calc_metrics(**kwargs)

    @property
    def n_clusters(self) -> int:
        if self.spike_clusters is None:
            return 0
        return np.max(self.spike_clusters) + 1

    @property
    def cluster_ids(self) -> NDArray:
        return np.arange(self.n_clusters)

    @property
    def n_channels(self) -> int:
        return self.params["n_chan"]

    @property
    def peak_channels(self) -> NDArray:
        return self.cluster_metrics["peak"].values

    @property
    def good_clusters(self) -> NDArray[np.bool_]:
        return self.cluster_metrics["label"].values == "good"

    @property
    def counts(self) -> NDArray:
        return self.cluster_metrics["n_spikes"].values

    def _load_params(self, **kwargs) -> None:
        params = kwargs
        params["KS_folder"] = self.ks_folder
        with open(os.path.join(self.ks_folder, "params.py")) as f:
            code = f.read()
            exec(code, {}, params)

        # update keynames
        params["data_path"] = params.pop("dat_path")

        # change dat_path to absolute filepath if not already
        if not os.path.isabs(params["data_path"]):
            params["data_path"] = os.path.abspath(
                os.path.join(self.ks_folder, params["data_path"])
            )
        params["n_chan"] = params.pop("n_channels_dat")

        self.params = CuratorParams().load(params)
        if not self.params.isinstance(dict):
            self.params = self.params._asdict()["data"]

    def _calc_metrics(self, **kwargs) -> None:
        metrics_path = os.path.join(self.ks_folder, "cilantro_metrics.tsv")
        load_metrics = os.path.exists(metrics_path) and not kwargs.get(
            "overwrite", False
        )

        # if not, load metrics from individual files
        self._load_params(**kwargs)
        self.raw_data = RawData(self.params["data_path"], self.params["n_chan"])
        # load spike_clusters
        try:
            self.spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_clusters.npy")
            ).flatten()
        except FileNotFoundError:
            self.spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_templates.npy")
            ).flatten()
        self.spike_times = np.load(
            os.path.join(self.ks_folder, "spike_times.npy")
        ).flatten()

        # we do not need spike times outside of data (negative or positive). resave as uint32 to save space.
        good_spikes = (self.spike_times >= 0) & (
            self.spike_times < len(self.raw_data.data)
        )
        if not good_spikes.all():
            self.spike_clusters = self.spike_clusters[good_spikes]
            self.spike_times = self.spike_times[good_spikes].astype(np.uint32)
            # resave spike_clusters
            np.save(
                os.path.join(self.ks_folder, "spike_clusters.npy"), self.spike_clusters
            )
            # resave spike_times
            np.save(os.path.join(self.ks_folder, "spike_times.npy"), self.spike_times)

        self.channel_pos = np.load(
            os.path.join(self.ks_folder, "channel_positions.npy")
        )

        # remove cluster_ids that are not in spike_clusters
        cluster_ids = np.unique(self.spike_clusters, return_counts=False)
        n_clusters = np.max(cluster_ids) + 1

        self.times_multi = slay.find_times_multi(
            self.spike_times,
            self.spike_clusters,
            np.arange(n_clusters),
            self.raw_data.data,
            self.params["pre_samples"],
            self.params["post_samples"],
        )
        cluster_ids = [
            i for i in cluster_ids if len(self.times_multi[i]) > 0
        ]  # fix cluster ids if no spikes in bounds

        self.mean_wf, _, self.spikes = slay.calc_mean_and_std_wf(
            self.params,
            n_clusters,
            cluster_ids,
            self.times_multi,
            self.raw_data.data,
            return_std=False,
            return_spikes=not load_metrics,
        )

        # load cilantro_metrics.tsv if it exists
        if load_metrics:
            self.cluster_metrics = pd.read_csv(
                metrics_path, sep="\t", index_col="cluster_id"
            )
            # will update values if merges happened as needed
            self.update_merged_metrics()
        else:
            self.create_dataframe(overwrite=kwargs.get("overwrite", False))

    def create_dataframe(self, overwrite) -> None:
        self.cluster_metrics = pd.DataFrame()
        self.cluster_metrics["cluster_id"] = self.cluster_ids
        self.cluster_metrics.set_index("cluster_id", inplace=True)

        # load cluster_group.tsv
        cl_labels = pd.read_csv(
            os.path.join(self.ks_folder, "cluster_group.tsv"),
            sep="\t",
            index_col="cluster_id",
        )
        if "label" not in cl_labels.columns:
            try:
                cl_labels.rename(columns={"KSLabel": "label"}, inplace=True)
            except KeyError:
                cl_labels.rename(columns={"group": "label"}, inplace=True)

        # merge KSLabel and group columns
        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            cl_labels,
            how="left",
            left_index=True,
            right_index=True,
        )
        self.cluster_metrics["n_spikes"] = [
            len(self.times_multi[i]) for i in self.cluster_ids
        ]
        # save as uint32
        self.cluster_metrics["n_spikes"] = self.cluster_metrics["n_spikes"].astype(
            "uint32"
        )

        # cluster waveform shapes
        wf_path = os.path.join(self.ks_folder, "cluster_wf_shape.tsv")
        if overwrite or not os.path.exists(wf_path):
            tqdm.write("Calculating waveform shape metrics...")
            num_peaks, num_troughs, wf_durs, spat_decay = calc_wf_shape_metrics(
                self.mean_wf, self.cluster_ids, self.channel_pos
            )
            wf_df = pd.DataFrame(
                {
                    "n_peaks": num_peaks,
                    "n_troughs": num_troughs,
                    "wf_dur": wf_durs,
                    "spat_decay": spat_decay,
                },
                index=self.cluster_ids,
            )
            wf_df.to_csv(wf_path, sep="\t", index=True, index_label="cluster_id")
            tqdm.write("WF file saved.")
        else:
            wf_df = pd.read_csv(wf_path, sep="\t", index_col="cluster_id")

        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            wf_df,
            how="left",
            left_index=True,
            right_index=True,
        )

        # refractory period violations
        cluster_rp_path = os.path.join(self.ks_folder, "cluster_RP_conf.tsv")
        if overwrite or not os.path.exists(cluster_rp_path):
            slid_rp_viols = calc_sliding_RP_viol(
                self.times_multi,
                self.cluster_ids,
                self.n_clusters,
            )
            srv_df = pd.DataFrame(
                {"slid_RP_viol": slid_rp_viols}, index=self.cluster_ids
            )
            srv_df.to_csv(
                cluster_rp_path, sep="\t", index=True, index_label="cluster_id"
            )
            tqdm.write("RP file saved.")
        else:
            srv_df = pd.read_csv(cluster_rp_path, sep="\t", index_col="cluster_id")

        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            srv_df,
            how="left",
            left_index=True,
            right_index=True,
        )
        # save as float32
        self.cluster_metrics["slid_RP_viol"] = self.cluster_metrics[
            "slid_RP_viol"
        ].astype("float32")

        channel_amplitudes = cp.max(self.mean_wf, 2) - cp.min(self.mean_wf, 2)
        amplitudes = cp.asnumpy(cp.max(channel_amplitudes, axis=1))
        self.cluster_metrics["amplitude"] = amplitudes.astype("float32")

        peaks = channel_amplitudes.argmax(1)
        self.cluster_metrics["peak"] = peaks.astype("uint16")

        self.cluster_metrics["firing_rate"] = self.cluster_metrics["n_spikes"] / (
            len(self.raw_data.data) / self.params["sample_rate"]
        )

        self.cluster_metrics["firing_rate"] = self.cluster_metrics[
            "firing_rate"
        ].astype("float32")

        # SNR
        snr_path = os.path.join(self.ks_folder, "cluster_SNR_good.tsv")
        tqdm.write("Calculating background standard deviation...")
        noise = extract_noise(
            self.raw_data.data,
            self.spike_times,
            self.params["pre_samples"],
            self.params["post_samples"],
        )
        noise_stds = cp.std(noise, axis=0) * slay.utils.get_bits_to_uV_factor(
            self.params
        )
        snrs = calc_SNR(self.mean_wf, noise_stds, self.cluster_ids)
        snr_df = pd.DataFrame({"SNR_good": snrs}, index=self.cluster_ids)
        snr_df.to_csv(
            snr_path,
            sep="\t",
            index=True,
            index_label="cluster_id",
        )
        tqdm.write("SNR file saved.")
        peak_channels = self.cluster_metrics["peak"].values
        stds = noise_stds[peak_channels]
        self.cluster_metrics["noise_stds"] = stds

        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            snr_df,
            left_index=True,
            right_index=True,
            how="left",
        )

        self.cluster_metrics["SNR_good"] = self.cluster_metrics["SNR_good"].astype(
            "float32"
        )

        # noise cutoff
        nc = calculate_noise_cutoff(
            self.spikes,
            self.cluster_metrics["peak"].values,
            self.cluster_ids,
            self.n_clusters,
        )
        self.cluster_metrics["noise_cutoff"] = nc

        # float32
        self.cluster_metrics["noise_cutoff"] = self.cluster_metrics[
            "noise_cutoff"
        ].astype("float32")

        # presence ratio
        calc_presence_ratio(self.cluster_metrics, self.times_multi)
        # float32
        self.cluster_metrics["presence_ratio"] = self.cluster_metrics[
            "presence_ratio"
        ].astype("float32")

        tqdm.write("Calculated metrics")

    def update_merged_metrics(self) -> None:
        # update any cluster_metrics that were merged
        try:
            with open(os.path.join(self.ks_folder, "automerge", "new2old.json")) as f:
                new2old = json.load(f)
                merges = {int(k): v for k, v in sorted(new2old.items())}
        except FileNotFoundError:
            tqdm.write("No need to update merge metrics as no merges found.")
            return

        # update mean_wf, cluster_labels, spike_times that were updated by slay
        self.spike_times = np.load(
            os.path.join(self.ks_folder, "spike_times.npy")
        ).flatten()
        self.spike_clusters = np.load(
            os.path.join(self.ks_folder, "spike_clusters.npy")
        ).flatten()
        self.mean_wf = np.load(os.path.join(self.ks_folder, "mean_waveforms.npy"))
        cluster_labels = pd.read_csv(
            os.path.join(self.ks_folder, "cluster_group.tsv"),
            sep="\t",
            index_col="cluster_id",
        )
        if "label_reason" not in cluster_labels.columns:
            cluster_labels["label_reason"] = ""

        # update times_multi from merges
        self.times_multi = slay.find_times_multi(
            self.spike_times,
            self.spike_clusters,
            np.arange(self.n_clusters),
            self.raw_data.data,
            self.params["pre_samples"],
            self.params["post_samples"],
        )

        for new_id, old_ids in merges.items():
            if new_id in self.cluster_metrics.index:
                continue

            # Reload spikes. TODO: Could be more efficient
            _, _, self.spikes = slay.calc_mean_and_std_wf(
                self.params,
                self.n_clusters,
                self.cluster_ids,
                self.times_multi,
                self.raw_data.data,
                return_std=False,
                return_spikes=True,
            )
            old_rows = self.cluster_metrics.loc[old_ids]

            # amplitude
            channel_amplitudes = (
                np.max(self.mean_wf[new_id], 1) - np.min(self.mean_wf[new_id], 1)
            ).flatten()
            new_amplitude = np.max(channel_amplitudes)

            # peak
            peak = np.argmax(channel_amplitudes, axis=0)

            # SNR_good
            snr = np.sum(old_rows["SNR_good"] * old_rows["n_spikes"]) / np.sum(
                old_rows["n_spikes"]
            )

            # slid_RP_viol
            slid_rp_viol = calc_sliding_RP_viol(self.times_multi, [new_id], 1)[0]

            # noise_cutoff
            sp_amplitudes = np.ptp(self.spikes[new_id], 1)
            _, nc, _ = noise_cutoff(sp_amplitudes)

            # presence_ratio
            pr = presence_ratio(self.times_multi[new_id])

            # firing rate
            fr = np.sum(old_rows["n_spikes"]) / (
                len(self.raw_data.data) / self.params["sample_rate"]
            )

            new_row = pd.DataFrame(
                {
                    "label": cluster_labels.loc[new_id, "label"],
                    "label_reason": cluster_labels.loc[new_id, "label_reason"],
                    "n_spikes": old_rows["n_spikes"].sum(),
                    "firing_rate": fr,
                    "SNR_good": snr,
                    "slid_RP_viol": slid_rp_viol,
                    "noise_cutoff": nc,
                    "presence_ratio": pr,
                    "peak": peak,
                    "amplitude": new_amplitude,
                },
                index=[new_id],
            )
            self.cluster_metrics = pd.concat(
                [self.cluster_metrics, new_row], axis=0, ignore_index=False
            )

    def auto_curate(self, args: dict = {}) -> None:
        tqdm.write("Auto-curating clusters...")
        schema = AutoCurateParams()
        params = schema.load(args)
        # reset labels
        self.cluster_metrics["label"] = "good"
        self.cluster_metrics["label_reason"] = ""

        # mark low-spike units as noise
        low_spike_units = self.cluster_metrics[
            self.cluster_metrics["firing_rate"] < params["min_fr"]
        ].index
        self.cluster_metrics.loc[low_spike_units, "label_reason"] = "low firing rate"
        self.cluster_metrics.loc[low_spike_units, "label"] = "noise"

        # mark low snr units as noise
        low_snr_units = self.cluster_metrics[
            (self.cluster_metrics["SNR_good"] < params["min_snr"])
            & (self.cluster_metrics["label"].isin(params["good_lbls"]))
        ].index
        self.cluster_metrics.loc[low_snr_units, "label_reason"] = "low SNR"
        self.cluster_metrics.loc[low_snr_units, "label"] = "noise"

        # mark units with high RP violations as mua
        high_rp_units = self.cluster_metrics[
            (self.cluster_metrics["slid_RP_viol"] > params["max_rp_viol"])
            & (self.cluster_metrics["label"].isin(params["good_lbls"]))
        ].index
        self.cluster_metrics.loc[high_rp_units, "label_reason"] = "high RP violations"
        self.cluster_metrics.loc[high_rp_units, "label"] = "mua"

        # mark units with too many peaks and troughs as noise
        high_peaks_units = self.cluster_metrics[
            (self.cluster_metrics["n_peaks"] > params["max_peaks"])
            & (self.cluster_metrics["label"].isin(params["good_lbls"]))
        ].index
        self.cluster_metrics.loc[high_peaks_units, "label_reason"] = "too many peaks"
        self.cluster_metrics.loc[high_peaks_units, "label"] = "noise"

        high_troughs_units = self.cluster_metrics[
            (self.cluster_metrics["n_troughs"] > params["max_troughs"])
            & (self.cluster_metrics["label"].isin(params["good_lbls"]))
        ].index
        self.cluster_metrics.loc[high_troughs_units, "label_reason"] = (
            "too many troughs"
        )
        self.cluster_metrics.loc[high_troughs_units, "label"] = "noise"

        # mark units with long waveform duration as noise
        long_wf_units = self.cluster_metrics[
            (self.cluster_metrics["wf_dur"] > params["max_wf_dur"])
            & (self.cluster_metrics["label"].isin(params["good_lbls"]))
        ].index
        self.cluster_metrics.loc[long_wf_units, "label_reason"] = "long waveform"
        self.cluster_metrics.loc[long_wf_units, "label"] = "noise"

        # mark units with low spatial decay as noise
        low_spat_decay_units = self.cluster_metrics[
            (self.cluster_metrics["spat_decay"] < params["min_spat_decay"])
            & (self.cluster_metrics["label"].isin(params["good_lbls"]))
        ].index
        self.cluster_metrics.loc[low_spat_decay_units, "label_reason"] = (
            "low spatial decay"
        )
        self.cluster_metrics.loc[low_spat_decay_units, "label"] = "noise"

        self.save_labels()

    def post_merge_curation(self, args: dict = {}) -> None:
        tqdm.write("Post-merge curation...")
        schema = AutoCurateParams()
        params = schema.load(args)

        self.update_merged_metrics()

        high_amp_units = self.cluster_metrics[
            (self.cluster_metrics["noise_cutoff"] > params["max_noise_cutoff"])
            & (self.cluster_metrics["label"] == "good")
        ].index
        self.cluster_metrics.loc[high_amp_units, "label_reason"] = (
            "high amplitude cutoff"
        )
        self.cluster_metrics.loc[high_amp_units, "label"] = "inc"

        low_pr_units = self.cluster_metrics[
            (self.cluster_metrics["presence_ratio"] < params["min_pr"])
            & (self.cluster_metrics["label"] == "good")
        ].index
        self.cluster_metrics.loc[low_pr_units, "label_reason"] = "low presence ratio"
        self.cluster_metrics.loc[low_pr_units, "label"] = "inc"

        self.save_labels()

    def save_labels(self) -> None:
        # save new cluster_group.tsv
        cluster_labels = self.cluster_metrics[["label", "label_reason"]]
        cluster_labels.to_csv(
            os.path.join(self.ks_folder, "cluster_group.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        tqdm.write("Saving cluster metrics to cilantro_metrics.tsv...")
        # save cluster_metrics in cilantro_metrics.tsv
        self.cluster_metrics.to_csv(
            os.path.join(self.ks_folder, "cilantro_metrics.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )

        self.raw_data.close()
