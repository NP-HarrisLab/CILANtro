import json
import os

import cupy as cp
import npx_utils as npx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from cilantropy.custom_metrics import (
    calc_presence_ratio,
    calc_SNR,
    calc_wf_shape_metrics,
    calculate_noise_cutoff,
    extract_noise,
    noise_cutoff,
    presence_ratio,
)
from cilantropy.params import AutoCurateParams, CuratorParams, PostMergeCurationParams
from cilantropy.rawdata import RawData


class Curator(object):
    """
    Class for curating spike sorted data.

    Attributes:
        cluster_metrics: pd.DataFrame
            Cluster metrics.
        spike_clusters: NDArray
            cluster_id for each spike. Shape (num_spikes,)
        n_clusters: int
            Number of clusters.
        cluster_ids: NDArray
            Unique cluster_ids.
        ks_folder: str
            Path to kilosort folder.
        raw_data: RawData
            RawData object.
        spike_times: NDArray
            Spike times. Shape (num_spikes,)
        params: CuratorParams
            Parameters for curation.
        cluster_metrics: pd.DataFrame
            Cluster metrics.
        mean_wf: NDArray
            Mean waveforms for each cluster. Shape (n_clusters, n_channels, pre_samples + post_samples)
        channel_pos: NDArray
            Channel positions. Shape (n_channels, 2)
        times_multi: NDArray
            Spike times for each cluster. Shape (n_clusters, num_spikes)
    """

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
        self.params_file: str = os.path.join(self.ks_folder, "cilantro_params.json")
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
        return self.cluster_metrics["num_spikes"].values

    @property
    def recording_duration_s(self) -> float:
        return len(self.raw_data.data) / self.params["sample_rate"]

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
        params["meta_path"] = params["data_path"].replace(".bin", ".meta")
        params["n_chan"] = params.pop("n_channels_dat")

        params.pop("overwrite")
        self.params = CuratorParams().load(params)
        # in case using old version of marshmallow, convert the struct to dict
        if not isinstance(self.params, dict):
            self.params = self.params._asdict()["data"]

        curation_params = {"curation_params": self.params}
        curation_params["curation_params"][
            "base_metrics_date"
        ] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(
            self.params_file,
            "w",
        ) as f:
            json.dump(curation_params, f, indent=4)

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

        self.times_multi = npx.find_times_multi(
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

        self.mean_wf = npx.calc_mean_wf(
            self.params,
            n_clusters,
            cluster_ids,
            self.times_multi,
            self.raw_data.data,
        )

        # calculate mean waveform for tracking
        npx.calc_mean_wf_split(
            self.params,
            n_clusters,
            cluster_ids,
            self.times_multi,
            self.raw_data.data,
            n_splits=2,
        )

        # load cilantro_metrics.tsv if it exists
        if load_metrics:
            self.cluster_metrics = pd.read_csv(
                metrics_path, sep="\t", index_col="cluster_id"
            )
        else:
            # TODO do not load in all spikes
            self.spikes = npx.extract_all_spikes(
                self.raw_data.data,
                self.times_multi,
                cluster_ids,
                self.params["pre_samples"],
                self.params["post_samples"],
                self.params["max_spikes"],
            )
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
        self.cluster_metrics["num_spikes"] = [
            len(self.times_multi[i]) for i in self.cluster_ids
        ]
        # save as uint32
        self.cluster_metrics["num_spikes"] = self.cluster_metrics["num_spikes"].astype(
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
            slid_rp_viols = npx.calc_sliding_RP_viol(
                self.times_multi,
                self.cluster_ids,
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

        self.cluster_metrics["firing_rate"] = self.cluster_metrics["num_spikes"] / (
            len(self.raw_data.data) / self.params["sample_rate"]
        )

        self.cluster_metrics["firing_rate"] = self.cluster_metrics[
            "firing_rate"
        ].astype("float32")

        # SNR
        snr_path = os.path.join(self.ks_folder, "cluster_SNR_good.tsv")
        noise_path = os.path.join(self.ks_folder, "noise_stds.tsv")
        noise_cutoff_path = os.path.join(self.ks_folder, "noise_cutoff.tsv")
        if overwrite or (
            not os.path.exists(snr_path)
            or not os.path.exists(noise_path)
            or not os.path.exists(noise_cutoff_path)
        ):
            tqdm.write("Calculating SNR...")
            noise = extract_noise(
                self.raw_data.data,
                self.spike_times,
                self.params["pre_samples"],
                self.params["post_samples"],
            )
            meta = npx.read_meta(self.params["meta_path"])
            noise_stds = cp.std(noise, axis=0) * npx.get_bits_to_uV(meta)
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
            # save noise_stds to csv
            noise_stds_df = pd.DataFrame({"noise_stds": stds}, index=self.cluster_ids)
            noise_stds_df.to_csv(
                noise_path,
                sep="\t",
                index=True,
                index_label="cluster_id",
            )

            # noise cutoff
            nc = calculate_noise_cutoff(
                self.spikes,
                self.cluster_metrics["peak"].values,
                self.cluster_ids,
                self.n_clusters,
            )
            # save noise_cutoff to csv
            noise_cutoff_df = pd.DataFrame({"noise_cutoff": nc}, index=self.cluster_ids)
            noise_cutoff_df.to_csv(
                noise_cutoff_path,
                sep="\t",
                index=True,
                index_label="cluster_id",
            )
            tqdm.write("Noise cutoff file saved.")

        else:
            snr_df = pd.read_csv(snr_path, sep="\t", index_col="cluster_id")
            noise_stds_df = pd.read_csv(noise_path, sep="\t", index_col="cluster_id")
            noise_cutoff_df = pd.read_csv(
                noise_cutoff_path, sep="\t", index_col="cluster_id"
            )

        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            snr_df,
            left_index=True,
            right_index=True,
            how="left",
        )
        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            noise_stds_df,
            left_index=True,
            right_index=True,
            how="left",
        )
        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            noise_cutoff_df,
            left_index=True,
            right_index=True,
            how="left",
        )

        self.cluster_metrics["SNR_good"] = self.cluster_metrics["SNR_good"].astype(
            "float32"
        )
        self.cluster_metrics["noise_stds"] = self.cluster_metrics["noise_stds"].astype(
            "float32"
        )
        self.cluster_metrics["noise_cutoff"] = self.cluster_metrics[
            "noise_cutoff"
        ].astype("float32")

        # presence ratio
        presence_ratio_path = os.path.join(self.ks_folder, "presence_ratio.tsv")
        if overwrite or not os.path.exists(presence_ratio_path):
            calc_presence_ratio(
                self.cluster_metrics,
                self.times_multi,
                self.recording_duration_s,
            )
            pr_df = self.cluster_metrics[["presence_ratio"]]
            pr_df.to_csv(
                presence_ratio_path,
                sep="\t",
                index=True,
                index_label="cluster_id",
            )
        else:
            pr_df = pd.read_csv(presence_ratio_path, sep="\t", index_col="cluster_id")
            self.cluster_metrics = pd.merge(
                self.cluster_metrics,
                pr_df,
                left_index=True,
                right_index=True,
                how="left",
            )
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
        if len(merges) == 0:
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
        new_ids = list(merges.keys())
        n_clusters = max(self.n_clusters, np.max(new_ids) + 1)
        # update times_multi from merges
        self.times_multi = npx.find_times_multi(
            self.spike_times,
            self.spike_clusters,
            np.arange(n_clusters),
            self.raw_data.data,
            self.params["pre_samples"],
            self.params["post_samples"],
        )
        cluster_ids = [k for k, v in self.times_multi.items() if len(v) > 0]
        # Reload spikes. TODO: do not load for all of recording
        self.spikes = npx.extract_all_spikes(
            self.raw_data.data,
            self.times_multi,
            cluster_ids,
            self.params["pre_samples"],
            self.params["post_samples"],
            self.params["max_spikes"],
        )
        for new_id, old_ids in merges.items():
            if new_id in self.cluster_metrics.index:
                continue

            old_rows = self.cluster_metrics.loc[old_ids]

            # amplitude
            channel_amplitudes = (
                np.max(self.mean_wf[new_id], 1) - np.min(self.mean_wf[new_id], 1)
            ).flatten()
            new_amplitude = np.max(channel_amplitudes)

            # peak
            peak = np.argmax(channel_amplitudes, axis=0)

            # SNR_good
            snr = np.sum(old_rows["SNR_good"] * old_rows["num_spikes"]) / np.sum(
                old_rows["num_spikes"]
            )

            # slid_RP_viol
            slid_rp_viol = npx.calc_sliding_RP_viol(self.times_multi, [new_id])[new_id]

            # noise_cutoff
            sp_amplitudes = np.ptp(self.spikes[new_id], 1)
            _, nc, _ = noise_cutoff(sp_amplitudes)

            # presence_ratio
            pr = presence_ratio(self.times_multi[new_id], self.recording_duration_s)

            # firing rate
            fr = np.sum(old_rows["num_spikes"]) / (
                len(self.raw_data.data) / self.params["sample_rate"]
            )

            new_row = pd.DataFrame(
                {
                    "label": "good",
                    "label_reason": f"merged from {old_ids}",
                    "num_spikes": old_rows["num_spikes"].sum(),
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
            self.cluster_metrics.loc[old_ids, "label"] = "merged"
            self.cluster_metrics.loc[old_ids, "label_reason"] = f"merged into {new_id}"
            self.cluster_metrics.loc[old_ids, "num_spikes"] = 0
        # save new metrics
        # cluster_SNR_good.tsv
        cluster_SNR_good = self.cluster_metrics[["SNR_good"]]
        cluster_SNR_good.to_csv(
            os.path.join(self.ks_folder, "cluster_SNR_good.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )
        # cluster_RP_conf.tsv
        cluster_RP_conf = self.cluster_metrics[["slid_RP_viol"]]
        cluster_RP_conf.to_csv(
            os.path.join(self.ks_folder, "cluster_RP_conf.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )
        # noise_cutoff.tsv
        nc = self.cluster_metrics[["noise_cutoff"]]
        nc.to_csv(
            os.path.join(self.ks_folder, "noise_cutoff.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )
        # presence_ratio.tsv
        pr = self.cluster_metrics[["presence_ratio"]]
        pr.to_csv(
            os.path.join(self.ks_folder, "presence_ratio.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )

        # cluster_group.tsv
        cluster_labels = self.cluster_metrics[["label", "label_reason"]]
        cluster_labels.to_csv(
            os.path.join(self.ks_folder, "cluster_group.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )

    def auto_curate(self, args: dict = {}) -> None:
        # check if auto_curate_params.json exists and if auto_curate_params are already in it
        if os.path.exists(self.params_file):
            with open(self.params_file) as f:
                auto_curate_params = json.load(f)
                if "auto_curate_params" in auto_curate_params:
                    params = auto_curate_params["auto_curate_params"]
                    if params == args:
                        tqdm.write("Already auto curated with same params.")
                        return
                    else:
                        tqdm.write("Auto curated with different params previously.")
                        return

        tqdm.write("Auto-curating clusters...")
        schema = AutoCurateParams()
        params = schema.load(args)
        # reset labels
        self.cluster_metrics["label"] = "good"
        self.cluster_metrics["label_reason"] = ""

        # mark low-spike units as noise
        if params["min_fr"] is not None:
            low_spike_units = self.cluster_metrics[
                self.cluster_metrics["firing_rate"] < params["min_fr"]
            ].index
            self.cluster_metrics.loc[low_spike_units, "label_reason"] = (
                "low firing rate"
            )
            self.cluster_metrics.loc[low_spike_units, "label"] = "noise"

        # mark low snr units as noise
        if params["min_snr"] is not None:
            low_snr_units = self.cluster_metrics[
                (self.cluster_metrics["SNR_good"] < params["min_snr"])
                & (self.cluster_metrics["label"].isin(params["good_lbls"]))
            ].index
            self.cluster_metrics.loc[low_snr_units, "label_reason"] = "low SNR"
            self.cluster_metrics.loc[low_snr_units, "label"] = "noise"

        # mark units with high RP violations as mua
        if params["max_rp_viol"] is not None:
            high_rp_units = self.cluster_metrics[
                (self.cluster_metrics["slid_RP_viol"] > params["max_rp_viol"])
                & (self.cluster_metrics["label"].isin(params["good_lbls"]))
            ].index
            self.cluster_metrics.loc[high_rp_units, "label_reason"] = (
                "high RP violations"
            )
            self.cluster_metrics.loc[high_rp_units, "label"] = "mua"

        # mark units with too many peaks and troughs as noise
        if params["max_peaks"] is not None:
            high_peaks_units = self.cluster_metrics[
                (self.cluster_metrics["n_peaks"] > params["max_peaks"])
                & (self.cluster_metrics["label"].isin(params["good_lbls"]))
            ].index
            self.cluster_metrics.loc[high_peaks_units, "label_reason"] = (
                "too many peaks"
            )
            self.cluster_metrics.loc[high_peaks_units, "label"] = "noise"

        if params["max_troughs"] is not None:
            high_troughs_units = self.cluster_metrics[
                (self.cluster_metrics["n_troughs"] > params["max_troughs"])
                & (self.cluster_metrics["label"].isin(params["good_lbls"]))
            ].index
            self.cluster_metrics.loc[high_troughs_units, "label_reason"] = (
                "too many troughs"
            )
            self.cluster_metrics.loc[high_troughs_units, "label"] = "noise"

        # mark units with long waveform duration as noise
        if params["max_wf_dur"] is not None:
            long_wf_units = self.cluster_metrics[
                (self.cluster_metrics["wf_dur"] > params["max_wf_dur"])
                & (self.cluster_metrics["label"].isin(params["good_lbls"]))
            ].index
            self.cluster_metrics.loc[long_wf_units, "label_reason"] = "long waveform"
            self.cluster_metrics.loc[long_wf_units, "label"] = "noise"

        # mark units with low spatial decay as noise
        if params["min_spat_decay"] is not None:
            low_spat_decay_units = self.cluster_metrics[
                (self.cluster_metrics["spat_decay"] < params["min_spat_decay"])
                & (self.cluster_metrics["label"].isin(params["good_lbls"]))
            ].index
            self.cluster_metrics.loc[low_spat_decay_units, "label_reason"] = (
                "low spatial decay"
            )
            self.cluster_metrics.loc[low_spat_decay_units, "label"] = "noise"

        self.save_labels()
        # save log of parameters used
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file) as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        else:
            existing_data = {}
        params["curation_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        existing_data.update({"auto_curate_params": params})
        with open(
            self.params_file,
            "w",
        ) as f:
            json.dump(existing_data, f, indent=4)

    def post_merge_curation(self, args: dict = {}) -> None:
        if os.path.exists(self.params_file):
            with open(self.params_file) as f:
                auto_curate_params = json.load(f)
                if "post_merge_params" in auto_curate_params:
                    params = auto_curate_params["post_merge_params"]
                    if params == args:
                        tqdm.write("Already post-merge curated with same params.")
                        return
                    else:
                        tqdm.write(
                            "Post-merge curated with different params previously."
                        )
                        return
        tqdm.write("Post-merge curation...")
        schema = PostMergeCurationParams()
        params = schema.load(args)

        self.update_merged_metrics()

        if params["max_noise_cutoff"] is not None:
            high_amp_units = self.cluster_metrics[
                (self.cluster_metrics["noise_cutoff"] > params["max_noise_cutoff"])
                & (self.cluster_metrics["label"] == "good")
            ].index
            self.cluster_metrics.loc[high_amp_units, "label_reason"] = (
                "high amplitude cutoff"
            )
            self.cluster_metrics.loc[high_amp_units, "label"] = "inc"

        if params["min_pr"] is not None:
            low_pr_units = self.cluster_metrics[
                (self.cluster_metrics["presence_ratio"] < params["min_pr"])
                & (self.cluster_metrics["label"] == "good")
            ].index
            self.cluster_metrics.loc[low_pr_units, "label_reason"] = (
                "low presence ratio"
            )
            self.cluster_metrics.loc[low_pr_units, "label"] = "inc"

        self.save_labels()
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file) as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        else:
            existing_data = {}
        # add date to params
        params["merge_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        existing_data.update({"post_merge_params": params})
        with open(
            self.params_file,
            "w",
        ) as f:
            json.dump(existing_data, f, indent=4)

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

    def save(self):
        tqdm.write("Saving cluster metrics to cilantro_metrics.tsv...")
        # save cluster_metrics in cilantro_metrics.tsv
        self.cluster_metrics.to_csv(
            os.path.join(self.ks_folder, "cilantro_metrics.tsv"),
            sep="\t",
            index=True,
            index_label="cluster_id",
        )

        self.raw_data.close()

    def close(self):
        self.save()
