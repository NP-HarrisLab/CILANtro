import json
import logging
import os
from pathlib import Path

import burst_detector as bd
import cupy as cp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import cilantropy.SGLXMetaToCoords as SGLXMeta
from cilantropy.custom_metrics import (
    calc_noise_cutoff,
    calc_presence_ratio,
    calc_sliding_RP_viol,
    calc_SNR,
    calc_wf_shape_metrics,
    extract_noise,
    noise_cutoff,
    presence_ratio,
)
from cilantropy.params import AutoCurateParams, CuratorParams
from cilantropy.rawdata import RawData
from cilantropy.utils import get_uVPerBit

# from spike_fixer.modules.merge import merge_clusters


logger = logging.getLogger("cilantropy")


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
        std_wf: NDArray
            Standard deviation of waveforms for each cluster. Shape (n_clusters, n_channels, pre_samples + post_samples)
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
        self.std_wf: NDArray
        self.channel_pos: NDArray
        self._calc_metrics(**kwargs)

    @property
    def n_clusters(self) -> int:
        return self.cluster_metrics["cluster_id"].iloc[-1] + 1

    @property
    def cluster_ids(self) -> NDArray:
        return self.cluster_metrics["cluster_id"].values

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
        if type(self.params) != dict:
            self.params = self.params._asdict()["data"]

    def _calc_metrics(self, **kwargs) -> None:
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
        self.spike_times = self.spike_times[
            (self.spike_times >= 0) & (self.spike_times < len(self.raw_data.data))
        ].astype(np.uint32)
        # resave spike_times
        np.save(os.path.join(self.ks_folder, "spike_times.npy"), self.spike_times)

        self.channel_pos = np.load(
            os.path.join(self.ks_folder, "channel_positions.npy")
        )

        # remove cluster_ids that are not in spike_clusters
        cluster_ids = np.unique(self.spike_clusters, return_counts=False)
        n_clusters = np.max(cluster_ids) + 1

        self.times_multi = bd.find_times_multi(
            self.spike_times,
            self.spike_clusters,
            cluster_ids,
            self.raw_data.data,
            self.params["pre_samples"],
            self.params["post_samples"],
        )

        self.mean_wf, self.std_wf, _ = bd.calc_mean_and_std_wf(
            self.params,
            n_clusters,
            cluster_ids,
            self.times_multi,
            self.raw_data.data,
            return_std=False,
            return_spikes=False,
        )

        self.cluster_metrics = pd.DataFrame()
        self.cluster_metrics["cluster_id"] = np.arange(n_clusters)

        self.fill_dataframe()
        logger.info("Calculated metrics")

    def fill_dataframe(self) -> None:
        self.cluster_metrics["spike_times"] = self.times_multi
        self.cluster_metrics["n_spikes"] = self.cluster_metrics["spike_times"].apply(
            len
        )

        # cluster waveform shapes
        wf_path = os.path.join(self.ks_folder, "cluster_wf_shape.tsv")
        if not os.path.exists(wf_path):
            logger.info("Calculating waveform shape metrics...")
            num_peaks, num_troughs, wf_durs, spat_decay = calc_wf_shape_metrics(
                self.mean_wf, self.cluster_ids, self.channel_pos
            )
            wf_df = pd.DataFrame(
                {
                    "cluster_id": self.cluster_ids,
                    "n_peaks": num_peaks,
                    "n_troughs": num_troughs,
                    "wf_dur": wf_durs,
                    "spat_decay": spat_decay,
                }
            )
            wf_df.to_csv(wf_path, sep="\t")
            logger.info("WF file saved.")
        else:
            wf_df = pd.read_csv(wf_path, sep="\t")
            if "spat_decays" in wf_df.columns:
                wf_df.rename(columns={"spat_decays": "spat_decay"}, inplace=True)
                wf_df.to_csv(wf_path, sep="\t", index=False)
            if "num_peaks" in wf_df.columns:
                wf_df.rename(columns={"num_peaks": "n_peaks"}, inplace=True)
                wf_df.to_csv(wf_path, sep="\t", index=False)
            if "num_troughs" in wf_df.columns:
                wf_df.rename(columns={"num_troughs": "n_troughs"}, inplace=True)
                wf_df.to_csv(wf_path, sep="\t", index=False)
            if "amplitude" in wf_df.columns:  # old version of cilantropy saved this
                wf_df.drop(columns=["amplitude"], axis=1, inplace=True)
                wf_df.to_csv(wf_path, sep="\t", index=False)
        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            wf_df,
            on="cluster_id",
            how="left",
        )
        # load J. Colonell's metrics.csv if it exists
        metrics_path = os.path.join(self.ks_folder, "metrics.csv")
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)
        else:
            metrics = pd.DataFrame()

        if "amplitude" not in metrics.columns:
            amplitude = cp.max(self.mean_wf, 2) - cp.min(self.mean_wf, 2)
            # convert to uV
            meta_path = self.params["data_path"].replace(".bin", ".meta")
            meta = SGLXMeta.readMeta(Path(meta_path))

            if "imDatPrb_type" in meta:
                pType = meta["imDatPrb_type"]
                if pType == "0":
                    probe_type = "NP1"
                else:
                    probe_type = "NP" + pType
            else:
                probe_type = "3A"  # 3A probe is default
            amplitude = cp.asnumpy(amplitude) * get_uVPerBit(
                meta, meta_path, probe_type
            )
            self.cluster_metrics["amplitude"] = [
                amplitude[i, :] for i in self.cluster_ids
            ]
        else:
            self.cluster_metrics["amplitude"] = self.cluster_metrics["cluster_id"].map(
                metrics.set_index("cluster_id")["amplitude"]
            )

        if "peak_channel" not in metrics.columns:
            self.cluster_metrics["peak"] = self.cluster_metrics["amplitude"].apply(
                lambda x: np.argmax(x, axis=0)
            )
        else:
            self.cluster_metrics["peak"] = self.cluster_metrics["cluster_id"].map(
                metrics.set_index("cluster_id")["peak_channel"]
            )

        if "firing_rate" not in metrics.columns:
            self.cluster_metrics["firing_rate"] = self.cluster_metrics["n_spikes"] / (
                len(self.raw_data.data) / self.params["sample_rate"]
            )
        else:
            self.cluster_metrics["firing_rate"] = self.cluster_metrics[
                "cluster_id"
            ].map(metrics.set_index("cluster_id")["firing_rate"])

        if "snr" not in metrics.columns:
            # SNR
            snr_path = os.path.join(self.ks_folder, "cluster_SNR_good.tsv")
            if not os.path.exists(snr_path):
                logger.info("Calculating background standard deviation...")
                noise = extract_noise(
                    self.raw_data.data,
                    self.spike_times,
                    self.params["pre_samples"],
                    self.params["post_samples"],
                    n_chan=self.n_channels,
                )
                noise_stds = np.std(noise, axis=1)
                snrs = calc_SNR(self.mean_wf, noise_stds, self.cluster_ids)
                snr_df = pd.DataFrame(
                    {"cluster_id": self.cluster_ids, "SNR_good": snrs}
                )
                snr_df.to_csv(
                    os.path.join(self.ks_folder, "cluster_SNR_good.tsv"), sep="\t"
                )
                logger.info("SNR file saved.")
            else:
                snr_df = pd.read_csv(snr_path, sep="\t")

            if "SNR_good" not in self.cluster_metrics.columns:
                self.cluster_metrics = pd.merge(
                    self.cluster_metrics, snr_df, on="cluster_id", how="outer"
                )
        else:
            self.cluster_metrics["SNR_good"] = self.cluster_metrics["cluster_id"].map(
                metrics.set_index("cluster_id")["snr"]
            )
            self.cluster_metrics["SNR_good"] = metrics["snr"]

        # refractory period violations
        if "slideingRP_viol" not in metrics.columns:
            cluster_rp_path = os.path.join(self.ks_folder, "cluster_RP_conf.tsv")
            if not os.path.exists(cluster_rp_path):
                logger.info("Calculating refractory period violations...")
                slid_rp_viols = calc_sliding_RP_viol(
                    self.times_multi, self.cluster_ids, self.n_clusters
                )
                srv_df = pd.DataFrame(
                    {"cluster_id": self.cluster_ids, "slid_RP_viol": slid_rp_viols}
                )
                srv_df.to_csv(cluster_rp_path, sep="\t")
                print("RP file saved.")
            else:
                srv_df = pd.read_csv(cluster_rp_path, sep="\t")

            if "slid_RP_viol" not in self.cluster_metrics.columns:
                self.cluster_metrics = pd.merge(
                    self.cluster_metrics, srv_df, on="cluster_id", how="outer"
                )
        else:
            self.cluster_metrics["slid_rp_viol"] = self.cluster_metrics[
                "cluster_id"
            ].map(metrics.set_index("cluster_id")["slideingRP_viol"])

        # noise cutoff
        if "nongauss_noise_cutoff" not in metrics.columns:
            logger.info("Calculating noise cutoff...")
            calc_noise_cutoff(self.cluster_metrics)
        else:
            self.cluster_metrics["noise_cutoff"] = self.cluster_metrics[
                "cluster_id"
            ].map(metrics.set_index("cluster_id")["nongauss_noise_cutoff"])

        # presence ratio
        if "presence_ratio" not in metrics.columns:
            logger.info("Calculating presence ratio...")
            calc_presence_ratio(self.cluster_metrics)
        else:
            self.cluster_metrics["presence_ratio"] = self.cluster_metrics[
                "cluster_id"
            ].map(metrics.set_index("cluster_id")["presence_ratio"])

        # if merged clusters after ecephys_spike_sorting, update values for new clusters
        if (
            not metrics.empty
            and metrics.iloc[-1]["cluster_id"]
            < self.cluster_metrics.iloc[-1]["cluster_id"]
            and os.path.exists(
                os.path.join(self.ks_folder, "automerge", "new2old.json")
            )
        ):
            self.update_merged_metrics()

    def update_merged_metrics(self) -> None:
        # update any cluster_metrics that were merged
        with open(os.path.join(self.ks_folder, "automerge", "new2old.json")) as f:
            new2old = json.load(f)
            merges = {int(k): v for k, v in sorted(new2old.items())}
        for new_id, old_ids in merges.items():
            old_rows = self.cluster_metrics[
                self.cluster_metrics["cluster_id"].isin(old_ids)
            ]

            # amplitude
            try:
                new_mean_wf = self.mean_wf[new_id]
            except IndexError:
                # weighted mean of old waveforms
                new_mean_wf = np.sum(
                    [self.mean_wf[i] * self.counts[i] for i in old_ids], axis=0
                ) / np.sum(self.counts[old_ids])
                while len(self.mean_wf) < new_id:
                    self.mean_wf = np.vstack([self.mean_wf, np.zeros_like(new_mean_wf)])
                self.mean_wf = np.vstack([self.mean_wf, new_mean_wf])
            meta_path = self.params["data_path"].replace(".bin", ".meta")
            meta = SGLXMeta.readMeta(Path(meta_path))

            if "imDatPrb_type" in meta:
                pType = meta["imDatPrb_type"]
                if pType == "0":
                    probe_type = "NP1"
                else:
                    probe_type = "NP" + pType
            else:
                probe_type = "3A"  # 3A probe is default
            new_amplitude = np.max(new_mean_wf, 2) - np.min(
                new_mean_wf, 2
            ) * get_uVPerBit(meta, meta_path, probe_type)
            self.cluster_metrics.loc[new_id, "amplitude"] = new_amplitude

            # peak
            self.cluster_metrics["peak"] = np.argmax(new_amplitude, axis=0)

            # SNR_good
            self.cluster_metrics.loc[new_id, "SNR_good"] = np.sum(
                old_rows["SNR_good"] * old_rows["n_spikes"]
            ) / np.sum(old_rows["n_spikes"])

            # slid_RP_viol
            self.cluster_metrics.loc[new_id, "slid_rp_viol"] = calc_sliding_RP_viol(
                self.times_multi, [new_id], self.n_clusters
            )

            # noise_cutoff
            self.cluster_metrics.loc[new_id, "noise_cutoff"] = noise_cutoff(
                new_amplitude
            )

            # presence_ratio
            self.cluster_metrics.loc[new_id, "presence_ratio"] = presence_ratio(
                self.cluster_metrics.loc[new_id, "spike_times"]
            )

    def auto_curate(self, args: dict = {}) -> None:
        schema = AutoCurateParams()
        params = schema.load(args)

        # mark low-spike units as noise
        low_spike_units = self.cluster_metrics[
            self.cluster_metrics["firing_rate"] < params["min_fr"]
        ].index
        self.cluster_metrics.loc[low_spike_units, "label_reason"] = "low firing rate"
        self.cluster_metrics.loc[low_spike_units, "label"] = "noise"

        # mark low snr units as noise
        low_snr_units = self.cluster_metrics[
            self.cluster_metrics["SNR_good"] < params["min_snr"]
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
            self.cluster_metrics["n_peaks"] > params["max_peaks"]
        ].index
        self.cluster_metrics.loc[high_peaks_units, "label_reason"] = "too many peaks"
        self.cluster_metrics.loc[high_peaks_units, "label"] = "noise"

        high_troughs_units = self.cluster_metrics[
            self.cluster_metrics["n_troughs"] > params["max_troughs"]
        ].index
        self.cluster_metrics.loc[high_troughs_units, "label_reason"] = (
            "too many troughs"
        )
        self.cluster_metrics.loc[high_troughs_units, "label"] = "noise"

        # mark units with long waveform duration as noise
        long_wf_units = self.cluster_metrics[
            self.cluster_metrics["wf_dur"] > params["max_wf_dur"]
        ].index
        self.cluster_metrics.loc[long_wf_units, "label_reason"] = "long waveform"
        self.cluster_metrics.loc[long_wf_units, "label"] = "noise"

        # mark units with low spatial decay as noise
        low_spat_decay_units = self.cluster_metrics[
            self.cluster_metrics["spat_decay"] < params["min_spat_decay"]
        ].index
        self.cluster_metrics.loc[low_spat_decay_units, "label_reason"] = (
            "low spatial decay"
        )
        self.cluster_metrics.loc[low_spat_decay_units, "label"] = "noise"

        if params["save"]:
            self.save_data()

    def post_merge_curation(self, args: dict = {}) -> None:
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

        if params["save"]:
            self.save_data()

    def save_data(self) -> None:
        # save cluster_metrics in cilantro_metrics.tsv
        self.cluster_metrics.to_csv(
            os.path.join(self.ks_folder, "cilantro_metrics.tsv"),
            sep="\t",
            index=False,
        )

        # save new cluster_group.tsv
        cluster_labels = self.cluster_metrics[["cluster_id", "label"]]
        cluster_labels.to_csv(
            os.path.join(self.ks_folder, "cluster_group.tsv"),
            sep="\t",
            index=False,
        )

        # save mean_wf
        np.save(os.path.join(self.ks_folder, "mean_waveforms.npy"), self.mean_wf)

        # save spike_clusters
        np.save(
            os.path.join(self.ks_folder, "spike_clusters.npy"),
            self.calc_spike_clusters(),
        )

    def calc_spike_clusters(self) -> NDArray[np.uint32]:  # but says int32 in doc
        # if new2old.json exists, update spike_clusters
        if os.path.exists(os.path.join(self.ks_folder, "automerge", "new2old.json")):
            with open(os.path.join(self.ks_folder, "automerge", "new2old.json")) as f:
                new2old = json.load(f)
                merges = {int(k): v for k, v in sorted(new2old.items())}
            for new_id, old_ids in merges.items():
                self.spike_clusters[np.isin(self.spike_clusters, old_ids)] = new_id
        return self.spike_clusters
