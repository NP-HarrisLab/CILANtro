import itertools
import json
import logging
import os

import burst_detector as bd
import cupy as cp
import numpy as np
import pandas as pd
from burst_detector import (
    calc_sliding_RP_viol,
    calc_SNR,
    calc_wf_shape_metrics,
    extract_noise,
)
from numpy.typing import NDArray

from cilantropy.params import AutoCurateParams, CuratorParams
from cilantropy.rawdata import RawData

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
        self.params: CuratorParams
        self.n_clusters: int
        self.cluster_ids: NDArray
        self.raw_data: RawData
        self.cluster_metrics: pd.DataFrame
        self.mean_wf: NDArray
        self.std_wf: NDArray
        self.channel_pos: NDArray
        self.merges: dict[int, list[int]] = None
        self._calc_metrics(**kwargs)

    @property
    def n_clusters(self) -> int:
        return self.cluster_metrics.shape[0]

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
        # load params from params.py
        params_file = os.path.join(self.ks_folder, "params.py")
        with open(params_file, "r") as f:
            for line in f:
                elem = line.split(sep="=")
                params[elem[0].strip()] = eval(elem[1].strip())

        # update keynames
        params["data_path"] = params.pop("dat_path")

        # change dat_path to absolute filepath if not already
        if not os.path.isabs(params["data_path"]):
            params["data_path"] = os.path.abspath(
                os.path.join(self.ks_folder, params["data_path"])
            )
        params["n_chan"] = params.pop("n_channels_dat")

        self.params = CuratorParams().load(params)

    def _calc_metrics(self, **kwargs) -> None:
        # check if cilantro_metrics.tsv exists
        metrics_path = os.path.join(self.ks_folder, "cilantro_metrics.tsv")
        params_path = os.path.join(self.ks_folder, "cilantro_params.json")

        # if os.path.exists(metrics_path) and os.path.exists(params_path): #  TODO need ot load other data too...
        #     self.cluster_metrics = pd.read_csv(metrics_path, sep="\t")
        #     try:
        #         with open(params_path, "r") as f:
        #             params = json.load(f)
        #         self.params = CuratorParams().load(params)
        #         return
        #     except json.JSONDecodeError:
        #         print("Error loading parameters from json file. Recalculating...")

        # if not, load metrics from individual files
        self._load_params(**kwargs)
        self.raw_data = RawData(self.params["data_path"], self.params["n_chan"])
        # load spike_clusters
        try:
            spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_clusters.npy")
            ).flatten()
        except FileNotFoundError:
            spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_templates.npy")
            ).flatten()
        self.spike_times = np.load(
            os.path.join(self.ks_folder, "spike_times.npy")
        ).flatten()
        self.channel_pos = np.load(
            os.path.join(self.ks_folder, "channel_positions.npy")
        )

        self.cluster_metrics = pd.DataFrame()

        seps = {".csv": ",", ".tsv": "\t"}
        exclude = ["waveform_metrics", "metrics"]

        for f in os.listdir(self.ks_folder):
            if (f.endswith(".csv") or f.endswith(".tsv")) and (f[:-4] not in exclude):
                df = pd.read_csv(os.path.join(self.ks_folder, f), sep=seps[f[-4:]])

                if "cluster_id" in df.columns:
                    new_feat = [
                        col
                        for col in df.columns
                        if col not in self.cluster_metrics.columns
                    ]
                    self.cluster_metrics = (
                        df
                        if self.cluster_metrics.empty
                        else pd.merge(
                            self.cluster_metrics,
                            df[["cluster_id"] + new_feat],
                            on="cluster_id",
                            how="outer",
                        )
                    )

        self._cleanup_metrics()  # cleanup column names and remove unnecessary ones

        # remove cluster_ids that are not in spike_clusters
        cluster_ids, counts = np.unique(spike_clusters, return_counts=True)
        n_spikes = np.zeros(self.n_clusters)
        n_spikes[cluster_ids] = counts

        # Occassionally n_spikes is not correct, so override it with the actual count
        self.cluster_metrics["n_spikes"] = n_spikes

        # not all clusters had spikes, update NaN to 0
        self.cluster_metrics["n_spikes"] = self.cluster_metrics["n_spikes"].fillna(0)
        # convert to int
        self.cluster_metrics["n_spikes"] = self.cluster_metrics["n_spikes"].astype(int)

        # add spike_times
        self.cluster_metrics["spike_times"] = self.cluster_metrics["cluster_id"].map(
            lambda x: self.spike_times[spike_clusters == x]
        )

        self.times_multi = bd.find_times_multi(
            self.spike_times,
            spike_clusters,
            np.arange(self.n_clusters),
            self.raw_data.data,
            self.params["pre_samples"],
            self.params["post_samples"],
        )
        self.mean_wf, self.std_wf, _ = bd.calc_mean_and_std_wf(
            self.params,
            self.n_clusters,
            cluster_ids,
            self.times_multi,
            self.raw_data.data,
            return_spikes=False,
        )

        print("LOADED")
        # self.cluster_metrics["mean_wf"] = [self.mean_wf[i, :, :] for i in self.cluster_ids]
        amplitude = cp.max(self.mean_wf, 2) - cp.min(self.mean_wf, 2)
        self.cluster_metrics["amplitude"] = [amplitude[i, :] for i in self.cluster_ids]
        self.cluster_metrics["peak"] = self.cluster_metrics["amplitude"].apply(
            lambda x: np.argmax(x, axis=0)
        )

        self._calculate_additional_metrics()
        logger.info("Calculated metrics")
        # save metrics and parameters to file
        self.cluster_metrics.to_csv(metrics_path, sep="\t", index=False)
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=4)

    def _cleanup_metrics(self) -> None:
        # remove duplicate columns
        self.cluster_metrics = self.cluster_metrics.loc[
            :, ~self.cluster_metrics.columns.duplicated()
        ]

        # rename template peak channel and save waveform peak channel
        if "ch" in self.cluster_metrics.columns:
            self.cluster_metrics.rename(columns={"ch": "template_peak"}, inplace=True)
            self.cluster_metrics["template_peak"] = pd.to_numeric(
                self.cluster_metrics["template_peak"],
                errors="coerce",
                downcast="integer",
            )

        # Add label_reason column
        self.cluster_metrics["label_reason"] = ""

        # rename group to label if it exists or use KSLabel
        if "group" not in self.cluster_metrics.columns:
            self.cluster_metrics["label"] = self.cluster_metrics["KSLabel"]
        else:
            self.cluster_metrics.rename(columns={"group": "label"}, inplace=True)

        # remove unnecessary columns
        self.cluster_metrics.drop(columns=["fr"], axis=1, inplace=True, errors="ignore")
        self.cluster_metrics.drop(
            columns=["amplitude", "amp", "Amplitude"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        self.cluster_metrics = self.cluster_metrics.loc[
            :, ~self.cluster_metrics.columns.str.contains("Unnamed")
        ]  # remove unnamed columns
        self.cluster_metrics.drop(columns=["ContamPct"], axis=1, inplace=True)
        self.cluster_metrics.drop(
            columns=["peak_channel"], axis=1, inplace=True, errors="ignore"
        )  # we will calculate this later

        # sort by cluster_id
        self.cluster_metrics.sort_values("cluster_id", inplace=True)

    def _calculate_additional_metrics(self) -> None:
        # TODO: don't save csv files, just save to dataframe
        # SNR
        snr_path = os.path.join(self.ks_folder, "cluster_SNR_good.tsv")
        if not os.path.exists(snr_path):
            logger.info("Calculating background standard deviation...")
            noise = extract_noise(
                self.raw_data.data, self.spike_times, 20, 62, n_chan=self.n_channels
            )
            noise_stds = np.std(noise, axis=1)
            snrs = calc_SNR(self.mean_wf, noise_stds, self.cluster_ids)
            snr_df = pd.DataFrame({"cluster_id": self.cluster_ids, "SNR_good": snrs})
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

        # refractory period violations
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

        # cluster waveform shapes
        wf_path = os.path.join(self.ks_folder, "cluster_wf_shape.tsv")
        if not os.path.exists(wf_path):
            logger.info("Calculating waveform shape metrics...")
            num_peaks, num_troughs, wf_durs, spat_decay, amplitude = (
                calc_wf_shape_metrics(self.mean_wf, self.cluster_ids, self.channel_pos)
            )
            wf_df = pd.DataFrame(
                {
                    "cluster_id": self.cluster_ids,
                    "n_peaks": num_peaks,
                    "n_troughs": num_troughs,
                    "wf_dur": wf_durs,
                    "spat_decay": spat_decay,
                    "amplitude": amplitude,
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

        new_columns = [
            col for col in wf_df.columns if col not in self.cluster_metrics.columns
        ]  # some of these columns may overlap with other data in table already
        self.cluster_metrics = pd.merge(
            self.cluster_metrics,
            wf_df[["cluster_id"] + new_columns],
            on="cluster_id",
            how="left",
        )

        # in case any new unnamed clusters
        self.cluster_metrics = self.cluster_metrics.loc[
            :, ~self.cluster_metrics.columns.str.contains("Unnamed")
        ]  # remove unnamed columns

    def auto_curate(self, save_labels=True, **kwargs) -> None:
        schema = AutoCurateParams()
        params = schema.load(kwargs)

        # mark low-spike units as noise
        low_spike_units = self.cluster_metrics[
            self.cluster_metrics["n_spikes"] < params["min_spikes"]
        ].index
        self.cluster_metrics.loc[low_spike_units, "label_reason"] = "low spike count"
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

        if save_labels:
            # overwrite cluster_group.tsv with new labels
            cluster_labels = self.cluster_metrics[["cluster_id", "label"]]
            cluster_labels.to_csv(
                os.path.join(self.ks_folder, "cluster_group.tsv"),
                sep="\t",
                index=False,
            )
            print("Labels saved.")

    def auto_merge(self, **kwargs) -> None:
        bd.run.main(kwargs)

    def auto_accept_merges(self) -> None:
        print("Auto Accepting Merges")
        # merge suggested clusters
        new2old = os.path.join(self.ks_folder, "automerge", "new2old.json")
        with open(new2old, "r") as f:
            self.merges = json.load(f)
            self.merges = {int(k): v for k, v in sorted(self.merges.items())}

        for new_id, old_ids in self.merges.items():
            self.merge_clusters(old_ids, new_id)

    def merge_clusters(self, cluster_ids: list[int], new_id) -> int:
        # TODO change old_row data to 0's?
        # TODO current code breaks if new id isnt next in line
        """
        Merge clusters together into a new cluster.

        Args:
            cluster_ids (list[int]): List of cluster_ids to merge.
            new_id (int): New cluster_id.

        Returns:
            int: New cluster_id.
        """
        old_rows = self.cluster_metrics[
            self.cluster_metrics["cluster_id"].isin(cluster_ids)
        ]

        # calculate new mean_wf and std_wf as weighted average
        weighted_mean_wf = np.sum(
            self.mean_wf[cluster_ids, :, :]
            * (
                old_rows["n_spikes"].values[:, np.newaxis, np.newaxis]
                / np.sum(old_rows["n_spikes"])
            ),
            axis=0,
        )

        weighted_std_wf = np.sum(
            self.std_wf[cluster_ids, :, :]
            * (
                old_rows["n_spikes"].values[:, np.newaxis, np.newaxis]
                / np.sum(old_rows["n_spikes"])
            ),
            axis=0,
        )

        new_mean_wf = np.zeros(
            (
                new_id + 1,
                self.n_channels,
                self.params["pre_samples"] + self.params["post_samples"],
            )
        )
        new_std_wf = np.zeros(
            (
                new_id + 1,
                self.n_channels,
                self.params["pre_samples"] + self.params["post_samples"],
            )
        )
        old_rows_i = min(new_id, self.mean_wf.shape[0])
        new_mean_wf[:old_rows_i, :, :] = self.mean_wf[
            :old_rows_i, :, :
        ]  # If get partial save
        new_mean_wf[new_id, :, :] = weighted_mean_wf
        new_std_wf[:old_rows_i, :, :] = self.std_wf[
            :old_rows_i, :, :
        ]  # If get partial save
        new_std_wf[new_id, :, :] = weighted_std_wf

        self.mean_wf = new_mean_wf
        self.std_wf = new_std_wf

        # calculate new metrics
        new_spike_times = np.array(
            list(itertools.chain.from_iterable(old_rows["spike_times"]))
        )
        self.times_multi.append(new_spike_times)
        new_slid_rp_viol = calc_sliding_RP_viol(
            [new_spike_times], clust_ids=[0], n_clust=1
        )[0]

        # calculate snr as weighted average by number of spikes
        new_snr = np.sum(old_rows["SNR_good"] * old_rows["n_spikes"]) / np.sum(
            old_rows["n_spikes"]
        )

        # number of peaks and troguhs should be the same
        n_peaks, n_troughs, wf_dur, spat_decay = calc_wf_shape_metrics(
            self.mean_wf, [new_id], self.channel_pos
        )
        new_n_peaks = n_peaks[new_id]
        new_n_troughs = n_troughs[new_id]
        new_wf_dur = wf_dur[new_id]
        new_spat_decay = spat_decay[new_id]

        # amplitude is the max-min of the mean waveform
        amplitude = np.max(self.mean_wf, 2) - np.min(self.mean_wf, 2)
        peak = np.argmax(amplitude, axis=1)

        new_amplitude = amplitude[new_id]
        new_peak = peak[new_id]

        # get new cluster id and create new row
        new_row = pd.DataFrame(
            {
                "cluster_id": new_id,
                "label": old_rows["label"].mode().values[0],  # TODO,
                "label_reason": old_rows["label_reason"].mode().values[0],  # TODO,
                "KSLabel": "",
                "n_spikes": old_rows["n_spikes"].sum(),
                "spike_times": [new_spike_times],
                "slid_rp_viol": new_slid_rp_viol,
                "SNR_good": new_snr,
                "n_peaks": new_n_peaks,
                "n_troughs": new_n_troughs,
                "wf_dur": new_wf_dur,
                "spat_decay": new_spat_decay,
                "amplitude": [new_amplitude],
                "peak": new_peak,
            }
        )

        # add new row to cluster_metrics
        self.cluster_metrics = pd.concat(
            [self.cluster_metrics, new_row], ignore_index=True
        )
        # update old rows to merged
        self.cluster_metrics.loc[
            self.cluster_metrics["cluster_id"].isin(cluster_ids), "label"
        ] = "merged"
        self.cluster_metrics.loc[
            self.cluster_metrics["cluster_id"].isin(cluster_ids), "label_reason"
        ] = f"merged into cluster_id {new_id}"

    def save_data(self) -> None:
        # TODO overwrite any other files?
        # save cluster_metrics in cilantro_metrics.tsv
        self.cluster_metrics.to_csv(
            os.path.join(self.ks_folder, "cilantro_metrics.tsv"),
            sep="\t",
            index=False,
        )
        # save cluster_Amplitude.tsv
        amp_df = self.cluster_metrics[["cluster_id", "amplitude"]]
        amp_df.to_csv(
            os.path.join(self.ks_folder, "cluster_Amplitude.tsv"), sep="\t", index=False
        )
        # TODO save cluster_ContamPct.tsv

        # save new cluster_group.tsv
        cluster_labels = self.cluster_metrics[["cluster_id", "label"]]
        cluster_labels.to_csv(
            os.path.join(self.ks_folder, "cluster_group.tsv"),
            sep="\t",
            index=False,
        )
        # save new cluster_RP_conf.tsv
        rp_df = self.cluster_metrics[["cluster_id", "slid_RP_viol"]]
        rp_df.to_csv(
            os.path.join(self.ks_folder, "cluster_RP_conf.tsv"), sep="\t", index=False
        )

        # save new cluster_SNR_good.tsv
        snr_df = self.cluster_metrics[["cluster_id", "SNR_good"]]
        snr_df.to_csv(
            os.path.join(self.ks_folder, "cluster_SNR_good.tsv"), sep="\t", index=False
        )
        # save in cluster_snr.npy
        np.save(
            os.path.join(self.ks_folder, "cluster_snr.npy"), snr_df["SNR_good"].values
        )

        # save new cluster_wf_shape.tsv
        wf_df = self.cluster_metrics[
            ["cluster_id", "n_peaks", "n_troughs", "wf_dur", "spat_decay", "amplitude"]
        ]
        wf_df.to_csv(
            os.path.join(self.ks_folder, "cluster_wf_shape.tsv"), sep="\t", index=False
        )

        # save new mean_wf and std_wf
        np.save(os.path.join(self.ks_folder, "mean_waveforms.npy"), self.mean_wf)
        np.save(os.path.join(self.ks_folder, "std_waveforms.npy"), self.std_wf)

        # save new spike_clusters.npy
        np.save(
            os.path.join(self.ks_folder, "spike_clusters.npy"),
            self.calc_spike_clusters(),
        )

    def calc_spike_clusters(self) -> NDArray[np.uint32]:  # but says int32 in doc
        spikes_clusters = np.load(os.path.join(self.ks_folder, "spike_clusters.npy")).flatten()
        for new_id, old_ids in self.merges.items():
            spikes_clusters[np.isin(spikes_clusters, old_ids)] = new_id
        return spikes_clusters
