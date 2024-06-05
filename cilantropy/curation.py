import logging
import os

import numpy as np
import pandas as pd
from marshmallow import Schema, fields
from tqdm import tqdm

from cilantropy.rawdata import RawData
from cilantropy.utils import extract_spikes, find_times_multi

logger = logging.getLogger("cilantropy")


class CuratorParams(Schema):
    # TODO get rid of RemovedInMarshmallow4Warning
    """
    Class for curation parameters
    """

    ks_folder = fields.Str(
        required=True,
        description="Path to kilosort folder.",
        validate=lambda x: os.path.exists(x),
    )
    data_path = fields.Str(
        required=True,
        description="Filepath to recording binary.",
        validate=lambda x: os.path.exists(x),
    )
    dtype = fields.Str(required=True, description="Data type of recording binary.")
    offset = fields.Int(required=True, description="Offset of recording binary.")
    sample_rate = fields.Float(
        required=True,
        description="Sampling rate of recording.",
        validate=lambda x: x > 0,
    )
    hp_filtered = fields.Bool(required=True, description="High-pass filtered or not.")
    n_channels = fields.Int(
        required=True,
        description="Number of channels in recording binary.",
        validate=lambda x: x > 0,
    )
    pre_samples = fields.Int(
        missing=20,
        description="Number of samples to extract before the peak of the spike.",
    )
    post_samples = fields.Int(
        missing=62,
        description="Number of samples to extract after the peak of the spike.",
    )
    max_spikes = fields.Int(
        required=False,
        missing=1000,
        description="Maximum number of spikes per cluster used to calculate mean waveforms and cross projections. -1 uses all spikes.",
    )
    # simularity_type = fields.Str(
    #     required=False,
    #     default="ae",
    #     description="Type of similarity metric to use, must be either 'ae' (for autoencoder similarity) or 'mean' (for mean similarity)",
    # )
    # check_jitter = fields.Bool(
    #     required=False,
    #     default=False,
    #     description="Whether mean similarity calculations should check for time shifts between waveforms.",
    # )
    # jitter_amount = fields.Int(
    #     required=False,
    #     default=5,
    #     description="Number of samples to search in each direction for time shifts between waveforms.",
    # )
    # similarity_threshold = fields.Float(
    #     required=False,
    #     default=0.4,
    #     description="Similarity threshold for a cluster pair to undergo further stages.",
    # )
    # min_spikes = fields.Int(
    #     required=False,
    #     default=100,
    #     description="Minimum number of spikes in a cluster to undergo further stages.",
    # )


class Curator(object):
    """
    Class for curating spike sorted data.

    Attributes:
        cluster_metrics: pd.DataFrame
            Cluster metrics.
        spike_clusters: np.ndarray
            cluster_id for each spike. Shape (n_spikes,)
    """

    def __init__(self, ks_folder: str) -> None:
        self.ks_folder = ks_folder
        self._load_ksdata()
        self._load_metrics()

    @property
    def n_clusters(self) -> int:
        return self.cluster_metrics.shape[0]

    @property
    def cluster_ids(self) -> np.ndarray:
        return np.unique(self.spike_clusters)

    def _load_ksdata(self) -> None:
        self._load_params()
        self.raw_data = RawData(self.params["data_path"], self.params["n_channels"])

        # load spike_clusters
        try:
            self.spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_clusters.npy")
            ).flatten()
        except FileNotFoundError:
            self.spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_templates.npy")
            ).flatten()

        # load spike_times
        self.spike_times = np.load(
            os.path.join(self.ks_folder, "spike_times.npy")
        ).flatten()

    def _load_params(self):
        params = {"ks_folder": self.ks_folder}

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
        params["n_channels"] = params.pop("n_channels_dat")

        self.params = CuratorParams().load(params)

    def _load_metrics(self) -> None:
        # check if cilantro_metrics.tsv exists
        metrics_path = os.path.join(self.ks_folder, "cilantro_metrics.tsv")
        if os.path.exists(metrics_path):
            self.cluster_metrics = pd.read_csv(metrics_path, sep="\t")
        else:
            # if not, load metrics from individual files
            self.cluster_metrics = pd.DataFrame()

            seps = {".csv": ",", ".tsv": "\t"}
            exclude = ["waveform_metrics", "metrics"]

            for f in os.listdir(self.ks_folder):
                if (f.endswith(".csv") or f.endswith(".tsv")) and (
                    f[:-4] not in exclude
                ):
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
            cluster_ids, counts = np.unique(self.spike_clusters, return_counts=True)

            self.cluster_metrics = self.cluster_metrics[
                self.cluster_metrics["cluster_id"].isin(cluster_ids)
            ]

            # Occassionally n_spikes is not correct, so override it with the actual count
            self.cluster_metrics["n_spikes"] = counts

            # add spike_times
            self.cluster_metrics["spike_times"] = self.cluster_metrics[
                "cluster_id"
            ].map(lambda x: self.spike_times[self.spike_clusters == x])

            self.mean_wf = self._calc_mean_wf()
            # TODO ensure order of cluster_ids is correct?
            self.cluster_metrics["peaks"] = np.argmax(
                np.max(self.mean_wf, 2) - np.min(self.mean_wf, 2), 1
            )
            print("done")
            # save metrics to file
            # self.cluster_metrics.to_csv(metrics_path, sep="\t", index=False)

    def _calc_mean_wf(self) -> np.ndarray:
        """
        Calculate mean waveforms for each cluster. Need to have loaded some metrics.

        Returns:
            np.ndarray: Mean waveforms for each cluster. Shape (n_clusters, n_channels, pre_samples + post_samples)
        """
        mean_wf_path = os.path.join(self.ks_folder, "mean_waveforms.npy")
        if os.path.exists(mean_wf_path):
            return np.load(mean_wf_path)

        logger.info("Calculating mean waveforms...")

        mean_wf = np.zeros(
            (
                self.n_clusters,
                self.params["n_channels"],
                self.params["pre_samples"] + self.params["post_samples"],
            )
        )
        for i in tqdm(self.cluster_ids, desc="Calculating mean waveforms"):
            spikes = self._extract_spikes(i)
            mean_wf[i, :, :] = np.mean(spikes, axis=0)

        np.save(mean_wf_path, mean_wf)

        return mean_wf

    def _extract_spikes(self, clust_id: int) -> np.ndarray:
        times: np.ndarray = self.cluster_metrics["spike_times"][clust_id]

        # ignore spikes cut off by ends of recording
        while (times[0] - self.params["pre_samples"]) < 0:
            times = times[1:]
        while (times[-1] + self.params["post_samples"]) >= self.raw_data.shape[0]:
            times = times[:-1]

        # randomly pick spikes if cluster has too many
        if (self.params["max_spikes"] != -1) and (
            self.params["max_spikes"] < times.shape[0]
        ):
            np.random.shuffle(times)
            times = times[: self.params["max_spikes"]]
        spikes = np.zeros(
            (
                times.shape[0],
                self.params["n_channels"],
                self.params["pre_samples"] + self.params["post_samples"],
            )
        )
        for i in range(times.shape[0]):
            spikes[i, :, :] = self.raw_data.data[
                times[i]
                - self.params["pre_samples"] : times[i]
                + self.params["post_samples"],
                :,
            ].T
        return spikes

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

        # Add label_reason column and set to NA
        self.cluster_metrics["label_reason"] = pd.NA  # TODO or empty string better?

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
        self.cluster_metrics.drop(
            columns=["peak_channel"], axis=1, inplace=True, errors="ignore"
        )  # we will calculate this later

        # sort by cluster_id
        self.cluster_metrics.sort_values("cluster_id", inplace=True)


if __name__ == "__main__":
    # ks_folder = r"D:\040224_INS2_4_DT3_rec_g0\catgt_040224_INS2_4_DT3_rec_g0\040224_INS2_4_DT3_rec_g0_imec0\imec0_ks2"
    ks_folder = r"D:\040224_INS2_4_DT3_rec_g0\040224_INS2_4_DT3_rec_g0_imec0"
    curator = Curator(ks_folder)
    # curator.template_similarity(0)
    # curator.merge([0, 1])
