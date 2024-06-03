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
        cluster_groups: pd.DataFrame
            Cluster groups.
        cluster_metrics: pd.DataFrame
            Cluster metrics.
        spike_clusters: np.ndarray
            cluster_id for each spike. Shape (n_spikes,)

        ---
        mean_wf: np.ndarray
            Mean waveforms for each cluster. Shape (n_clusters_orig, n_channels, pre_samples + post_samples)

    """

    def __init__(self, ks_folder: str) -> None:
        self.ks_folder = ks_folder
        self._load_params()
        self._load_ks_files()
        self._initialize()

    @property
    def n_clusters(self):
        return len(self.cluster_ids)

    def get_new_id(self):
        return max(self.cluster_ids) + 1

    def template_similarity(self, cluster_id: int) -> np.ndarray:
        """
        Calculate the similarity of the given cluster to all other clusters.

        Args:
            cluster_id (int): Cluster ID to compare to all other clusters.

        Returns:
            np.ndarray: Similarity of the given cluster to all other clusters. Shape (n_clusters,)
        """
        # return pre-calculated values if they exist. TODO recalculate if other clusters have merged?
        if self.cluster_template_similarity[cluster_id, 0] != -1:
            return self.cluster_template_similarity[cluster_id, :]

        # calculate similarity
        sims = np.max(
            self.similar_templates[self.cluster_templates[cluster_id], :], axis=0
        )  # max similarity of cluster to each template

        def sim_ij(cj):
            if cj not in self.counts:
                return 0
            if cj < self.n_templates:
                return sims[cj]
            return np.max(
                sims[self.cluster_templates[cj]]
            )  # max similarity of cluster to each template

        for j in range(self.n_clusters):
            self.cluster_template_similarity[cluster_id, j] = sim_ij(j)

        self.cluster_template_similarity[cluster_id, :] = np.nan_to_num(
            self.cluster_template_similarity[cluster_id, :]
        )
        self.cluster_template_similarity[
            cluster_id, self.cluster_template_similarity[cluster_id, :] < 0.001
        ] = 0
        return self.cluster_template_similarity[cluster_id, :]

    def merge(self, cluster_ids: list[int]) -> int:
        """
        Merge clusters together.

        Args:
            cluster_ids (list[int]): List of cluster IDs to merge together.

        Returns:
            int: New cluster ID of the merged clusters.
        """
        new_id = self.get_new_id()

        # update spike_clusters
        for i in range(len(cluster_ids)):
            for ind in self.cl_inds[cluster_ids[i]]:
                self.spike_clusters[ind] = new_id

        # update cl_times
        temp = self.cl_times[cluster_ids[0]]
        for i in range(1, len(cluster_ids)):
            temp = np.append(temp, self.cl_times[cluster_ids[i]])
        self.cl_times.append(temp)

        # update cl_inds
        temp = self.cl_inds[cluster_ids[0]]
        for i in range(1, len(cluster_ids)):
            temp = np.append(temp, self.cl_inds[cluster_ids[i]])
        self.cl_inds.append(temp)

        # update cluster_templates
        temp = list(self.cluster_templates[cluster_ids[0]])
        for i in range(1, len(cluster_ids)):
            temp += list(self.cluster_templates[cluster_ids[i]])
        self.cluster_templates[new_id] = temp

        # update spikes
        temp = self.spikes[cluster_ids[0]]
        for i in range(1, len(cluster_ids)):
            temp = np.append(temp, self.spikes[cluster_ids[i]], axis=0)
        self.spikes[new_id] = temp

        # update mean_wf
        temp = np.sum([self.mean_wf[i] * self.counts[i] for i in cluster_ids], axis=0)
        self.mean_wf = np.concatenate(
            (self.mean_wf, np.expand_dims(temp, axis=0)), axis=0
        )

        # expand sim arrays
        self.cluster_template_similarity = np.concatenate(
            (self.cluster_template_similarity, -1 * np.ones((1, self.n_clusters))),
            axis=0,
        )
        self.cluster_autoencoder_similarity = np.concatenate(
            (self.cluster_autoencoder_similarity, -1 * np.ones((1, self.n_clusters))),
            axis=0,
        )

        # AI Gen code for merging clusters
        # self.cluster_ids = np.append(self.cluster_ids, new_id)
        # self.counts[new_id] = 0
        # self.counts[new_id] = sum([self.counts[i] for i in cluster_ids])

        # # update spike_clusters
        # for i in cluster_ids:
        #     self.spike_clusters[self.spike_clusters == i] = new_id

        # # update cluster_metrics
        # new_metrics = self.cluster_metrics.loc[
        #     self.cluster_metrics["cluster_id"].isin(cluster_ids)
        # ].mean()
        # new_metrics["cluster_id"] = new_id
        # self.cluster_metrics = self.cluster_metrics.append(
        #     new_metrics, ignore_index=True
        # )

        # # update mean_wf
        # self.mean_wf[new_id, :, :] = np.mean(self.mean_wf[cluster_ids, :, :], axis=0)

        # # update template_counts
        # self.template_counts[new_id] = np.concatenate(
        #     [self.template_counts[i] for i in cluster_ids]
        # )
        logger.info(f"Merged clusters {cluster_ids} into {new_id}.")
        return new_id

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

    def _load_ks_files(self):
        """
        load kilosort files and raw data.
            * spike_times.npy
            * spike_clusters.npy | spike_templates.npy
            * cluster_group.tsv
            * channel_positions.npy
            * spike_templates.npy
            * similar_templates.npy

        Sets attributes:
            - spike_times
            - spike_clusters
            - cluster_groups
            - channel_positions
            - spike_templates
            - similar_templates
            - n_templates
        """
        self.spike_times = np.load(
            os.path.join(self.ks_folder, "spike_times.npy")
        ).flatten()
        try:
            self.spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_clusters.npy")
            ).flatten()
        except FileNotFoundError:
            self.spike_clusters = np.load(
                os.path.join(self.ks_folder, "spike_templates.npy")
            ).flatten()
        self.cluster_groups = pd.read_csv(
            os.path.join(self.ks_folder, "cluster_group.tsv"), sep="\t"
        )
        self.channel_positions = np.load(
            os.path.join(self.ks_folder, "channel_positions.npy")
        )
        self.spike_templates = np.load(
            os.path.join(self.ks_folder, "spike_templates.npy")
        ).flatten()
        self.similar_templates = np.load(
            os.path.join(self.ks_folder, "similar_templates.npy")
        )
        self.n_templates = self.similar_templates.shape[0]

        self.raw_data = RawData(self.params["data_path"], self.params["n_channels"])

    def _initialize(self):
        self.cluster_ids, self.counts = np.unique(
            self.spike_clusters, return_counts=True
        )
        self.counts = dict(zip(self.cluster_ids, self.counts))
        self.cl_times, self.cl_inds = find_times_multi(
            self.spike_times,
            self.spike_clusters,
            np.arange(self.n_clusters),
            return_inds=True,
        )
        self.spikes = (
            self._load_spikes()
        )  # TODO this is expensive, can we not recalculate every time?
        self.mean_wf = self._calc_mean_wf()
        self.cluster_metrics = self._load_cluster_metrics()

        self.cluster_template_similarity = -1 * np.ones(
            (self.n_clusters, self.n_clusters)
        )
        self.cluster_autoencoder_similarity = -1 * np.ones(
            (self.n_clusters, self.n_clusters)
        )
        self.cluster_templates = self._calc_template_counts()

    def _calc_mean_wf(self):
        """
        Calculate mean waveforms for each cluster.

        Returns:
            np.ndarray: Mean waveforms for each cluster. Shape (n_clusters, n_channels, pre_samples + post_samples)
        """
        mean_wf_path = os.path.join(self.ks_folder, "mean_waveforms.npy")
        # TODO ensure right size?
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
        for i in self.cluster_ids:
            if self.counts[i] > 0:
                mean_wf[i, :, :] = np.nanmean(self.spikes[i], axis=0)

        np.save(mean_wf_path, mean_wf)

        return mean_wf

    def _load_cluster_metrics(self) -> pd.DataFrame:
        """
        Load the cluster metrics from the cluster metrics files.

        Returns:
            pd.DataFrame: DataFrame containing the cluster metrics.
        """
        # check if cilantro_metrics.tsv exists - use new file in order to to not overwrite other metrics
        metrics_path = os.path.join(self.ks_folder, "cilantro_metrics.tsv")
        # TODO: check if cilantro_metrics.tsv is up to date
        if os.path.exists(metrics_path):
            return pd.read_csv(metrics_path, sep="\t")

        # if not, load metrics from individual files
        metrics = pd.DataFrame()

        seps = {".csv": ",", ".tsv": "\t"}
        exclude = ["waveform_metrics", "metrics"]

        for f in os.listdir(self.ks_folder):
            if (f.endswith(".csv") or f.endswith(".tsv")) and (f[:-4] not in exclude):
                df = pd.read_csv(os.path.join(self.ks_folder, f), sep=seps[f[-4:]])

                if "cluster_id" in df.columns:
                    new_feat = [col for col in df.columns if col not in metrics.columns]
                    metrics = (
                        df
                        if metrics.empty
                        else pd.merge(
                            metrics,
                            df[["cluster_id"] + new_feat],
                            on="cluster_id",
                            how="outer",
                        )
                    )

        # remove duplicate columns
        metrics = metrics.loc[:, ~metrics.columns.duplicated()]
        metrics.drop(columns=["fr"], axis=1, inplace=True, errors="ignore")

        # remove amplitude columns
        metrics.drop(
            columns=["amplitude", "amp", "Amplitude"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        # remove columns with "Unnamed" in the name
        metrics = metrics.loc[:, ~metrics.columns.str.contains("Unnamed")]

        # Occassionally n_spikes is not correct, so override it with the actual count
        metrics["n_spikes"] = (
            metrics["cluster_id"].map(self.counts).fillna(metrics["n_spikes"])
        )

        # Add label_reason column and set to NA
        metrics["label_reason"] = pd.NA  # TODO or empty string better?

        # rename template peak channel and save waveform peak channel
        if "ch" in metrics.columns:
            metrics.rename(columns={"ch": "template_peak"}, inplace=True)
            metrics["template_peak"] = pd.to_numeric(
                metrics["template_peak"], errors="coerce", downcast="integer"
            )

        # peak_channel exists in some files, but we will recalculate to ensure it is waveform peak channel and set non-cluster channels to NaN
        metrics.drop(columns=["peak_channel"], axis=1, inplace=True, errors="ignore")

        peaks = np.full_like(metrics["cluster_id"], np.nan)
        peaks[self.cluster_ids] = np.argmax(
            np.max(self.mean_wf, 2) - np.min(self.mean_wf, 2), 1
        )
        metrics["peaks"] = peaks

        # rename group to label if it exists or use KSLabel
        if "group" not in metrics.columns:
            metrics["label"] = metrics["KSLabel"]
        else:
            metrics.rename(columns={"group": "label"}, inplace=True)

        # save metrics to file
        # metrics.to_csv(metrics_path, sep="\t", index=False)
        return metrics

    def _calc_template_counts(self) -> dict[int, np.ndarray]:
        """
        Calculate the template counts for each cluster.

        Returns:
            dict[int, np.ndarray]: Template counts for each cluster.
        """
        template_counts = {}
        for i in range(self.n_clusters):
            if i in self.counts:
                spike_ids = self.cl_inds[i]
                templates = self.spike_templates[spike_ids]
                template_counts_i = np.bincount(templates, minlength=self.n_templates)
                template_counts[i] = np.nonzero(template_counts_i)[0]

        return template_counts

    def _load_spikes(self):
        """
        Load spikes from the raw data.
        """
        spikes = {}
        # TODO this seems wrong since may have already had some curation
        # TODO this is expensive, can we not recalculate every time? has it already been calculated?
        for i in tqdm(range(self.n_clusters), desc="Loading spike data"):
            if (i in self.cluster_ids) and (self.counts[i] > 0):
                spikes[i] = extract_spikes(
                    self.raw_data.data,
                    self.cl_times,
                    i,
                    n_chan=self.params["n_channels"],
                    pre_samples=self.params["pre_samples"],
                    post_samples=self.params["post_samples"],
                    max_spikes=self.params["max_spikes"],
                )
        return spikes


if __name__ == "__main__":
    ks_folder = r"D:\040224_INS2_4_DT3_rec_g0\catgt_040224_INS2_4_DT3_rec_g0\040224_INS2_4_DT3_rec_g0_imec0\imec0_ks2"
    curator = Curator(ks_folder)
    curator.template_similarity(0)
    curator.merge([0, 1])
