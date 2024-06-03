import numpy as np


def extract_spikes(
    data: np.ndarray[np.int_],
    times_multi: list[np.ndarray[np.float_]],
    clust_id: int,
    pre_samples: int = 20,
    post_samples: int = 62,
    n_chan: int = 385,
    max_spikes: int = -1,
) -> np.ndarray[np.int_]:
    """
    Extracts spike waveforms for the specified cluster.

    If the cluster contains more than `max_spikes` spikes, `max_spikes` random
    spikes are extracted instead.

    Args:
        data (np.ndarray): Ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        times_multi (list): Spike times indexed by cluster id.
        clust_id (list): The cluster to extract spikes from
        pre_samples (int): The number of samples to extract before the peak of the
            spike. Defaults to 20.
        post_samples (int): The number of samples to extract after the peak of the
            spike. Defaults to 62.
        n_chan (int): The number of channels in the recording. Defaults to
            385 to match NP 1.0/2.0.
        max_spikes (int): The maximum number of spikes to extract. If -1, all
            spikes are extracted. Defaults to -1.

    Returns:
        spikes (np.ndarray): Array of extracted spike waveforms with shape
            (# of spikes, # of channels, # of timepoints).
    """
    times: np.ndarray[np.int_] = times_multi[clust_id].astype("int64")

    # Ignore spikes that are cut off by the ends of the recording
    while (times[0] - pre_samples) < 0:
        times = times[1:]
    while (times[-1] + post_samples) >= data.shape[0]:
        times = times[:-1]

    # Randomly pick spikes if the cluster has too many
    if (max_spikes != -1) and (max_spikes < times.shape[0]):
        np.random.shuffle(times)
        times = times[:max_spikes]

    spikes: np.ndarray[np.int_] = np.zeros(
        (times.shape[0], n_chan, pre_samples + post_samples), dtype="int64"
    )
    for i in range(times.shape[0]):
        spikes[i, :, :] = data[times[i] - pre_samples : times[i] + post_samples, :].T

    return spikes


def find_times_multi(
    sp_times: np.ndarray[np.float_],
    sp_clust: np.ndarray[np.int_],
    clust_ids: list[int],
    return_inds: bool = False,
) -> list[np.ndarray[np.float_]]:
    """
    Finds all the spike times for each of the specified clusters.

    Args:
        sp_times (np.ndarray): Spike times (in any unit of time).
        sp_clust (np.ndarray): Spike cluster assignments.
        clust_ids (list[int]): Clusters for which spike times should be returned.
        return_inds (bool, optional): Whether to return the indices of the spike times as well. Defaults to False.

    Returns:
        list[NDArray[np.float_]]: Found cluster spike times. If `return_inds` is True,
        it also returns the indices of the spike times for each cluster.
    """
    # Initialize the returned list and map cluster id to list index
    cl_times: list = []
    cl_inds: list = []
    cl_to_ind: dict[int, int] = {}

    for i in range(len(clust_ids)):
        cl_times.append([])
        cl_inds.append([])
        cl_to_ind[clust_ids[i]] = i

    for i in range(sp_times.shape[0]):
        if sp_clust[i] in cl_to_ind:
            cl_times[cl_to_ind[sp_clust[i]]].append(sp_times[i])
            cl_inds[cl_to_ind[sp_clust[i]]].append(i)
    for i in range(len(cl_times)):
        cl_times[i] = np.array(cl_times[i])
        cl_inds[i] = np.array(cl_inds[i])
    if return_inds:
        return cl_times, cl_inds
    return cl_times
