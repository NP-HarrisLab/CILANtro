import numpy as np
import pytest

from cilantropy.curation import Curator


@pytest.fixture
def curator() -> Curator:
    ks_folder = r"D:\040224_INS2_4_DT3_rec_g0\catgt_040224_INS2_4_DT3_rec_g0\040224_INS2_4_DT3_rec_g0_imec0\imec0_ks2"
    params = {"max_spikes": 10}
    return Curator(ks_folder, params)


def test_get_new_id(curator: Curator):
    new_id = curator.get_new_id()
    assert new_id == max(curator.cluster_ids) + 1


def test_template_similarity(curator: Curator):
    cluster_id = 0
    similarity = curator.template_similarity(cluster_id)
    assert isinstance(similarity, np.ndarray)
    assert similarity.shape == (curator.n_clusters,)


def test_merge(curator: Curator):
    cluster_ids = [0, 1, 2]
    new_id = curator.merge(cluster_ids)
    assert new_id == max(curator.cluster_ids)
    assert new_id in curator.cluster_ids
    assert all(id not in curator.cluster_ids for id in cluster_ids)
    # TODO check other things that are merged


if __name__ == "__main__":
    pytest.main()
