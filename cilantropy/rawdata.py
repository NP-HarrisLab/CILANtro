import numpy as np


# TODO LFP vs AP?
class RawData(object):
    def __init__(self, bin_path: str, n_channels: int | None = None) -> None:
        self.bin_path = bin_path
        self.meta_path = bin_path.replace(".bin", ".meta")
        self.n_channels = self._get_num_channels() if n_channels is None else n_channels

        self.data = np.memmap(self.bin_path, dtype="int16", mode="r")
        self.data = np.reshape(self.data, (-1, self.n_channels))

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def _get_num_channels(self) -> int:
        """Get the number of channels from the meta file."""
        with open(self.meta_path, "r") as f:
            for meta in f.readlines():
                if "nSavedChans" in meta:
                    return int(meta.split("=")[1])
        raise ValueError("Number of channels not found in meta file")
