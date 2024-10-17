import os

from marshmallow import Schema, fields


class CuratorParams(Schema):
    # TODO get rid of RemovedInMarshmallow4Warning
    """
    Class for curation parameters
    """
    KS_folder = fields.Str(
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
    n_chan = fields.Int(
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
    good_lbls = fields.List(
        fields.String,
        required=False,
        cli_as_single_argument=True,
        missing=["good", "mua"],
        description="Cluster labels that denote non-noise clusters.",
    )
    max_spikes = fields.Int(
        required=False,
        missing=500,
        description="Maximum number of spikes per cluster used to calculate mean waveforms and cross projections. -1 uses all spikes.",
    )


class AutoCurateParams(Schema):
    """
    Class for auto-curation parameters
    """

    min_spikes = fields.Int(
        required=False,
        missing=100,
        description="Minimum number of spikes in a cluster to undergo further stages.",
    )
    min_snr = fields.Float(
        required=False,
        missing=3.0,
        description="Minimum SNR for a cluster to undergo further stages.",
    )
    max_rp_viol = fields.Float(
        required=False,
        missing=0.1,
        description="Maximum refractory period violations for a cluster to undergo further stages.",
    )
    max_peaks = fields.Int(
        required=False,
        missing=1,
        description="Maximum number of peaks in a cluster waveform for it to be considered noise.",
    )
    max_troughs = fields.Int(
        required=False,
        missing=1,
        description="Maximum number of troughs in a cluster waveform for it to be considered noise.",
    )
    max_wf_dur = fields.Float(
        required=False,
        missing=0.9,
        description="Maximum duration of a cluster waveform for it to be considered noise.",
    )
    min_spat_decay = fields.Float(
        required=False,
        missing=-0.1,
        description="Minimum spatial decay of a cluster waveform for it to be considered noise.",
    )
    good_lbls = fields.List(
        fields.String,
        required=False,
        cli_as_single_argument=True,
        missing=["good", "mua"],
        description="Cluster labels that denote non-noise clusters.",
    )
