import numpy as np

# Format for parameter specification:
# parameter: {
#     'gui_name': text displayed next to edit box in GUI
#     'type': callable datatype for this parameter, like int or float.
#     'min': minimum value allowed (inclusive).
#     'max': maximum value allowed (inclusive).
#     'exclude': list of individual values to exclude from allowed range.
#     'default': default value used by gui and API
#     'step': which step of the pipeline the parameter is used in, from:
#             ['data', 'preprocessing', 'spike detection',
#              'clustering', 'postprocessing']
#     'description': Explanation of parameter's use. Populates parameter help
#                    in GUI.
# }

MAIN_PARAMETERS = {
    # NOTE: n_chan_bin must be specified by user when running through API
    "n_chan_bin": {
        "gui_name": "number of channels",
        "type": int,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 385,
        "step": "data",
        "description": """
            Total number of channels in the binary file, which may be different
            from the number of channels containing ephys data. The value of this
            parameter *must* be specified by the user, or `run_kilosort` will
            raise a ValueError.
            """,
    },
    "fs": {
        "gui_name": "sampling frequency",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 30000,
        "step": "data",
        "description": """
            Sampling frequency of probe.
            """,
    },
    "batch_size": {
        "gui_name": "batch size",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 60000,
        "step": "data",
        "description": """
            Number of samples included in each batch of data.
            """,
    },
    "nblocks": {
        "gui_name": "nblocks",
        "type": int,
        "min": 0,
        "max": np.inf,
        "exclude": [],
        "default": 1,
        "step": "preprocessing",
        "description": """
            Number of non-overlapping blocks for drift correction
            (additional nblocks-1 blocks are created in the overlaps).
            """,
    },
    "Th_universal": {
        "gui_name": "Th (universal)",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 9,
        "step": "spike detection",
        "description": """
            Spike detection threshold for universal templates.
            Th(1) in previous versions of Kilosort.
            """,
    },
    "Th_learned": {
        "gui_name": "Th (learned)",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 8,
        "step": "spike detection",
        "description": """
            Spike detection threshold for learned templates.
            Th(2) in previous versions of Kilosort.
            """,
    },
    "tmin": {
        "gui_name": "tmin",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [],
        "default": 0,
        "step": "data",
        "description": """
            Time in seconds when data used for sorting should begin. By default,
            begins at 0 seconds.
            """,
    },
    "tmax": {
        "gui_name": "tmax",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": np.inf,
        "step": "data",
        "description": """
            Time in seconds when data used for sorting should end. By default,
            ends at the end of the recording.
            """,
    },
}


EXTRA_PARAMETERS = {
    ### DATA
    "nt": {
        "gui_name": "nt",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 61,
        "step": "data",
        "description": """
            Number of samples per waveform. Also size of symmetric padding
            for filtering.
            """,
    },
    ### PREPROCESSING
    "artifact_threshold": {
        "gui_name": "artifact threshold",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [],
        "default": np.inf,
        "step": "preprocessing",
        "description": """
            If a batch contains absolute values above this number, it will be
            zeroed out under the assumption that a recording artifact is present.
            By default, the threshold is infinite (so that no zeroing occurs).
            """,
    },
    "nskip": {
        "gui_name": "nskip",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 25,
        "step": "preprocessing",
        "description": """
            Batch stride for computing whitening matrix.
            """,
    },
    "whitening_range": {
        "gui_name": "whitening range",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 32,
        "step": "preprocessing",
        "description": """
            Number of nearby channels used to estimate the whitening matrix.
            """,
    },
    "binning_depth": {
        "gui_name": "binning_depth",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 5,
        "step": "preprocessing",
        "description": """
            For drift correction, vertical bin size in microns used for
            2D histogram.
            """,
    },
    "sig_interp": {
        "gui_name": "sig_interp",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 20,
        "step": "preprocessing",
        "description": """
            Approximate spatial smoothness scale in units of microns.
            """,
    },
    "drift_smoothing": {
        "gui_name": "drift smoothing",
        "type": list,
        "min": None,
        "max": None,
        "exclude": [],
        "default": [0.5, 0.5, 0.5],
        "step": "preprocessing",
        "description": """
            Amount of gaussian smoothing to apply to the spatiotemporal drift
            estimation, for x,y,time axes in units of registration blocks
            (for x,y axes) and batch size (for time axis). The x,y smoothing has
            no effect for `nblocks = 1`.
            """,
    },
    "remove_chan_delays": {
        "gui_name": "remove channel delays",
        "type": bool,
        "min": None,
        "max": None,
        "exclude": [],
        "default": False,
        "step": "preprocessing",
        "description": """
            If True, will remove channel delays from the data. This is useful
            for intramuscualr EMG recordings, where delays between motor unit
            action potentials can be significant and affect spike sorting.
            """,
    },
    "remove_spike_outliers": {
        "gui_name": "remove spike outliers",
        "type": bool,
        "min": None,
        "max": None,
        "exclude": [],
        "default": False,
        "step": "preprocessing",
        "description": """
            If True, will remove outlier spikes from the data. This is useful
            for recordings with a high number of artifacts, which can affect
            spike sorting during initialization. HDBSCAN is used to identify
            outlier spikes.
            """,
    },
    "hdbscan_min_cluster_size": {
        "gui_name": "adjust minimum cluster size for detecting spike outliers",
        "type": int,
        "min": None,
        "max": None,
        "exclude": [],
        "default": 20,
        "step": "preprocessing",
        "description": """
            If remove_spike_outliers is True, this sets the minimum number of spikes to be considered a cluster. See the HDBSCAN min_cluster_size parameter in the sklearn documentation for more details.
            """,
    },
    ### SPIKE DETECTION
    # NOTE: if left as None, will be set to `int(20 * settings['nt']/61)`
    "nt0min": {
        "gui_name": "nt0min",
        "type": int,
        "min": 0,
        "max": np.inf,
        "exclude": [],
        "default": None,
        "step": "spike detection",
        "description": """
            Sample index for aligning waveforms, so that their minimum 
            or maximum value happens here. Defaults to 
            `int(20 * settings['nt']/61)`.
            """,
    },
    "dmin": {
        "gui_name": "dmin",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": None,
        "step": "spike detection",
        "description": """
            Vertical spacing of template centers used for spike detection,
            in microns. Determined automatically by default.
            """,
    },
    "dminx": {
        "gui_name": "dminx",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 32,
        "step": "spike detection",
        "description": """
            Horizontal spacing of template centers used for spike detection,
            in microns. The default 32um should work well for Neuropixels 1 and
            Neuropixels 2 probes. For other probe geometries, try setting this 
            to the median lateral distance between contacts to start.
            """,
    },
    "min_template_size": {
        "gui_name": "min template size",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 10,
        "step": "spike detection",
        "description": """
            Standard deviation of the smallest, spatial envelope Gaussian used
            for universal templates.
            """,
    },
    "template_sizes": {
        "gui_name": "template sizes",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 5,
        "step": "spike detection",
        "description": """
            Number of sizes for universal spike templates (multiples of the
            min_template_size).
            """,
    },
    "nearest_chans": {
        "gui_name": "nearest chans",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 10,
        "step": "spike detection",
        "description": """
            Number of nearest channels to consider when finding local maxima
            during spike detection.
            """,
    },
    "nearest_templates": {
        "gui_name": "nearest templates",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 100,
        "step": "spike detection",
        "description": """
            Number of nearest spike template locations to consider when finding
            local maxima during spike detection.
            """,
    },
    "max_channel_distance": {
        "gui_name": "max channel distance",
        "type": float,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": None,
        "step": "spike detection",
        "description": """
            Templates farther away than this from their nearest channel will
            not be used. Also limits distance between compared channels during
            clustering.
            """,
    },
    "templates_from_data": {
        "gui_name": "templates from data",
        "type": bool,
        "min": None,
        "max": None,
        "exclude": [],
        "default": True,
        "step": "spike detection",
        "description": """
            Indicates whether spike shapes used in universal templates should be 
            estimated from the data or loaded from the predefined templates.
            """,
    },
    "n_templates": {
        "gui_name": "n templates",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 6,
        "step": "spike detection",
        "description": """
            Number of single-channel templates to use for the universal
            templates (only used if templates_from_data is True).
            """,
    },
    "n_pcs": {
        "gui_name": "n pcs",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 6,
        "step": "spike detection",
        "description": """
            Number of single-channel PCs to use for extracting spike features
            (only used if templates_from_data is True).
            """,
    },
    "Th_single_ch": {
        "gui_name": "Th (single channel)",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 6,
        "step": "spike detection",
        "description": """
            For single channel threshold crossings to compute universal-
            templates. In units of whitened data standard deviations. 
            """,
    },
    ### CLUSTERING
    "acg_threshold": {
        "gui_name": "acg threshold",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 0.2,
        "step": "clustering",
        "description": """
            Fraction of refractory period violations that are allowed in the ACG 
            compared to baseline; used to assign "good" units. 
            """,
    },
    "ccg_threshold": {
        "gui_name": "ccg threshold",
        "type": float,
        "min": 0,
        "max": np.inf,
        "exclude": [0],
        "default": 0.25,
        "step": "clustering",
        "description": """
            Fraction of refractory period violations that are allowed in the CCG
            compared to baseline; used to perform splits and merges.
            """,
    },
    "cluster_downsampling": {
        "gui_name": "cluster downsampling",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": 20,
        "step": "clustering",
        "description": """
            Inverse fraction of nodes used as landmarks during clustering
            (can be 1, but that slows down the optimization). 
            """,
    },
    "x_centers": {
        "gui_name": "x centers",
        "type": int,
        "min": 1,
        "max": np.inf,
        "exclude": [],
        "default": None,
        "step": "clustering",
        "description": """
            Number of x-positions to use when determining center points for
            template groupings. If None, this will be determined automatically
            by finding peaks in channel density. For 2D array type probes, we
            recommend specifying this so that centers are placed every few
            hundred microns.
            """,
    },
    ### POSTPROCESSING
    "duplicate_spike_bins": {
        "gui_name": "duplicate spike bins",
        "type": int,
        "min": 0,
        "max": np.inf,
        "exclude": [],
        "default": 7,
        "step": "postprocessing",
        "description": """
            Number of bins for which subsequent spikes from the same cluster are
            assumed to be artifacts. A value of 0 disables this step.
            """,
    },
}

# Add default values to descriptions
for k, v in MAIN_PARAMETERS.items():
    s = f"""
        Default value: {str(v["default"])}   
        Min, max: ({str(v['min'])}, {str(v['max'])})   
        Type: {v['type'].__name__}
        """
    v["description"] += s
for k, v in EXTRA_PARAMETERS.items():
    s = f"""
        Default value: {str(v["default"])}   
        Min, max: ({str(v['min'])}, {str(v['max'])})   
        Type: {v['type'].__name__}
        """
    v["description"] += s

main_defaults = {k: v["default"] for k, v in MAIN_PARAMETERS.items()}
extra_defaults = {k: v["default"] for k, v in EXTRA_PARAMETERS.items()}
# In the format expected by `run_kilosort`
DEFAULT_SETTINGS = {**main_defaults, **extra_defaults}
