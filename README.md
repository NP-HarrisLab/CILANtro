# CILANtro



# Running the CILANtro pipeline

1. One-time: adjust the default quality metric thresholds in [`schemas.py`](http://schemas.py) if desired. These are in the `missing` argument of each field in the `AutoCurateParams` class definition.
2. In the main function at the bottom of `auto_curate.py` , change `“folder”` in the `params` dictionary to the desired folder. Note that CILANtro will automatically search for all SGLX recordings in the folder, so you can batch sort + curate many recordings by changing `folder` to a folder containing all the top-level SGLX recording folders
3. If the recordings you want to sort are stored on your local hard drive (C: or D:), change `processing_drive` to the drive that the recordings are on.
4. If you don’t want to auto-accept merge suggestions, set `auto_accept_merges` and `run_post_merge_curation` to False.
    1. Once you’ve merged the desired suggestions in Phy, you can rerun `auto_curate.py` with only `run_post_merge_curation` set to True and all other stages (including `overwrite`) set to False. This will recalculate the metrics on the merged clusters and apply additional post merge filters.
5. Adjust any other parameters as desired in `main` (E.g. setting `ni_present` and `run_TPrime` to true if you have ni data, or setting `process_lf` to false if you don't want to extract LFP.)
6. Run `auto_curate.py` from the command line with `python auto_curate.py`.

# Installation Instructions

## Clone repos

```bash
*navigate to desired directory*
conda create -n pipeline python=3.10

git clone https://github.com/NP-HarrisLab/ecephys_spike_sorting.git
git clone https://github.com/NP-HarrisLab/CILANtro.git
git clone https://github.com/saikoukunt/SLAy.git
git clone https://github.com/NP-HarrisLab/npx_utils.git
git clone https://github.com/kwikteam/npy-matlab
```

## Install packages

```bash
conda activate pipeline
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
conda install scipy numpy pandas scikit-learn matplotlib
pip install cupy-cuda12x marshmallow
```

## Install CILANTro

```bash
cd <cilantro_directory>
pip install -e .
```

## Install npx_utils

```bash
cd <npx_utils_directory>
pip install -e .
```

## Install ecephys_spike_sorting

1. Download and extract TPrime, CatGT, and CWaves from the [SpikeGLX website](https://billkarsh.github.io/SpikeGLX/#catgt)
2. Install Kilosort 4
    
    ```bash
    pip install kilosort
    ```
    
3. Downgrade setuptools
    
    ```bash
    pip uninstall setuptools
    pip install setuptools==59.8.0
    ```
    
4. Install the package
    
    ```bash
    cd <ecephys_directory>
    pip install -e .
    pip install h5py
    pip install phylib
    ```
    
5. In `ecephys_spike_sorting/scripts/create_input_json_params.json`, change the directories to point to your local installations (if you only want to run kilosort4, you still must set the KS2.5 path to a folder THAT EXISTS)
6. (OPTIONAL) If you want to run kilosort2.5 
    1. Install kilosort2.5 from using the instructions [here](https://github.com/MouseLand/Kilosort/tree/kilosort25) 
    2. Install the MATLAB engine for python
        
        ```bash
        conda activate pipeline
        cd <matlabroot>\extern\engines\python
        python setup.py install
        ```
        
        Replace with the root directory of your MATLAB installation, for example: `C:\Program Files\MATLAB\R2021b`
        

## Install SLAy

```bash
cd <slay_directory>
pip install -e .
```
