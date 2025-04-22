import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import ecephys_spike_sorting as ecephys
import npx_utils as npx
import numpy as np
import slay
from tqdm import tqdm

from cilantropy.curation import Curator


def run_catgt(cmd):
    ecephys_json_path = os.path.join(
        os.path.dirname(ecephys.sglx_pipeline.__file__), "create_input_json_params.json"
    )
    # load json file with parameters for catGT
    with open(ecephys_json_path, "r") as f:
        ecephys_json = json.load(f)
    catGTPath = ecephys_json["catGTPath"]

    if sys.platform.startswith("win"):
        os_str = "win"
        # build windows command line
        # catGTexe_fullpath = catGTPath.replace('\\', '/') + "/runit.bat"
        # call catGT directly with params. CatGT.log file will be saved lcoally
        # in current working directory (with the calling script)
        catGTexe_fullpath = catGTPath.replace("\\", "/") + "/CatGT"
    elif sys.platform.startswith("linux"):
        os_str = "linux"
        catGTexe_fullpath = catGTPath.replace("\\", "/") + "/runit.sh"
    else:
        print("unknown system, cannot run CatGt")

    cmd = catGTexe_fullpath + " " + cmd
    print(cmd)
    subprocess.Popen(cmd, shell="False").wait()


if __name__ == "__main__":
    # SET PARAMETERS ############################################
    # folder should be one above supercat _g0 folders
    params = {
        "folder": r"D:\Psilocybin\Cohort_2\T13\20250417_T13_acute",
        "overwrite": False,
        "processing_drive": "D:",
        "delete": False,  # delete from processing drive
        "supercat_params": {
            "trim_edges": False,
            "run_CatGT": True,
            "ni_present": False,
            "runTPrime": False,
            "run_kilosort": True,
            "run_kilosort_postprocessing": True,
            "run_noise_templates": False,
            "run_mean_waveforms": False,
            "run_quality_metrics": False,
            "ks_ver": "4",
        },
        "curator_params": {"overwrite": False},  # default
        "run_auto_curate": False,
        "auto_curate_params": {},  # default
        "run_merge": False,
        "merge_params": {
            "overwrite": False,
            "plot_merges": False,
            "max_spikes": 500,
            "auto_accept_merges": False,
        },  # default
        "run_post_merge_curation": False,
        "post_merge_curation_params": {},
    }
    orig_drive = params["folder"].split("\\")[0]
    npx_directory = params["folder"]

    proc_npx_directory = npx_directory.replace(orig_drive, params["processing_drive"])
    dest_folder = proc_npx_directory

    if params["supercat_params"]["run_CatGT"]:
        catgt_folders_orig = [
            os.path.join(npx_directory, folder)
            for folder in os.listdir(npx_directory)
            if folder.startswith("catgt")
        ]
        if len(catgt_folders_orig) < 2:
            raise ValueError("Not enough folders to compare.")

        catgt_folders = [
            catgt_folder.replace(orig_drive, params["processing_drive"])
            for catgt_folder in catgt_folders_orig
        ]
        for catgt_folder_proc, catgt_folder_orig in zip(
            catgt_folders, catgt_folders_orig
        ):
            npx.copy_folder_with_progress(catgt_folder_orig, catgt_folder_proc)

        times = []
        lf = True
        probe_ids = ""
        for catgt_folder in catgt_folders:
            run_name = os.path.basename(catgt_folder).split("catgt_")[-1]
            probe_folders = [
                os.path.join(catgt_folder, name)
                for name in os.listdir(catgt_folder)
                if os.path.isdir(os.path.join(catgt_folder, name))
            ]
            probe_ids = [probe_folder[-1] for probe_folder in probe_folders]

            meta_file = os.path.join(
                probe_folders[0],
                run_name + f"_tcat.imec{probe_ids[0]}.ap.meta",
            )
            meta = npx.read_meta(meta_file)
            time = datetime.fromisoformat(meta["fileCreateTime"])
            times.append(time)

            lf_bin = os.path.join(
                probe_folders[0],
                run_name + f"_tcat.imec{probe_ids[0]}.lf.bin",
            )
            if not os.path.exists(lf_bin):
                lf = False

            probe_ids = ",".join(probe_ids)

        # sort folders by creation time
        catgt_folders = [folder for _, folder in sorted(zip(times, catgt_folders))]
        run_name = os.path.basename(catgt_folders[0]).split("catgt_")[-1]
        run_name = run_name[:-3]  # remove _g0
        probes_folders = [
            os.path.join(catgt_folders[0], name)
            for name in os.listdir(catgt_folders[0])
            if os.path.isdir(os.path.join(catgt_folders[0], name))
        ]
        probe_ids = [probe_folder[-1] for probe_folder in probes_folders]
        probe_ids = ",".join(probe_ids)
    else:
        supercat_folders = [
            os.path.join(npx_directory, folder)
            for folder in os.listdir(npx_directory)
            if folder.startswith("supercat")
        ]
        if len(supercat_folders) < 1:
            raise ValueError("No supercat folders found.")
        supercat_folder = supercat_folders[0]
        run_name = supercat_folder.split("supercat_")[-1]
        run_name = run_name[:-3]
        catgt_folders = []
        probes_folders = [
            os.path.join(supercat_folder, name)
            for name in os.listdir(supercat_folder)
            if os.path.isdir(os.path.join(supercat_folder, name))
        ]
        probe_ids = [probe_folder[-1] for probe_folder in probes_folders]
        probe_ids = ",".join(probe_ids)
    supercat_params = params["supercat_params"]
    supercat_params["run_name"] = run_name
    supercat_params["supercat"] = True
    supercat_params["supercat_folders"] = catgt_folders
    supercat_params["npx_directory"] = proc_npx_directory
    supercat_params["probes"] = probe_ids
    ecephys.sglx_pipeline.main(supercat_params)

    supercat_folders = [
        os.path.join(dest_folder, folder)
        for folder in os.listdir(dest_folder)
        if folder.startswith("supercat")
    ]
    supercat_folder = supercat_folders[0]
    ks_folders = npx.get_ks_folders(supercat_folder, catgt=False)
    for ks_folder in ks_folders:
        npx.copy_folder_with_progress(ks_folder, ks_folder + "_jc")
        with Curator(ks_folder, **params["curator_params"]) as curator:
            if params["run_auto_curate"]:
                curator.auto_curate(params["auto_curate_params"])
            if params["run_merge"]:
                if params["merge_params"]["overwrite"] and os.path.exists(
                    os.path.join(ks_folder, "automerge", "new2old.json")
                ):
                    shutil.rmtree(os.path.join(ks_folder, "automerge"))
                if not os.path.exists(
                    os.path.join(ks_folder, "automerge", "new2old.json")
                ):
                    slay.run.main({"KS_folder": ks_folder, **params["merge_params"]})
                else:
                    tqdm.write("Merges already exists")

            if params["run_post_merge_curation"]:
                curator.post_merge_curation(params["post_merge_curation_params"])

    if orig_drive != params["processing_drive"]:
        supercat_folders = [
            os.path.join(dest_folder, folder)
            for folder in os.listdir(dest_folder)
            if folder.startswith("supercat")
        ]
        if len(supercat_folders) < 1:
            raise ValueError("No supercat folders found.")
        if len(supercat_folders) > 1:
            raise ValueError("More than one supercat folder found.")
        supercat_folder = supercat_folders[0]
        npx.copy_folder_with_progress(
            supercat_folder,
            supercat_folder.replace(params["processing_drive"], orig_drive),
        )
        if params["delete"]:
            shutil.rmtree(dest_folder)
