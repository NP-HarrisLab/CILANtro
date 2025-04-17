import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import ecephys_spike_sorting as ecephys
import npx_utils as npx


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
        "folder": r"Z:\Psilocybin\Cohort_2\T12\20250415_T12_baseline",
        "overwrite": True,
        "trim_edges": False,  # make false if use maxsecs when catgt the runs
        "processing_drive": "D:",
        "delete": False,  # delete from processing drive
    }
    orig_drive = params["folder"].split("\\")[0]
    npx_directory = params["folder"]

    proc_npx_directory = npx_directory.replace(orig_drive, params["processing_drive"])
    npx.copy_folder_with_progress(npx_directory, proc_npx_directory)
    dest_folder = proc_npx_directory

    catgt_folders = [
        os.path.join(proc_npx_directory, folder)
        for folder in os.listdir(proc_npx_directory)
        if folder.startswith("catgt")
    ]
    if len(catgt_folders) < 2:
        raise ValueError("Not enough folders to compare.")

    times = []
    lf = True
    probe_ids = ""
    for catgt_folder in catgt_folders:
        run_name = os.path.basename(catgt_folder).split("catgt_")[-1]
        probe_folders = [
            os.path.join(catgt_folder, name)
            for name in os.listdir(os.path.join(proc_npx_directory, run_name))
            if os.path.isdir(os.path.join(proc_npx_directory, run_name, name))
        ]
        probe_ids = [probe_folder[-1] for probe_folder in probe_folders]

        meta_file = os.path.join(
            probe_folders[0], run_name + f"_tcat.imec{probe_ids[0]}.ap.meta"
        )
        meta = npx.read_meta(meta_file)
        time = datetime.fromisoformat(meta["fileCreateTime"])
        times.append(time)

        lf_bin = os.path.join(
            probe_folders[0], run_name + f"_tcat.imec{probe_ids[0]}.lf.bin"
        )
        if not os.path.exists(lf_bin):
            lf = False

        probe_ids = ",".join(probe_ids)

    # sort folders by creation time
    catgt_folders = [folder for _, folder in sorted(zip(times, catgt_folders))]

    supercat = "-supercat="
    for folder in catgt_folders:
        supercat += "{" + proc_npx_directory + "," + os.path.basename(folder) + "}"
    supercat += " -ap"
    if lf:
        supercat += " -lf"
    if params["trim_edges"]:
        supercat += " -trim_edges"
    supercat += " -prb=" + probe_ids
    supercat += " -prb_fld -out_prb_fld"
    supercat += " -dest=" + dest_folder

    run_catgt(supercat)

    if orig_drive != params["processing_drive"]:
        npx.copy_folder_with_progress(
            dest_folder, dest_folder.replace(params["processing_drive"], orig_drive)
        )
        if params["delete"]:
            shutil.rmtree(proc_npx_directory)
