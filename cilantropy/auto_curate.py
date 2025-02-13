import datetime
import os
import re
import shutil

import slay
from ecephys_spike_sorting import sglx_pipeline
from npx_utils import copy_folder_with_progress, get_ks_folders, is_run_folder
from tqdm import tqdm

import cilantropy
from cilantropy.curation import Curator


def get_run_info(folder, ks_ver, params):
    """currently assumes gate 0-9 index. will process separately as well - need to update"""
    root_dir = os.path.abspath(folder)
    if is_run_folder(folder):
        return [
            get_ecephys_params(
                os.path.dirname(folder), os.path.basename(folder), ks_ver, params
            )
        ]
    pattern = re.compile(r".*_g[0-9]$")
    run_info = []
    for root, dirs, _ in os.walk(root_dir):
        if "$RECYCLE.BIN" in root:
            continue
        for d in dirs:
            if d.startswith("catgt") or d.startswith("old"):
                continue
            if pattern.match(d):
                info = get_ecephys_params(root, d, ks_ver, params)
                if info:
                    run_info.append(info)
    return run_info


def get_ecephys_params(npx_directory, run_dir, ks_ver, params):
    catgt_folder = os.path.join(npx_directory, f"catgt_{run_dir}")
    probe_folders = [
        os.path.join(catgt_folder, name)
        for name in os.listdir(os.path.join(npx_directory, run_dir))
        if os.path.isdir(os.path.join(npx_directory, run_dir, name))
    ]
    probe_ids = [probe_folder[-1] for probe_folder in probe_folders]

    # check what has been previously run
    needs_catgt = False
    needs_tprime = False
    needs_kilosort = False
    needs_kilosort_postprocessing = False
    needs_noise_templates = False
    needs_mean_waveforms = False
    needs_quality_metrics = False
    for probe_id, probe_folder in zip(probe_ids, probe_folders):
        # catGT
        tcat_ap = os.path.join(
            probe_folder,
            f"{run_dir}_tcat.imec{probe_id}.ap.bin",
        )
        tcat_lf = os.path.join(
            probe_folder,
            f"{run_dir}_tcat.imec{probe_id}.lf.bin",
        )
        if (params["process_lf"] and not os.path.exists(tcat_lf)) or not os.path.exists(
            tcat_ap
        ):
            needs_catgt = True
        # kilosort
        ks_folder = os.path.join(probe_folder, f"imec{probe_id}_ks{ks_ver}")
        needs_kilosort = not os.path.exists(ks_folder)
        # kilosort postprocessing
        ks_orig = os.path.join(probe_folder, f"imec{probe_id}_ks{ks_ver}_orig")
        needs_kilosort_postprocessing = not os.path.exists(ks_orig)  # TODO
        # noise templates
        needs_noise_templates = not os.path.exists(ks_orig)  # TODO
        # mean waveforms
        needs_mean_waveforms = not os.path.exists(
            os.path.join(ks_folder, "mean_waveforms.npy")
        )  # TODO
        # quality metrics
        needs_quality_metrics = not os.path.exists(
            os.path.join(ks_folder, "metrics.csv")
        )
    # tprime
    tprime = os.path.join(catgt_folder, f"{run_dir}_TPrime_cmd.txt")
    needs_tprime = not os.path.exists(tprime)

    run_catgt = params["run_CatGT"] and (needs_catgt or params["overwrite"])
    run_tprime = params["runTPrime"] and (needs_tprime or params["overwrite"])
    run_kilosort = params["run_kilosort"] and (needs_kilosort or params["overwrite"])
    run_kilosort_postprocessing = params["run_kilosort_postprocessing"] and (
        needs_kilosort_postprocessing or params["overwrite"]
    )
    run_noise_templates = params["run_noise_templates"] and (
        needs_noise_templates or params["overwrite"]
    )
    run_mean_waveforms = params["run_mean_waveforms"] and (
        needs_mean_waveforms or params["overwrite"]
    )
    run_quality_metrics = params["run_quality_metrics"] and (
        needs_quality_metrics or params["overwrite"]
    )

    run_name, gate = run_dir.split("_g")
    info = {
        "ks_ver": ks_ver,
        "npx_directory": npx_directory,
        "run_name": run_name,
        "gate_index": gate,
        "probes": ",".join(probe_ids),
    }
    info = {**info, **params}
    info["run_CatGT"] = run_catgt
    info["runTPrime"] = run_tprime
    info["run_kilosort"] = run_kilosort
    info["run_kilosort_postprocessing"] = run_kilosort_postprocessing
    info["run_noise_templates"] = run_noise_templates
    info["run_mean_waveforms"] = run_mean_waveforms
    info["run_quality_metrics"] = run_quality_metrics
    return info


def run_custom_metrics(ks_folder, args):
    paths = [
        os.path.join(ks_folder, "cluster_SNR_good.tsv"),
        os.path.join(ks_folder, "cluster_RP_conf.tsv"),
        os.path.join(ks_folder, "cluster_wf_shape.tsv"),
        os.path.join(ks_folder, "mean_waveforms.npy"),
    ]

    if not args["overwrite"] and all(os.path.exists(path) for path in paths):
        tqdm.write(f"Custom metrics already exist for {ks_folder}")
        return
    tqdm.write(f"Running custom metrics for {ks_folder}")
    args["KS_folder"] = ks_folder
    cilantropy.custom_metrics(args)


if __name__ == "__main__":
    # SET PARAMETERS ############################################
    params = {
        "folder": r"Z:\Psilocybin",
        "ks_ver": "4",
        "ecephys_params": {
            "overwrite": False,
            "run_CatGT": True,  # leave to True if you want to run processing on sorted catgt files
            "process_lf": True,
            "ni_present": False,
            "runTPrime": False,
            "run_kilosort": True,
            "run_kilosort_postprocessing": True,
            "run_noise_templates": False,
            "run_mean_waveforms": False,
            "run_quality_metrics": False,
        },
        "curator_params": {"overwrite": True},  # default
        "run_auto_curate": True,
        "auto_curate_params": {},  # default
        "run_merge": True,
        "merge_params": {
            "overwrite": True,
            "plot_merges": False,
            "max_spikes": 500,
            "auto_accept_merges": True,
        },  # default
        "run_post_merge_curation": True,
        "post_merge_curation_params": {},
    }
    reset = True
    processing_drive = "D:"
    ############################################################
    # ecephys_spike_sorting pipeline
    run_info = get_run_info(
        params["folder"], params["ks_ver"], params["ecephys_params"]
    )
    # sort by date
    run_info = sorted(run_info, key=lambda x: x["run_name"])

    # run ecephys_spike_sorting
    for info in tqdm(run_info, "Processing runs..."):
        f"Processing {info['run_name']}"
        # join run_info and ecephys_params
        sglx_pipeline.main(info)

    ks_folders = get_ks_folders(
        params["folder"], params["ks_ver"], catgt=params["ecephys_params"]["run_CatGT"]
    )
    # sort by date
    ks_folders = sorted(ks_folders)
    # remove any with old in the name
    ks_folders = [ks_folder for ks_folder in ks_folders if "old" not in ks_folder]
    pbar = tqdm(ks_folders, "Processing Kilosort folders...")
    for ks_folder in pbar:
        pbar.set_description(f"Processing {ks_folder}")

        # move data to D: then process
        ks_folder_orig = ks_folder
        old_drive = ks_folder.split(os.sep)[0]
        probe_folder = os.path.dirname(ks_folder)
        new_probe_folder = probe_folder.replace(old_drive, processing_drive)

        # if modification date is within 24 hours, skip
        # mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(ks_folder))
        # if (datetime.datetime.now() - mod_date).days < 2:
        #     tqdm.write(f"Skipping {ks_folder}")
        #     continue
        if not ks_folder.startswith(processing_drive):
            # if not os.path.exists(new_probe_folder):
            copy_folder_with_progress(probe_folder, new_probe_folder)
            ks_folder = ks_folder.replace(old_drive, processing_drive)
            if reset:
                jc_folder = f"{ks_folder}_jc" # TODO
                if os.path.exists(jc_folder):
                    # replace ks_folder with jc_folder
                    shutil.rmtree(ks_folder)
                    shutil.move(jc_folder, ks_folder)

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

        # transfer over ks_folder back to Z:
        if ks_folder != ks_folder_orig:
            shutil.copytree(ks_folder, ks_folder_orig, dirs_exist_ok=True)
            shutil.rmtree(new_probe_folder)
