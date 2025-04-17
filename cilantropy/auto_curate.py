import os
import re
import shutil
from datetime import datetime

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
            if d.startswith("catgt") or d.startswith("old") or "SvyPrb" in d:
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
        )
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
        "folder": r"Z:\Psilocybin\Cohort_2\T12\20250415_T12_baseline",
        "ks_ver": "4",
        "ecephys_params": {
            "overwrite": True,
            "run_CatGT": True,  # leave to True if you want to run processing on sorted catgt files
            "process_lf": True,
            "ni_present": False,
            # "ni_extract_string": "-xa=0,0,0,1,1,0 -xa=0,0,1,1,1,10 -xa=0,0,2,1,1,500 -xa=0,0,3,1,1,500",
            "runTPrime": False,
            "run_kilosort": True,
            "run_kilosort_postprocessing": True,
            "run_noise_templates": False,
            "run_mean_waveforms": False,
            "run_quality_metrics": False,
            "maxsecs": 15 * 60,  # 15 minutes
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
    processing_drive = "D:"
    orig_drive = params["folder"].split(os.sep)[0]
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

        # copy over to D: and process
        probe_folders = [
            os.path.join(
                info["npx_directory"],
                f"{info['run_name']}_g{info['gate_index']}",
                f"{info['run_name']}_g{info['gate_index']}_imec{probe_id}",
            )
            for probe_id in info["probes"].split(",")
        ]
        catgt_probe_folders = [
            os.path.join(
                info["npx_directory"],
                f"catgt_{info['run_name']}_g{info['gate_index']}",
                f"{info['run_name']}_g{info['gate_index']}_imec{probe_id}",
            )
            for probe_id in info["probes"].split(",")
        ]
        # if os.path.exists(probe_folders[0]):
        #     # check if already processed
        #     mod_time = datetime.fromtimestamp(os.path.getmtime(probe_folders[0]))
        #     if mod_time >= datetime(2025, 4, 1):
        #         continue
        copy_folders = probe_folders if info["run_CatGT"] else catgt_probe_folders
        for probe_folder in copy_folders:
            probe_id = int(probe_folder.split("imec")[-1])
            if orig_drive != processing_drive:
                new_probe_folder = probe_folder.replace(orig_drive, processing_drive)
                copy_folder_with_progress(probe_folder, new_probe_folder)

        info["npx_directory"] = info["npx_directory"].replace(
            orig_drive, processing_drive
        )

        # run ecephys pipeline
        sglx_pipeline.main(info)  # multi-run pipeline

        catgt_folder = os.path.join(
            info["npx_directory"].replace(processing_drive, orig_drive),
            f"catgt_{info['run_name']}_g{info['gate_index']}",
        )
        catgt_folder_proc = catgt_folder.replace(orig_drive, processing_drive)
        # copy over catgt folder
        if info["run_CatGT"] or not os.path.exists(catgt_folder):
            copy_folder_with_progress(catgt_folder_proc, catgt_folder, overwrite=True)

        for probe_folder in catgt_probe_folders:
            probe_id = int(probe_folder.split("imec")[-1])
            ks_folder_orig = os.path.join(
                probe_folder, f"imec{probe_id}_ks{info['ks_ver']}"
            )
            ks_folder_proc = ks_folder_orig.replace(orig_drive, processing_drive)

            if (
                params["ecephys_params"]["run_kilosort"]
                and params["ecephys_params"]["run_kilosort_postprocessing"]
            ):
                copy_folder_with_progress(
                    ks_folder_proc, f"{ks_folder_proc}_jc", overwrite=True
                )
                copy_folder_with_progress(
                    ks_folder_proc + "_jc", ks_folder_orig + "_jc", overwrite=True
                )
            elif not os.path.exists(ks_folder_proc + "_orig"):
                copy_folder_with_progress(ks_folder_proc, ks_folder_proc + "_orig")
                copy_folder_with_progress(ks_folder_proc, ks_folder_orig + "_orig")

            with Curator(ks_folder_proc, **params["curator_params"]) as curator:
                if params["run_auto_curate"]:
                    curator.auto_curate(params["auto_curate_params"])
                if params["run_merge"]:
                    if params["merge_params"]["overwrite"] and os.path.exists(
                        os.path.join(ks_folder_proc, "automerge", "new2old.json")
                    ):
                        shutil.rmtree(os.path.join(ks_folder_proc, "automerge"))
                    if not os.path.exists(
                        os.path.join(ks_folder_proc, "automerge", "new2old.json")
                    ):
                        slay.run.main(
                            {"KS_folder": ks_folder_proc, **params["merge_params"]}
                        )
                    else:
                        tqdm.write("Merges already exists")

                if params["run_post_merge_curation"]:
                    curator.post_merge_curation(params["post_merge_curation_params"])

            if orig_drive != processing_drive:
                print("Copying ks folder")
                copy_folder_with_progress(
                    ks_folder_proc, ks_folder_orig, overwrite=True
                )

        if orig_drive != processing_drive:
            shutil.rmtree(catgt_folder_proc)
