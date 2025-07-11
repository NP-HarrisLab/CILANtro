import logging
import os
import re
import shutil
from datetime import datetime

import npx_utils as npx
import slay
from ecephys_spike_sorting import sglx_pipeline
from npx_utils import copy_folder_with_progress, get_ks_folders, is_run_folder
from tqdm import tqdm

import cilantropy
from cilantropy.curation import Curator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(r"D:\auto_curate.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_probe_id(probe_folder):
    name = os.path.basename(probe_folder)
    probe_id = re.search(r"(?<=imec)\d+", name)
    if probe_id is None:
        raise ValueError(f"Probe ID not found in {name}")
    return int(probe_id.group(0))


def get_processing_needs(folder, params, ks_ver):
    probe_folders = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ]

    probe_ids = [get_probe_id(probe_folder) for probe_folder in probe_folders]

    name = os.path.basename(probe_folders[0])
    run_dir = name.split("_imec")[0]

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
        if params["process_lf"]:
            params["process_lf"] = not os.path.exists(tcat_lf)

        if params["process_lf"] or not os.path.exists(tcat_ap):
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
    tprime = os.path.join(folder, f"{run_dir}_TPrime_cmd.txt")
    needs_tprime = not os.path.exists(tprime)

    run_name, gate = run_dir.split("_g")
    probes = ",".join(map(str, probe_ids))
    info = {
        "ks_ver": ks_ver,
        "npx_directory": os.path.dirname(folder),
        "run_name": run_name,
        "gate_index": gate,
        "probes": probes,
    }
    info = {**info, **params}
    info["run_CatGT"] = params.get("run_CatGT", False) and (
        needs_catgt or params["overwrite"]
    )
    info["runTPrime"] = params["runTPrime"] and (needs_tprime or params["overwrite"])
    info["run_kilosort"] = params["run_kilosort"] and (
        needs_kilosort or params["overwrite"]
    )
    info["run_kilosort_postprocessing"] = params["run_kilosort_postprocessing"] and (
        needs_kilosort_postprocessing or params["overwrite"]
    )
    info["run_noise_templates"] = params["run_noise_templates"] and (
        needs_noise_templates or params["overwrite"]
    )
    info["run_mean_waveforms"] = params["run_mean_waveforms"] and (
        needs_mean_waveforms or params["overwrite"]
    )
    info["run_quality_metrics"] = params["run_quality_metrics"] and (
        needs_quality_metrics or params["overwrite"]
    )
    return info


def get_run_info(folder, ks_ver, params, supercat_params=None):
    # TODO expand so it can handle a _gX folder as input
    run_info = []
    if npx.is_run_folder(folder):
        if supercat_params is not None and folder.startswith("supercat_"):
            info = get_processing_needs(folder, supercat_params, ks_ver)
            info["run_CatGT"] = False
            info["supercat"] = True
            info["run_supercat"] = False
            return [info]
        info = get_processing_needs(folder, params, ks_ver)
        if "acute_drug" in folder:
            info["maxsecs"] = 60 * 60  # 1 hour
        return [info]

    for root, dirs, _ in os.walk(folder):
        if "$RECYCLE.BIN" in root:
            continue
        # check if sub folders are run folders
        run_folders = [os.path.join(root, d) for d in dirs if is_run_folder(d)]
        if len(run_folders) > 0:
            if supercat_params is not None:
                # concatenate params and supercat_params
                supercat_params = {**params, **supercat_params}
                supercat_folders = [
                    run_folder
                    for run_folder in run_folders
                    if os.path.basename(run_folder).startswith("supercat")
                ]
                if len(supercat_folders) > 1:
                    raise ValueError(
                        "More than one supercat folder found. Please check."
                    )
                elif len(supercat_folders) == 1:
                    info = get_processing_needs(
                        supercat_folders[0], supercat_params, ks_ver
                    )
                    info["run_CatGT"] = False
                    info["supercat"] = True
                    info["run_supercat"] = False
                    run_info.append(info)
                    dirs[:] = []
                    continue
            # need to catgt
            orig_run_folders = [
                run_folder
                for run_folder in run_folders
                if not os.path.basename(run_folder).startswith("catgt")
                and not os.path.basename(run_folder).startswith("supercat")
            ]
            catgt_folders = []
            run_supercat = supercat_params is not None and len(orig_run_folders) > 1
            probes = ""
            gate_index = None
            for orig_run_folder in orig_run_folders:
                catgt_basename = "catgt_" + os.path.basename(orig_run_folder)
                catgt_folder = os.path.join(
                    os.path.dirname(orig_run_folder), catgt_basename
                )
                catgt_folders.append(catgt_folder)
                if os.path.exists(catgt_folder):
                    info = get_processing_needs(catgt_folder, params, ks_ver)
                else:
                    run_name, gate = os.path.basename(orig_run_folder).split("_g")
                    probe_folders = [
                        os.path.join(orig_run_folder, name)
                        for name in os.listdir(orig_run_folder)
                        if os.path.isdir(os.path.join(orig_run_folder, name))
                    ]
                    probe_ids = [
                        get_probe_id(probe_folder) for probe_folder in probe_folders
                    ]

                    info = {
                        "ks_ver": ks_ver,
                        "npx_directory": root,
                        "run_name": run_name,
                        "gate_index": gate,
                        "probes": ",".join(map(str, probe_ids)),
                    }
                    info = {**info, **params}
                # TODO make this better
                if run_supercat:
                    info["run_kilosort"] = False
                    info["run_kilosort_postprocessing"] = False
                    info["curate"] = False
                if "acute_drug" in orig_run_folder:
                    info["maxsecs"] = 60 * 60  # 1 hour
                run_info.append(info)
                probes = info["probes"]
                gate_index = info["gate_index"]
            if run_supercat:
                # sort catgt_folders by name
                catgt_folders.sort()
                # combine with params
                info = supercat_params.copy()
                info["ks_ver"] = ks_ver
                info["run_CatGT"] = True
                info["run_name"] = os.path.basename(root)
                info["npx_directory"] = root
                info["probes"] = probes
                info["supercat"] = True
                info["run_supercat"] = True
                info["supercat_folders"] = catgt_folders
                info["gate_index"] = gate_index
                run_info.append(info)
            dirs[:] = []
            continue
    return run_info


if __name__ == "__main__":
    # SET PARAMETERS ############################################
    params = {
        "folders": [
            r"D:\Psilocybin\Cohort_2b\T16\20250522_T16_baseline1\catgt_20250522_T16_baseline1_g0",
        ],
        "ks_ver": "4",
        "ecephys_params": {
            "overwrite": False,
            "run_CatGT": True,
            "process_lf": True,
            "ni_present": False,
            # "ni_extract_string": "-xa=0,0,0,1,1,0 -xa=0,0,1,1,1,10 -xa=0,0,2,1,1,500 -xa=0,0,3,1,1,500",
            "runTPrime": False,
            "run_kilosort": True,
            "run_kilosort_postprocessing": True,
            "run_noise_templates": False,
            "run_mean_waveforms": False,
            "run_quality_metrics": False,
            "maxsecs": 15 * 60,
        },
        "run_supercat": True,
        "supercat_params": {
            "trim_edges": False,
        },
        "curator_params": {"overwrite": True},
        "run_auto_curate": True,
        "reset_auto_curate": True,  # reset auto curate if True
        "auto_curate_params": {
            "min_fr": 0.1,
            "min_snr": 2,
            "max_rp_viol": 0.1,
            "max_peaks": None,
            "max_troughs": None,
            "max_wf_dur": None,
            "min_spat_decay": None,
        },
        "run_merge": True,
        "merge_params": {
            "overwrite": True,
            "plot_merges": False,
            "max_spikes": 500,
            "auto_accept_merges": True,
        },  # default
        "run_post_merge_curation": True,
        "post_merge_curation_params": {"max_noise_cutoff": 5, "min_pr": 0.9},
        "delete": True,
    }
    logger.info(
        "Starting auto-curation at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    processing_drive = "D:"
    try:
        for folder in params["folders"]:
            orig_drive = folder.split(os.sep)[0]
            supercat_params = (
                params["supercat_params"] if params["run_supercat"] else None
            )
            ############################################################
            # ecephys_spike_sorting pipeline
            run_info = get_run_info(
                folder, params["ks_ver"], params["ecephys_params"], supercat_params
            )

            # run ecephys_spike_sorting
            for info in tqdm(run_info, "Processing runs..."):
                processes = [
                    info["run_CatGT"],
                    info["run_kilosort"],
                    info["runTPrime"],
                    info["run_kilosort_postprocessing"],
                    info["run_noise_templates"],
                    info["run_mean_waveforms"],
                    info["run_quality_metrics"],
                    params["run_auto_curate"],
                    params["run_merge"],
                    params["run_post_merge_curation"],
                    params["run_supercat"],
                ]
                if not any(processes):
                    tqdm.write(f"Nothing to process for {info['run_name']}")
                    continue
                tqdm.write(f"Processing {info['run_name']}")
                logger.info("\tProcessing %s", info["run_name"])
                # check if need to move data
                moved_folders = []
                run_supercat = info.get("run_supercat", False)
                supercat = info.get("supercat", False)
                if orig_drive != processing_drive:
                    if info["run_CatGT"] and not params["run_supercat"]:
                        # copy raw run folder over
                        run_folder = os.path.join(
                            info["npx_directory"],
                            f"{info['run_name']}_g{info['gate_index']}",
                        )
                        copy_folder_with_progress(
                            run_folder, run_folder.replace(orig_drive, processing_drive)
                        )
                        moved_folders.append(run_folder)
                    elif run_supercat:
                        # copy catgt folders over
                        for catgt_folder in info["supercat_folders"]:
                            copy_folder_with_progress(
                                catgt_folder,
                                catgt_folder.replace(orig_drive, processing_drive),
                            )
                            moved_folders.append(catgt_folder)
                    elif any(
                        [
                            info["run_kilosort"],
                            info["run_kilosort_postprocessing"],
                            info["run_noise_templates"],
                            info["run_mean_waveforms"],
                            info["run_quality_metrics"],
                            params["run_auto_curate"],
                            params["run_merge"],
                            params["run_post_merge_curation"],
                            params["run_supercat"],
                        ]
                    ):
                        if not supercat:
                            # copy catgt folder over
                            catgt_folder = os.path.join(
                                info["npx_directory"],
                                f"catgt_{info['run_name']}_g{info['gate_index']}",
                            )
                            copy_folder_with_progress(
                                catgt_folder,
                                catgt_folder.replace(orig_drive, processing_drive),
                            )
                            moved_folders.append(catgt_folder)
                        else:
                            # copy supercat folder over
                            supercat_folder = os.path.join(
                                info["npx_directory"],
                                f"supercat_{info['run_name']}_g{info['gate_index']}",
                            )
                            copy_folder_with_progress(
                                supercat_folder,
                                supercat_folder.replace(orig_drive, processing_drive),
                            )
                            moved_folders.append(supercat_folder)
                    info["npx_directory"] = info["npx_directory"].replace(
                        orig_drive, processing_drive
                    )

                # run ecephys pipeline
                sglx_pipeline.main(info)  # multi-run pipeline
                logger.info("\t\tSGLX complete")

                if run_supercat:
                    # copy over supercat folder
                    supercat_folder = os.path.join(
                        info["npx_directory"],
                        f"supercat_{info['run_name']}_g{info['gate_index']}",
                    )
                    copy_folder_with_progress(
                        supercat_folder,
                        supercat_folder.replace(processing_drive, orig_drive),
                    )
                    logger.info("\t\tCopied SGLX files to original drive")
                elif info["run_CatGT"]:
                    # copy over catgt folder
                    catgt_folder = os.path.join(
                        info["npx_directory"],
                        f"catgt_{info['run_name']}_g{info['gate_index']}",
                    )
                    copy_folder_with_progress(
                        catgt_folder, catgt_folder.replace(processing_drive, orig_drive)
                    )
                    logger.info("\tCopied SGLX files to original drive")
                probe_folders = [
                    os.path.join(
                        info["npx_directory"],
                        f"{info['run_name']}_g{info['gate_index']}",
                        f"{info['run_name']}_g{info['gate_index']}_imec{probe_id}",
                    )
                    for probe_id in info["probes"].split(",")
                ]
                supercat_probe_folders = [
                    os.path.join(
                        info["npx_directory"],
                        f"supercat_{info['run_name']}_g{info['gate_index']}",
                        f"{info['run_name']}_g{info['gate_index']}_imec{probe_id}",
                    )
                    for probe_id in info["probes"].split(",")
                ]
                # keep only folders that exist
                supercat_probe_folders = [
                    folder
                    for folder in supercat_probe_folders
                    if os.path.exists(folder)
                ]
                catgt_probe_folders = [
                    os.path.join(
                        info["npx_directory"],
                        f"catgt_{info['run_name']}_g{info['gate_index']}",
                        f"{info['run_name']}_g{info['gate_index']}_imec{probe_id}",
                    )
                    for probe_id in info["probes"].split(",")
                ]
                # keep only folders that exist
                catgt_probe_folders = [
                    folder for folder in catgt_probe_folders if os.path.exists(folder)
                ]
                if len(supercat_probe_folders) > 0:
                    probe_folders = supercat_probe_folders
                elif len(catgt_probe_folders) > 0:
                    probe_folders = catgt_probe_folders
                else:
                    probe_folders = probe_folders

                for probe_folder in probe_folders:
                    probe_id = int(probe_folder.split("imec")[-1])
                    logger.info("\t\tProcessing probe %d", probe_id)
                    ks_folder_proc = os.path.join(
                        probe_folder, f"imec{probe_id}_ks{info['ks_ver']}"
                    )
                    ks_folder_orig = ks_folder_proc.replace(
                        processing_drive, orig_drive
                    )

                    if (
                        info["run_kilosort"]
                        and not info["run_kilosort_postprocessing"]
                        and not os.path.exists(ks_folder_proc + "_orig")
                    ):
                        copy_folder_with_progress(
                            ks_folder_proc, ks_folder_proc + "_orig"
                        )
                        copy_folder_with_progress(
                            ks_folder_proc, ks_folder_orig + "_orig"
                        )
                    if params["reset_auto_curate"] and os.path.exists(
                        ks_folder_proc + "_jc"
                    ):
                        copy_folder_with_progress(
                            ks_folder_proc + "_jc", ks_folder_proc, overwrite=True
                        )
                        logger.info("\t\t\tReset ks folder")
                    run_curation = info.get("curate", True)
                    if run_curation:
                        with Curator(
                            ks_folder_proc, **params["curator_params"]
                        ) as curator:
                            if (
                                info["run_kilosort"]
                                and info["run_kilosort_postprocessing"]
                            ):
                                copy_folder_with_progress(
                                    ks_folder_proc,
                                    f"{ks_folder_proc}_jc",
                                    overwrite=True,
                                )
                                copy_folder_with_progress(
                                    ks_folder_proc + "_jc",
                                    ks_folder_orig + "_jc",
                                    overwrite=True,
                                )
                            if params["run_auto_curate"]:
                                curator.auto_curate(params["auto_curate_params"])
                                logger.info("\t\t\tRan auto-curation")
                            if params["run_merge"]:
                                if params["merge_params"][
                                    "overwrite"
                                ] and os.path.exists(
                                    os.path.join(
                                        ks_folder_proc, "automerge", "new2old.json"
                                    )
                                ):
                                    shutil.rmtree(
                                        os.path.join(ks_folder_proc, "automerge")
                                    )
                                if not os.path.exists(
                                    os.path.join(
                                        ks_folder_proc, "automerge", "new2old.json"
                                    )
                                ):
                                    slay.run.main(
                                        {
                                            "KS_folder": ks_folder_proc,
                                            **params["merge_params"],
                                        }
                                    )
                                    logger.info("\t\t\tRan merge")
                                else:
                                    tqdm.write("Merges already exists")
                                    logger.info("\t\t\tMerges already existed")

                            if params["run_post_merge_curation"]:
                                curator.post_merge_curation(
                                    params["post_merge_curation_params"]
                                )
                                logger.info("\t\t\tRan post-merge curation")

                    if orig_drive != processing_drive:
                        print("Copying ks folder")
                        copy_folder_with_progress(
                            ks_folder_proc, ks_folder_orig, overwrite=True
                        )
                        logger.info("\t\t\tCopied ks folder to original drive")

                if params["delete"] and (orig_drive != processing_drive):
                    for folder in moved_folders:
                        proc_folder = folder.replace(orig_drive, processing_drive)
                        shutil.rmtree(proc_folder)
                        tqdm.write(f"Deleted {proc_folder}")
                        logger.info("\t\tDeleted folders")
        logger.info(
            "Finished auto-curation at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise e
