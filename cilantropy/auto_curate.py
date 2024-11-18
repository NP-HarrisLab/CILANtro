import os
import re
import shutil

import slay
from ecephys_spike_sorting import sglx_pipeline
from tqdm import tqdm

import cilantropy
from cilantropy.curation import Curator


def get_run_info(folder, ks_ver, params):
    """currently assumes gate 0-9 index. will process separately as well - need to update"""
    root_dir = os.path.abspath(folder)
    pattern = re.compile(r".*_g[0-9]$")
    if pattern.match(folder):
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
            if d.startswith("catgt"):
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
    for probe_folder in probe_folders:
        probe_id = probe_folder[-1]
        # catGT
        tcat_ap = os.path.join(
            probe_folder,
            f"{run_dir}_tcat.imec{probe_id}.ap.bin",
        )
        tcat_lf = os.path.join(
            probe_folder,
            f"{run_dir}_tcat.imec{probe_id}.lf.bin",
        )
        if params["process_lf"] and not os.path.exists(tcat_lf):
            needs_catgt = True
        if not os.path.exists(tcat_ap):
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
    params["run_CatGT"] = params["run_CatGT"] and (needs_catgt or params["overwrite"])
    params["runTPrime"] = params["runTPrime"] and (needs_tprime or params["overwrite"])
    params["run_kilosort"] = params["run_kilosort"] and (
        needs_kilosort or params["overwrite"]
    )
    params["run_kilosort_postprocessing"] = params["run_kilosort_postprocessing"] and (
        needs_kilosort_postprocessing or params["overwrite"]
    )
    params["run_noise_templates"] = params["run_noise_templates"] and (
        needs_noise_templates or params["overwrite"]
    )
    params["run_mean_waveforms"] = params["run_mean_waveforms"] and (
        needs_mean_waveforms or params["overwrite"]
    )
    params["run_quality_metrics"] = params["run_quality_metrics"] and (
        needs_quality_metrics or params["overwrite"]
    )
    if not any(
        [
            params["run_CatGT"],
            params["runTPrime"],
            params["run_kilosort"],
            params["run_kilosort_postprocessing"],
            params["run_noise_templates"],
            params["run_mean_waveforms"],
            params["run_quality_metrics"],
        ]
    ):
        return

    run_name, gate = run_dir.split("_g")
    info = {
        "ks_ver": ks_ver,
        "npx_directory": npx_directory,
        "run_name": run_name,
        "gate_index": gate,
        "probes": ",".join(probe_ids),
    }
    info = {**info, **params}
    return info


def get_ks_folders(root_dir, ks_ver):
    root_dir = os.path.abspath(root_dir)
    pattern = re.compile(r"imec\d_ks\d+")
    matching_folders = []
    for root, dirs, _ in os.walk(root_dir):
        if "$RECYCLE.BIN" in root:
            continue
        for dir in dirs:
            if pattern.match(dir):
                if dir.split("_")[-1] == f"ks{ks_ver}":
                    matching_folders.append(os.path.join(root, dir))
    return matching_folders


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
    print(f"Running custom metrics for {ks_folder}")
    args["KS_folder"] = ks_folder
    cilantropy.custom_metrics(args)


if __name__ == "__main__":
    # SET PARAMETERS ############################################
    params = {
        "folder": "E:\\T09\\20241022_T09_OF_Test1",  # 21 and Hab
        "ks_ver": "4",
        "ecephys_params": {
            "overwrite": False,
            "run_CatGT": True,
            "process_lf": True,
            "ni_present": False,
            "runTPrime": True,
            "run_kilosort": True,
            "run_kilosort_postprocessing": True,
            "run_noise_templates": False,
            "run_mean_waveforms": False,
            "run_quality_metrics": False,
        },
        "run_custom_metrics": False,  # no reason since run in Curator
        "custom_metrics_params": {
            "overwrite": False,
        },  # default
        "curator_params": {},  # default
        "run_auto_curate": True,
        "auto_curate_params": {"save": True},  # default
        "run_merge": True,
        "merge_params": {
            "overwrite": False,
            "plot_merges": False,
            "max_spikes": 500,
            "auto_accept_merges": True,
        },  # default
        "run_post_merge_curation": True,
        "post_merge_curation_params": {"save": True},
    }

    ############################################################
    # ecephys_spike_sorting pipeline
    run_info = get_run_info(
        params["folder"], params["ks_ver"], params["ecephys_params"]
    )
    # sort by date
    run_info = sorted(run_info, key=lambda x: x["run_name"])

    # run ecephys_spike_sorting
    for info in tqdm(run_info, "Processing runs..."):
        # join run_info and ecephys_params
        sglx_pipeline.main(info)

    ks_folders = get_ks_folders(params["folder"], params["ks_ver"])
    for ks_folder in tqdm(ks_folders):
        tqdm.write(f"Processing {ks_folder}")

        if params["run_custom_metrics"]:
            run_custom_metrics(ks_folder, params["custom_metrics_params"])
        curator = Curator(ks_folder, **params["curator_params"])

        if params["run_auto_curate"]:
            curator.auto_curate(params["auto_curate_params"])

        if params["run_merge"]:
            if params["merge_params"]["overwrite"] and os.path.exists(
                os.path.join(ks_folder, "automerge", "new2old.json")
            ):
                shutil.rmtree(os.path.join(ks_folder, "automerge"))
            if not os.path.exists(os.path.join(ks_folder, "automerge", "new2old.json")):
                slay.run.main({"KS_folder": ks_folder, **params["merge_params"]})
            else:
                print("Merges already exists")

        if params["run_post_merge_curation"]:
            curator.post_merge_curation(params["post_merge_curation_params"])
