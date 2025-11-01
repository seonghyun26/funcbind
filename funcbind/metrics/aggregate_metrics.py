from collections import defaultdict
import torch
import numpy as np
import os
import wandb
import hydra


@hydra.main(config_path="../configs", config_name="aggregate_metrics", version_base=None)
def main(config):
    fb_pretrained_path_ref = config.get("fb_pretrained_path_ref", None)
    cdrs_aug_ref = config.get("cdrs_aug_ref", None)
    assert len(config["sweep_id_list"]) != 0
    run_ids, use_single_dataset = get_runs_in_sweep(config["sweep_id_list"], fb_pretrained_path_ref=fb_pretrained_path_ref, cdrs_aug_ref=cdrs_aug_ref)

    max_number = config["max_number"] if config["max_number"] is not None else range(200)
    all_numbers = []

    metrics = defaultdict(list)
    for fb_model in run_ids.keys():
        for run_id in run_ids[fb_model]:
            samples_path = f"{run_id}samples/"
            for target_folder in os.listdir(samples_path):
                if "target" in target_folder:
                    number_str = target_folder.split('_')[-1]
                    try:
                        number = int(number_str)
                    except:
                        print(f"Number not found in: {target_folder}")
                        continue
                    if number in max_number:
                        metrics_path = samples_path + target_folder + "/metrics.pt"
                        if not os.path.exists(metrics_path):
                            print(f"Metrics file not found: {metrics_path}")
                            continue
                        metrics_ = torch.load(metrics_path, weights_only=False)
                        if len(metrics_) == 0:
                            print(f"No metrics found in: {metrics_path}")
                            continue
                        if use_single_dataset == "sabdab" and metrics_['general_stats'][0][0]['ratio_with_seed_len'] == 0:
                            print(f"Ratio with seed len is 0 in: {metrics_path}")
                            continue
                        all_numbers.append(number)

                        for k, v in metrics_.items():
                            if k == "molecular_stats":
                                metrics["molecular_stats"].append(metrics_["molecular_stats"][0])
                            if k == "res_dock" and len(metrics_["res_dock"]) > 0:
                                metrics["res_dock"].append(metrics_["res_dock"][0])
                            if k == "res_posecheck" and len(metrics_["res_posecheck"]) > 0:
                                metrics["res_posecheck"].append(metrics_["res_posecheck"][0])
                            if k == "general_stats":
                                if number not in [5, 15, 21] and use_single_dataset == "sabdab":
                                    metrics_["general_stats"][0][0]["classifier_predictions_seed_length"] = np.nan
                                metrics["general_stats"].append(metrics_["general_stats"][0])
                            if k == "segment_reconstruction_scores":
                                metrics["segment_reconstruction_scores"].append(
                                    metrics_["segment_reconstruction_scores"][0]
                                )
                            if k == "res_mcpp_metrics":
                                metrics["res_mcpp_metrics"].append(metrics_["res_mcpp_metrics"][0])

        if use_single_dataset == "xdocked":
            print("n_targets", len(metrics['res_dock']))
            print("n_mols per target", np.nanmean([len(r) for r in metrics['res_dock']]))

        # molecular stats
        n_atoms = [x["n_atoms"] for r in metrics['molecular_stats'] for x in r]
        res = {
            "n_atoms_mean": np.nanmean(n_atoms).item(),
            "n_atoms_median": np.nanmedian(n_atoms).item(),
        }
        if use_single_dataset == "xdocked":
            # docking stats
            vina_score = []
            vina_min = []
            vina_dock = []
            n_errors = 0
            for i in range(len(metrics['res_dock'])):
                for j in range(len(metrics['res_dock'][i])):
                    try:
                        vina_score_ = metrics['res_dock'][i][j]["vina"]["score_only"][0]["affinity"]
                        if vina_score_ is not None and vina_score_ <= 15 and vina_score_ != np.nan:
                            vina_score.append(vina_score_)
                        vina_min_ = metrics['res_dock'][i][j]["vina"]["minimize"][0]["affinity"]
                        if vina_min_ is not None and vina_min_ <= 15 and vina_min_ != np.nan:
                            vina_min.append(vina_min_)
                        if "dock" in metrics['res_dock'][i][j]["vina"]:
                            vina_dock.append(metrics['res_dock'][i][j]["vina"]["dock"][0]["affinity"])
                    except Exception as e:
                        print("--error", e)
                        n_errors += 1
            print("n_errors", n_errors)
            res.update({
                "vina_score_mean": np.nanmean(vina_score).item(),
                "vina_score_median": np.nanmedian(vina_score).item(),
                "vina_min_mean": np.nanmean(vina_min).item(),
                "vina_min_median": np.nanmedian(vina_min).item(),
                "vina_dock_mean": np.nanmean(vina_dock).item(),
                "vina_dock_median": np.nanmedian(vina_dock).item(),
            })
            # validity
            validity = [r[0]['validity'] for r in metrics['molecular_stats']]
            # QED, SA, diversity
            qed = [x["chem_results"]["qed"] for r in metrics['res_dock'] for x in r]
            sa = [x["chem_results"]["sa"] for r in metrics['res_dock'] for x in r]
            diversity = [x for r in metrics['res_dock'] if len(r) > 0 for x in r[0]["diversity"]]
            res.update({
                "qed": np.nanmean(qed).item(),
                "sa": np.nanmean(sa).item(),
                "diversity": np.nanmean(diversity).item(),
                "validity": np.nanmean(validity).item(),
            })
            # posecheck
            clashes = []
            se = []
            for i in range(len(metrics['res_posecheck'])):
                if 'strain' in metrics['res_posecheck'][i]:
                    for s_val in metrics['res_posecheck'][i]['strain']:
                        if s_val is not None:
                            se.append(s_val)
                clashes += metrics['res_posecheck'][i]['clashes']
            res.update({
                "clashes_mean": np.nanmean(clashes).item(),
                "se_median": np.nanmedian(se).item(),
            })
        elif use_single_dataset == "sabdab":
            segment_seq_recovery_ = []
            segment_rmsd_ = []
            ddG_ = []
            validity_ = []
            uniqueness_ = []
            ratio_with_seed_len_ = []
            recovery_vs_seed_ = []
            classifier_predictions_seed_length_ = []
            for i in range(len(metrics['general_stats'])):
                uniqueness_.append(metrics['general_stats'][i][0]['uniqueness'])
                validity_.append(metrics['general_stats'][i][0]['validity'])
                ratio_with_seed_len_.append(metrics['general_stats'][i][0]['ratio_with_seed_len'])
                if "recovery_vs_seed" in metrics['general_stats'][i][0]:
                    recovery_vs_seed_.append(metrics['general_stats'][i][0]['recovery_vs_seed'])
                if "classifier_predictions_seed_length" in metrics['general_stats'][i][0]:
                    classifier_predictions_seed_length_.append(metrics['general_stats'][i][0]['classifier_predictions_seed_length'])
            n_errors = 0
            for i in range(len(metrics['segment_reconstruction_scores'])):
                for j in range(len(metrics['segment_reconstruction_scores'][i])):
                    try:
                        segment_seq_recovery_.append(metrics['segment_reconstruction_scores'][i][j]['segment_seq_recovery'])
                        segment_rmsd_.append(metrics['segment_reconstruction_scores'][i][j]['segment_rmsd'])
                        ddG_.append(metrics['segment_reconstruction_scores'][i][j]['ddG'] if "ddG" in metrics['segment_reconstruction_scores'][i][j] else metrics['segment_reconstruction_scores'][i][j]['ddG_cross'])
                    except Exception as e:
                        print("--error", e)
                        n_errors += 1
            res.update({"validity": np.nanmean(validity_).item()})
            res.update({"uniqueness": np.nanmean(uniqueness_).item()})
            res.update({"segment_seq_recovery": np.nanmean(segment_seq_recovery_)})
            res.update({"segment_rmsd": np.nanmean(segment_rmsd_)})
            res.update({"IMP": sum(1 for ddG in ddG_ if ddG < 0) / len(ddG_)})
            res.update({"ratio_with_seed_len": np.nanmean(ratio_with_seed_len_)})
            res.update({"recovery_vs_seed": np.nanmean(recovery_vs_seed_) if recovery_vs_seed_ else np.nan})
            res.update({"classifier_predictions_seed_length": np.nanmean(classifier_predictions_seed_length_)})

            # Print classifier predictions details for sabdab
            if classifier_predictions_seed_length_ is not None and len(classifier_predictions_seed_length_) > 0:
                print("classifier_predictions_seed_length details:", [x for x in classifier_predictions_seed_length_ if not (isinstance(x, (float, np.floating)) and np.isnan(x))])
            if recovery_vs_seed_:
                recovery_vs_seed_details = [x for x in recovery_vs_seed_ if not (isinstance(x, (float, np.floating)) and np.isnan(x))]
                print("recovery_vs_seed details:", recovery_vs_seed_details)
        else:
            # Flatten MCPP metrics double-nested lists and filter out None values
            lRMSD = [y for r in metrics['res_mcpp_metrics'] for x in r["lRMSD"] for y in x if y is not None]
            iRMSD = [y for r in metrics['res_mcpp_metrics'] for x in r["iRMSD"] for y in x if y is not None]
            TM_score = [y for r in metrics['res_mcpp_metrics'] for x in r["TM_score"] for y in x if y is not None]
            TS_full = [y for r in metrics['res_mcpp_metrics'] for x in r["TS_full"] for y in x if y is not None]
            Ratio_TSPR = [y for r in metrics['res_mcpp_metrics'] for x in r["Ratio_>0.5_TSPR"] for y in x if y is not None]

            res.update({"lRMSD": np.nanmean(lRMSD).item()})
            res.update({"iRMSD": np.nanmean(iRMSD).item()})
            res.update({"TM_score": np.nanmean(TM_score).item()})
            res.update({"TS_full": np.nanmean(TS_full).item()})
            res.update({"Ratio_>0.5_TSPR": np.nanmean(Ratio_TSPR).item()})

            # Flatten general stats lists and filter out None values
            cyclization = [x for r in metrics['general_stats'] for x in r['cyclization'] if x is not None]
            L_CAA = [x for r in metrics['general_stats'] for x in r['L_CAA'] if x is not None]
            D_CAA = [x for r in metrics['general_stats'] for x in r['D_CAA'] if x is not None]
            N_methyl = [x for r in metrics['general_stats'] for x in r['N_methyl'] if x is not None]
            known_NCAA = [x for r in metrics['general_stats'] for x in r['known_NCAA'] if x is not None]
            unknown_NCAA = [x for r in metrics['general_stats'] for x in r['unknown_NCAA'] if x is not None]
            unreasonable_AA = [x for r in metrics['general_stats'] for x in r['unreasonable_AA'] if x is not None]

            # For amino acid counts, compute sums and fractions (like in log_results)
            total_AA = sum(L_CAA) + sum(D_CAA) + sum(N_methyl) + sum(known_NCAA) + sum(unknown_NCAA)

            res.update({"cyclization_frac": sum(cyclization)/len(cyclization) if cyclization else 0})
            res.update({"L_CAA_frac": sum(L_CAA)/total_AA if total_AA > 0 else 0})
            res.update({"D_CAA_frac": sum(D_CAA)/total_AA if total_AA > 0 else 0})
            res.update({"N_methyl_frac": sum(N_methyl)/total_AA if total_AA > 0 else 0})
            res.update({"known_NCAA_frac": sum(known_NCAA)/total_AA if total_AA > 0 else 0})
            res.update({"unknown_NCAA_frac": sum(unknown_NCAA)/total_AA if total_AA > 0 else 0})
            res.update({"unreasonable_AA_frac": sum(unreasonable_AA)/total_AA if total_AA > 0 else 0})
        # print nice res

        print("==> Model", fb_model)
        print(str(sorted(all_numbers)).replace(' ', ''))
        print("len(all_numbers)", len(all_numbers))
        for k, v in res.items():
            if isinstance(v, (float, np.floating)):
                print(f"{k}: {v:.3f}")
            elif isinstance(v, int):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v}")
        print("------------")


def get_runs_in_sweep(sweep_id_list, fb_pretrained_path_ref=None, cdrs_aug_ref=None):
    """
    Retrieve all runs associated with a specific sweep.

    Args:
        entity (str): The wandb entity (username or team name).
        project (str): The wandb project name.
        sweep_id (str): The ID of the sweep.

    Returns:
        list: A list of wandb.Run objects.
    """
    grouped_runs = defaultdict(set)
    for sweep_id in sweep_id_list:
        try:
            runs = wandb.Api().runs(                
                f"{gt.getuser()}/funcbind",
                filters={"sweep": sweep_id},
                per_page=200,
            )
            for run_sweep in runs:
                # try:
                fb_pretrained_path = run_sweep.config.get('fb_pretrained_path', None)
                diffusion = "diffusion" in run_sweep.config["sampler"]["name"] if isinstance(run_sweep.config["sampler"], dict) else "diffusion" in run_sweep.config["sampler"]
                if diffusion:
                    save_name = "eval_test_diffusion"
                    step = run_sweep.config["sampler"]["N"]
                else:
                    save_name = "eval_test_multi_measurement_walkjump"
                    step = run_sweep.config.get('sampler.mcmc.steps', 100)
                cdrs_aug = run_sweep.config.get('dset.cdrs_aug', None)
                if (fb_pretrained_path_ref is None or (fb_pretrained_path_ref is not None and fb_pretrained_path == fb_pretrained_path_ref)) and (cdrs_aug_ref is None or (cdrs_aug_ref is not None and cdrs_aug == cdrs_aug_ref)):
                    grouped_runs[fb_pretrained_path].add(os.path.join(fb_pretrained_path, f"{save_name}_s{step}_{run_sweep.name}/"))
            use_single_dataset = run_sweep.config["dset"].get('use_single_dataset', "sabdab")
        except Exception as e:
            print(f"Error with sweep {sweep_id}: {e}")
    return grouped_runs, use_single_dataset


if __name__ == "__main__":
    main()
