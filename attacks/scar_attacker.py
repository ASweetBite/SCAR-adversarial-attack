import gc
import json
import os
import time
import csv
from typing import List, Dict

import numpy as np
import torch

from attacks.optimizers import GeneticAlgorithmOptimizer, GreedyOptimizer, BeamSearchOptimizer
from attacks.rankers import RNNS_Ranker
from utils.model_zoo import ModelZooQueryTracker


class SCARAttacker:
    def __init__(self, model_zoo, get_all_vars_fn, mlm_gen, llm_gen, rename_fn, mode: str, config: dict):
        # Initializes the attacker and parses hierarchical configuration parameters.
        self.model_zoo = ModelZooQueryTracker(model_zoo)
        self.model_names = self.model_zoo.model_names
        self.mode = mode
        self.config = config

        _global = config.get('global', {})
        run_params = config.get('run_params', {})

        cg_cfg = config.get('candidate_generation', {})
        lw_cfg = cg_cfg.get('lightweight', {})
        hw_cfg = cg_cfg.get('heavyweight', {})

        attack_cfg = config.get('attack', {})
        scar_cfg = attack_cfg.get('scar', {})

        self.result_dir = _global.get('result_dir', "./results")
        self.run_mode = run_params.get('run_mode', 'attack')
        self.optimizer_type = str(attack_cfg.get('algorithm', 'beam')).lower()

        self.mlm_batch_size = cg_cfg.get('mlm_batch_size', 4)

        self.top_k_mlm = lw_cfg.get('top_k_mlm', 60)
        self.mlm_top_n_keep = lw_cfg.get('top_n_keep', 50)

        self.llm_top_m = hw_cfg.get('top_m', 25)

        self.top_k = scar_cfg.get('top_k', 5)
        self.total_quota = scar_cfg.get('total_quota', 50)
        self.llm_probe_quota = scar_cfg.get('llm_probe_quota', 4)
        self.max_llm_enrich_attempts = scar_cfg.get('max_llm_enrich_attempts', 2)
        self.rerank_after_llm_enrich = scar_cfg.get('rerank_after_llm_enrich', True)

        self.get_all_vars_fn = get_all_vars_fn
        self.mlm_gen = mlm_gen
        self.llm_gen = llm_gen
        self.rename_fn = rename_fn

        self.attack_logs = []

    def _log(self, message=""):
        # Appends message to the local attack execution log and outputs it to console.
        print(message)
        self.attack_logs.append(message)

    def _merge_candidate_pools(self, mlm_pool: dict, llm_pool: dict, final_quota: int) -> dict:
        # Merges MLM and LLM candidate replacement pools safely.
        final_pool = {}
        all_vars = set(mlm_pool.keys()).union(set(llm_pool.keys()))

        for var in all_vars:
            llm_cands = llm_pool.get(var, [])
            mlm_cands = mlm_pool.get(var, [])
            merged_cands = list(llm_cands)

            if len(merged_cands) < final_quota:
                for cand in mlm_cands:
                    if cand not in merged_cands:
                        merged_cands.append(cand)
                        if len(merged_cands) >= final_quota:
                            break

            final_pool[var] = merged_cands[:final_quota]
        return final_pool

    def attack(self, dataset: List[Dict]):
        # Orchestrates the multi-stage attack optimization on the provided dataset.
        self.attack_logs = []

        stats = {atk: {vic: {"total": 0, "fooled": 0, "success_queries": []} for vic in self.model_names} for atk in
                 self.model_names}
        storage_orig = {m: [] for m in self.model_names}
        storage_adv = {m: [] for m in self.model_names}

        total_valid_sample_time = 0.0
        valid_sample_count = 0
        model_time_stats = {m: 0.0 for m in self.model_names}
        model_valid_counts = {m: 0 for m in self.model_names}
        shared_prep_time = 0.0

        rankers = {m: RNNS_Ranker(self.model_zoo, m, self.rename_fn) for m in self.model_names}
        optimizers = {}
        for m in self.model_names:
            opt_kwargs = {"model_zoo": self.model_zoo, "target_model": m, "rename_fn": self.rename_fn,
                          "mode": self.mode, "config": self.config}
            if self.optimizer_type == "greedy":
                optimizers[m] = GreedyOptimizer(**opt_kwargs)
            elif self.optimizer_type == "beam":
                optimizers[m] = BeamSearchOptimizer(**opt_kwargs)
            elif self.optimizer_type == "ga":
                optimizers[m] = GeneticAlgorithmOptimizer(**opt_kwargs)

        for idx, sample in enumerate(dataset):
            self._log(f"\n=========================================")
            self._log(f"🎯· [Sample {idx}] Started processing")
            self._log(f"=========================================")
            t_sample_start = time.time()
            t_shared_start = time.time()

            code = sample["code"]
            ground_truth = sample.get("label")
            orig_predictions = {}
            has_correct_pred = False

            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}
                if pred == ground_truth: has_correct_pred = True

            if not has_correct_pred:
                self._log(f"    - Skipping: All models predicted incorrectly.")
                continue

            variables = self.get_all_vars_fn(code)
            if not variables: continue

            code_bytes = code.encode("utf-8")
            analyzer = self.mlm_gen.analyzer

            t_ast_start = time.time()
            full_identifiers = analyzer.extract_identifiers(code_bytes)
            batch_tasks = []

            for var in variables:
                if var not in full_identifiers: continue
                is_callable_or_class = all(
                    occ.get("entity_type") in ["function", "method", "class"] for occ in full_identifiers[var])

                if is_callable_or_class:
                    target_code_str = code
                else:
                    try:
                        target_code_str = analyzer.get_folded_code(code_bytes, var)
                    except Exception:
                        target_code_str = code

                batch_tasks.append({
                    "target_name": var, "code_str": target_code_str,
                    "full_code_str": code, "full_identifiers": full_identifiers
                })

            self._log(f"    - AST Setup: {time.time() - t_ast_start:.2f}s")

            t_mlm_start = time.time()
            mlm_full_pool = {}

            for i in range(0, len(batch_tasks), self.mlm_batch_size):
                chunk = batch_tasks[i:i + self.mlm_batch_size]
                try:
                    chunk_pool = self.mlm_gen.generate_candidates(
                        chunk,
                        top_k_mlm=self.top_k_mlm,
                        top_n_keep=self.mlm_top_n_keep
                    )
                    mlm_full_pool.update(chunk_pool)
                finally:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            self._log(f"    - MLM Generation: {time.time() - t_mlm_start:.2f}s")

            variables = [v for v in variables if mlm_full_pool.get(v)]
            if not variables: continue

            batch_tasks_by_var = {task["target_name"]: task for task in batch_tasks}
            sample_llm_cache = {v: [] for v in variables}
            deep_enrich_attempts = {v: 0 for v in variables}

            t_probe_start = time.time()
            try:
                llm_probe_pool = self.llm_gen.generate_candidates(batch_tasks, target_quota=self.llm_probe_quota)
                for var, cands in llm_probe_pool.items():
                    if var in sample_llm_cache:
                        sample_llm_cache[var] = list(set(cands))
            finally:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            self._log(f"    - LLM Probe: {time.time() - t_probe_start:.2f}s")

            rnns_eval_pool = self._merge_candidate_pools(mlm_full_pool, sample_llm_cache, final_quota=self.total_quota)
            current_shared_time = time.time() - t_shared_start
            sample_attacked_by_any = False

            for atk_model in self.model_names:
                t_atk_model_start = time.time()

                orig_pred = orig_predictions[atk_model]["pred"]
                if orig_pred != ground_truth:
                    self._log(f"    - {atk_model}: Initial prediction incorrect, skipping model attack.")
                    continue

                stats[atk_model][atk_model]["total"] += 1
                self.model_zoo.reset_counter()

                t_rnns_start = time.time()
                actual_top_k = min(self.top_k, len(variables))

                rnns_output = rankers[atk_model].rank_variables(
                    code=code, variables=variables.copy(), subs_pool=rnns_eval_pool,
                    reference_label=orig_pred, top_k=actual_top_k
                )

                if len(rnns_output) == 3:
                    ranked_vars, all_scores, rnns_best_seed = rnns_output
                else:
                    ranked_vars, all_scores = rnns_output
                self._log(f"    - First RNNS Rank: {time.time() - t_rnns_start:.2f}s")

                target_vars = ranked_vars[:actual_top_k]
                target_scores = {var: all_scores[var] for var in target_vars}

                t_enrich_start = time.time()
                tasks_to_generate = []

                for var in target_vars:
                    task = batch_tasks_by_var.get(var)
                    if not task: continue
                    cached_cands = sample_llm_cache.get(var, [])
                    attempts = deep_enrich_attempts.get(var, 0)

                    if len(cached_cands) < self.llm_top_m and attempts < self.max_llm_enrich_attempts:
                        tasks_to_generate.append(task)

                deep_enriched_this_round = False
                if tasks_to_generate:
                    missed_vars = [t['target_name'] for t in tasks_to_generate]
                    try:
                        new_llm_pool = self.llm_gen.generate_candidates(tasks_to_generate, target_quota=self.llm_top_m)
                        for var in missed_vars: deep_enrich_attempts[var] = deep_enrich_attempts.get(var, 0) + 1
                        for var, cands in new_llm_pool.items():
                            old_cands = sample_llm_cache.get(var, [])
                            merged = list(set(old_cands + list(cands or [])))
                            if len(merged) > len(old_cands): deep_enriched_this_round = True
                            sample_llm_cache[var] = merged
                    finally:
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                self._log(f"    - LLM Deep Enrichment: {time.time() - t_enrich_start:.2f}s")

                final_subs_pool = self._merge_candidate_pools(mlm_full_pool, sample_llm_cache, self.total_quota)
                candidate_counts = {v: len(final_subs_pool.get(v, [])) for v in target_vars}

                if self.rerank_after_llm_enrich and deep_enriched_this_round and target_vars:
                    t_rerank_start = time.time()

                    rerank_output = rankers[atk_model].rank_variables(
                        code=code, variables=target_vars.copy(), subs_pool=final_subs_pool,
                        reference_label=orig_pred, top_k=len(target_vars)
                    )
                    if len(rerank_output) == 3:
                        target_vars, rerank_scores, rnns_best_seed = rerank_output
                    else:
                        target_vars, rerank_scores = rerank_output

                    target_scores = {var: rerank_scores.get(var, all_scores.get(var, 0.0)) for var in target_vars}
                    self._log(f"    - Second RNNS Re-rank: {time.time() - t_rerank_start:.2f}s")

                t_opt_start = time.time()
                run_kwargs = {
                    "code": code, "original_pred": orig_pred,
                    "target_vars": target_vars, "subs_pool": final_subs_pool,
                    "variable_scores": target_scores
                }
                if self.optimizer_type == "ga":
                    if rnns_best_seed: run_kwargs["rnns_best_seed"] = rnns_best_seed
                    run_kwargs["all_vars"] = ranked_vars
                    run_kwargs["variable_scores"] = all_scores
                if self.optimizer_type == "bo":
                    run_kwargs["rnns_best_seed"] = rnns_best_seed

                opt_results = optimizers[atk_model].run(**run_kwargs)
                is_success, adv_code, adv_probs, adv_pred = opt_results[:4]

                self._log(f"    - Optimization Run ({self.optimizer_type.upper()}): {time.time() - t_opt_start:.2f}s")

                queries_consumed = self.model_zoo.get_query_count()

                sample_record = {
                    "sample_index": idx, "original_code": code,
                    "adversarial_code": adv_code if is_success else "",
                    "ground_truth_label": ground_truth,
                    "original_prediction": orig_pred, "adversarial_prediction": adv_pred,
                    "is_success": is_success, "candidate_counts": json.dumps(candidate_counts, ensure_ascii=False),
                    "queries_consumed": queries_consumed,
                    "attack_time_seconds": round(time.time() - t_atk_model_start, 2)
                }
                storage_adv[atk_model].append(sample_record)

                if is_success:
                    stats[atk_model][atk_model]["fooled"] += 1
                    stats[atk_model][atk_model]["success_queries"].append(queries_consumed)
                    self._log(f"    ✅ Success | {orig_pred} -> {adv_pred} | Queries: {queries_consumed}")

                    for vic_model in self.model_names:
                        if vic_model == atk_model: continue
                        if orig_predictions[vic_model]["pred"] == ground_truth:
                            stats[atk_model][vic_model]["total"] += 1
                            _, vic_adv_pred = self.model_zoo.predict(adv_code, vic_model)
                            if vic_adv_pred != orig_predictions[vic_model]["pred"]:
                                stats[atk_model][vic_model]["fooled"] += 1
                else:
                    self._log(f"    ❌ Failed | Queries: {queries_consumed}")

                model_elapsed = time.time() - t_atk_model_start
                model_time_stats[atk_model] += model_elapsed
                model_valid_counts[atk_model] += 1
                sample_attacked_by_any = True

            sample_elapsed = time.time() - t_sample_start
            if sample_attacked_by_any:
                total_valid_sample_time += sample_elapsed
                valid_sample_count += 1
                shared_prep_time += current_shared_time

        self._log("\n" + "=" * 50)
        self._log("🎯 FINAL ATTACK SUMMARY")
        self._log("=" * 50)

        asr_matrix, avg_queries = {}, {}
        for atk_m in self.model_names:
            asr_matrix[atk_m] = {}
            success_queries = stats[atk_m][atk_m]["success_queries"]
            avg_q = round(np.mean(success_queries), 2) if success_queries else 0.0
            avg_queries[atk_m] = avg_q

            total_atk = stats[atk_m][atk_m]["total"]
            fooled_atk = stats[atk_m][atk_m]["fooled"]
            asr_atk = (fooled_atk / total_atk * 100) if total_atk > 0 else 0.0

            self._log(f"🛡️ Target Model: {atk_m.upper()}")
            self._log(f"   ► ASR (Attack Success Rate) : {asr_atk:.2f}% ({fooled_atk}/{total_atk})")
            self._log(f"   ► Avg. Queries (Success)    : {avg_q}")
            self._log("-" * 50)

            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                asr_matrix[atk_m][vic_m] = round(asr, 2)

        self._log("\n" + "=" * 50)
        self._log("⏱️ TIME STATISTICS (Valid Samples Only)")
        self._log("=" * 50)
        avg_sample_time = (total_valid_sample_time / valid_sample_count) if valid_sample_count > 0 else 0.0
        avg_shared_time = (shared_prep_time / valid_sample_count) if valid_sample_count > 0 else 0.0

        self._log(f"   ► Valid Attacked Samples    : {valid_sample_count}")
        self._log(f"   ► Avg. Total Time / Sample  : {avg_sample_time:.2f}s")
        self._log(f"   ► Avg. Shared Prep Time     : {avg_shared_time:.2f}s (AST, Global MLM, LLM Probe)")
        self._log("-" * 50)
        self._log("   [Breakdown by Target Model (RNNS + LLM Enrich + Optimizer)]")

        for m in self.model_names:
            m_count = model_valid_counts[m]
            avg_m_time = (model_time_stats[m] / m_count) if m_count > 0 else 0.0
            self._log(f"     * {m.upper():<12} | Valid attacks: {m_count:<3} | Avg Time: {avg_m_time:.2f}s")
        self._log("=" * 50)

        self.save_results(storage_orig, storage_adv)

        return asr_matrix, avg_queries

    def save_results(self, storage_orig, storage_adv):
        # Saves detailed attack results and execution logs to files.
        result_dir = self.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        log_filename = os.path.join(result_dir, f"attack_logs_{self.mode}_{int(time.time())}.txt")
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.attack_logs))
            print(f"[INFO] Successfully saved logs to: {log_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save logs to {log_filename}: {e}")

        for model in self.model_names:
            adv_data = storage_adv[model]
            if adv_data:
                adv_filename = f"adv_test_set_{model}_{self.mode}.csv"
                adv_path = os.path.join(result_dir, adv_filename)
                self._write_csv(adv_path, adv_data)

            if self.run_mode == "dataset" and storage_orig[model]:
                orig_filename = f"orig_dataset_{model}_{self.mode}.csv"
                orig_path = os.path.join(result_dir, orig_filename)
                self._write_csv(orig_path, storage_orig[model])

    def _write_csv(self, filename, data):
        # Helper function to write list of dictionaries to a CSV file.
        if not data: return
        try:
            fieldnames = list(data[0].keys())
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            print(f"[INFO] Saved {len(data)} detailed records to CSV: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV {filename}: {e}")

    def print_summary(self, stats):
        # Displays the final cross-model transferability ASR percentage matrix.
        self._log("\n" + "=" * 90)
        self._log("📊 FINAL CROSS-MODEL TRANSFERABILITY MATRIX (ASR %)")
        self._log("=" * 90)
        header = f"{'Attacker \\ Victim':<20} |"
        for m in self.model_names:
            header += f" {m:<13} |"
        self._log(header)
        self._log("-" * len(header))
        for atk_m in self.model_names:
            row = f"{atk_m:<20} |"
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                row += f" {asr:>11.2f}% |"
            self._log(row)
        self._log("=" * 90 + "\n")