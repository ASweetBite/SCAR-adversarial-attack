import random

import numpy as np


class GeneticAlgorithmOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        ga_cfg = config.get('genetic_algorithm', {}) if config else {}
        run_cfg = config.get('run_params', {}) if config else {}

        self.pop_size = ga_cfg.get('pop_size', 40)
        self.max_generations = run_cfg.get('iterations', 60)
        self.run_mode = run_cfg.get('run_mode', 'attack')

        self.stagnation_limit = ga_cfg.get('stagnation_threshold', 5)
        self.m_rate_min = ga_cfg.get('mutation_rate_min', 0.1)
        self.m_rate_max = ga_cfg.get('mutation_rate_max', 0.5)

    def _calculate_fitness(self, probs, original_pred):
        if self.mode != "binary":
            orig_idx = 0 if original_pred == -1 else original_pred
            orig_idx = min(orig_idx, len(probs) - 1 if isinstance(probs, (list, np.ndarray)) else 0)
            orig_prob = max(probs[orig_idx] if isinstance(probs, (list, np.ndarray)) else probs, 1e-9)
            return -orig_prob

        is_orig_vuln = (original_pred == 1)
        p_safe = float(probs[0])
        p_vuln = float(probs[1]) if len(probs) > 1 else 1.0 - p_safe

        orig_prob = max(p_vuln if is_orig_vuln else p_safe, 1e-9)
        target_prob = max(p_safe if is_orig_vuln else p_vuln, 1e-9)
        return math.log(target_prob) - math.log(orig_prob)

    def _get_target_prob(self, probs, original_pred):
        if self.mode != "binary":
            orig_idx = 0 if original_pred == -1 else original_pred
            orig_idx = min(orig_idx, len(probs) - 1 if isinstance(probs, (list, np.ndarray)) else 0)
            orig_prob = probs[orig_idx] if isinstance(probs, (list, np.ndarray)) else float(probs)
            return 1.0 - orig_prob

        is_orig_vuln = (original_pred == 1)
        p_safe = float(probs[0])
        p_vuln = float(probs[1]) if len(probs) > 1 else 1.0 - p_safe
        return p_safe if is_orig_vuln else p_vuln

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None, rnns_best_seed=None, all_vars=None):
        """Executes a genetic algorithm to find the optimal adversarial variable substitutions."""

        # 1. 确定基因全集与边缘基因
        if all_vars is None:
            all_vars = target_vars

        background_vars = [v for v in all_vars if v not in target_vars]

        # 2. 构建非对称变异概率表 (Asymmetric Mutation Probabilities)
        mutation_probs = {}
        if variable_scores and target_vars:
            scores = [variable_scores.get(v, 0) for v in target_vars]
            min_s, max_s = min(scores), max(scores)
            for var in target_vars:
                score = variable_scores.get(var, 0)
                if max_s > min_s:
                    mutation_probs[var] = self.m_rate_min + (self.m_rate_max - self.m_rate_min) * (
                            (score - min_s) / (max_s - min_s))
                else:
                    mutation_probs[var] = (self.m_rate_min + self.m_rate_max) / 2
        else:
            for v in target_vars: mutation_probs[v] = 0.3

        # 🌟 为边缘基因赋予极低的探索性变异概率（例如 3%）
        # 这使得 GA 偶尔能摸奖，但大部分算力依然集中在 target_vars 上
        for var in background_vars:
            mutation_probs[var] = 0.03

        def get_safe_choice(var, pool, current_val=None):
            choices = list(set(pool)) if pool else []
            if not choices:
                return var
            if current_val and len(choices) > 1 and current_val in choices:
                choices.remove(current_val)
            return random.choice(choices)

        fitness_cache = {}
        best_code, best_fitness, best_probs, best_pred = code, float('-inf'), None, original_pred
        stagnation_counter = 0

        # --- 初始化种群 (此时染色体长度为 len(all_vars)) ---
        population = [{var: var for var in all_vars}]  # 1. 保留完全不突变的原始基因

        # 2. 注入 RNNS 精英种子
        if rnns_best_seed:
            seed_ind = {var: rnns_best_seed.get(var, var) for var in all_vars}
            population.append(seed_ind)

        # 3. 填满剩余种群
        while len(population) < self.pop_size:
            ind = {}
            for v in all_vars:
                # 初始种群生成时，靶点高频突变，边缘基因低频突变
                if v in target_vars and random.random() < 0.8:
                    ind[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                elif v in background_vars and random.random() < 0.1:
                    ind[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                else:
                    ind[v] = v
            population.append(ind)

        print(f"\n--- 🧬 GA 初始化完成 (种群: {self.pop_size}, 核心基因: {len(target_vars)}, 边缘基因: {len(background_vars)}) ---")

        for gen in range(self.max_generations):
            evaluated = []
            codes_to_predict = []
            keys_to_predict = []

            previous_best_fitness = best_fitness

            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                if cache_key not in fitness_cache:
                    try:
                        mutated_code = self.rename_fn(code, rename_map)
                        if mutated_code:
                            codes_to_predict.append(mutated_code)
                            keys_to_predict.append(cache_key)
                    except Exception:
                        fitness_cache[cache_key] = (float('-inf'), original_pred, code, None)

            if codes_to_predict:
                batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)
                for i in range(len(codes_to_predict)):
                    probs = batch_probs[i]
                    pred = batch_preds[i]
                    fitness = self._calculate_fitness(probs, original_pred)
                    fitness_cache[keys_to_predict[i]] = (fitness, pred, codes_to_predict[i], probs)

            # 记录最优
            generation_best_fitness = float('-inf')
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                if cache_key in fitness_cache:
                    fitness, pred, mutated_code, probs = fitness_cache[cache_key]
                    if probs is not None:
                        evaluated.append((ind, fitness, pred, mutated_code, probs))

                        if fitness > generation_best_fitness:
                            generation_best_fitness = fitness

                        if fitness > best_fitness:
                            best_fitness, best_code, best_probs, best_pred = fitness, mutated_code, probs, pred
                            current_target_prob = self._get_target_prob(probs, original_pred)
                            print(f"  [Gen {gen + 1:02d}] 🌟 突破! 适应度: {fitness:.4f} | 目标概率: {current_target_prob:.2%} | 预测: {pred}")

                        if pred != original_pred and self.run_mode == "attack":
                            final_target_prob = self._get_target_prob(probs, original_pred)
                            print(f"\n🎉 攻击成功！在第 {gen + 1} 代突破防线。最终目标概率: {final_target_prob:.2%}")
                            return True, mutated_code, probs, pred

            if best_probs is not None:
                current_target_prob = self._get_target_prob(best_probs, original_pred)
                print(f"[Gen {gen + 1:02d}/{self.max_generations}] 历史最优适应度: {best_fitness:.4f} | 目标概率: {current_target_prob:.2%}")

            # --- 繁衍逻辑 (交叉与突变现在覆盖全基因段) ---
            unique_evaluated = []
            seen_genes = set()
            for ind_tuple in evaluated:
                gene_signature = frozenset(ind_tuple[0].items())
                if gene_signature not in seen_genes:
                    seen_genes.add(gene_signature)
                    unique_evaluated.append(ind_tuple)

            if best_fitness <= previous_best_fitness + 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            unique_evaluated.sort(key=lambda x: x[1], reverse=True)

            # 停滞重启机制 (Restart)
            if stagnation_counter >= self.stagnation_limit:
                best_elite = unique_evaluated[0][0] if unique_evaluated else population[0]
                population = [best_elite]
                while len(population) < self.pop_size:
                    ind = {}
                    for v in all_vars:
                        if random.random() < (0.8 if v in target_vars else 0.1):
                            ind[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                        else:
                            ind[v] = best_elite[v]
                    population.append(ind)
                stagnation_counter = 0
                continue

            num_elites = max(2, min(len(unique_evaluated), self.pop_size // 4))
            elites = [x[0] for x in unique_evaluated[:num_elites]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                if len(elites) >= 2:
                    p1, p2 = random.sample(elites, 2)
                else:
                    p1, p2 = elites[0], elites[0]

                # 交叉 (Crossover): 在全量变量上进行
                child = {v: (p1[v] if random.random() > 0.5 else p2[v]) for v in all_vars}

                # 突变 (Mutation): 依据动态/非对称概率字典进行触发
                for v in child:
                    if random.random() < mutation_probs.get(v, 0.03):
                        child[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v], current_val=child[v])

                new_pop.append(child)

            population = new_pop

        if best_probs is not None:
            final_target_prob = self._get_target_prob(best_probs, original_pred)
            print(f"\n⚠️ 攻击结束。未能改变模型预测。最终目标概率峰值: {final_target_prob:.2%}")

        return (best_pred != original_pred), best_code, best_probs, best_pred


class GreedyOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        self.run_mode = run_cfg.get('run_mode', 'attack')

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        """Executes a sequential greedy search to apply variable substitutions and bypass model defenses."""
        if variable_scores:
            sorted_vars = sorted(target_vars, key=lambda v: variable_scores.get(v, 0), reverse=True)
        else:
            sorted_vars = target_vars

        current_code = code
        current_best_probs = None
        current_best_pred = original_pred
        overall_best_fitness = float('-inf')
        overall_best_code = code

        for var in sorted_vars:
            candidates = list(set(subs_pool.get(var, [])))
            if not candidates:
                continue

            codes_to_predict = []
            for cand in candidates:
                if cand == var:
                    continue
                try:
                    temp_code = self.rename_fn(current_code, {var: cand})
                    if temp_code:
                        codes_to_predict.append((cand, temp_code))
                except Exception:
                    continue

            if not codes_to_predict:
                continue

            candidate_strings = [item[1] for item in codes_to_predict]

            batch_probs, batch_preds = self.model_zoo.batch_predict(candidate_strings, self.target_model)

            best_var_fitness = float('-inf')
            best_var_code = None
            best_var_probs = None
            best_var_pred = None

            for i in range(len(codes_to_predict)):
                probs = batch_probs[i]
                pred = batch_preds[i]

                orig_idx = 0 if original_pred == -1 else original_pred

                if orig_idx >= len(probs):
                    orig_idx = len(probs) - 1

                orig_prob = max(probs[orig_idx], 1e-9)

                if self.mode == "binary":
                    target_idx = 1 if original_pred == -1 else 0

                    if target_idx >= len(probs):
                        target_idx = len(probs) - 1

                    target_prob = max(probs[target_idx], 1e-9)

                    fitness = math.log(target_prob) - math.log(orig_prob)
                else:
                    fitness = -orig_prob

                if fitness > best_var_fitness:
                    best_var_fitness = fitness
                    best_var_code = candidate_strings[i]
                    best_var_probs = probs
                    best_var_pred = pred

            if best_var_code and best_var_fitness > float('-inf'):
                current_code = best_var_code
                current_best_probs = best_var_probs
                current_best_pred = best_var_pred

                if best_var_fitness > overall_best_fitness:
                    overall_best_fitness = best_var_fitness
                    overall_best_code = best_var_code

                if current_best_pred != original_pred and self.run_mode == "attack":
                    verify_probs, verify_pred = self.model_zoo.predict(current_code, self.target_model)

                    if verify_pred != original_pred:
                        return True, current_code, verify_probs, verify_pred
                    else:
                        current_best_pred = verify_pred

        final_probs, final_pred = self.model_zoo.predict(overall_best_code, self.target_model)
        is_success = (final_pred != original_pred)

        return is_success, overall_best_code, final_probs, final_pred


import math
from typing import List, Dict, Tuple


class BeamSearchOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        beam_cfg = config.get('beam_params', {}) if config else {}
        self.run_mode = run_cfg.get('run_mode', 'attack')

        self.beam_size = beam_cfg.get('beam_size', 3)
        self.cand_chunk_size = beam_cfg.get('cand_chunk_size', 10)

        # ==========================================================
        # Beam 早停策略：可配置
        #   none/disabled/off/false : 不早停，遍历当前变量全部候选
        #   dynamic                 : 保留原逻辑：候选 chunk 有显著提升且变量数充足才早停
        #   gain                    : 只要候选 chunk 有显著提升就早停
        #   patience                : 连续若干 chunk 没有显著提升才早停
        # ==========================================================
        self.early_stop_delta = beam_cfg.get('early_stop_delta', 0.3)
        self.early_stop_strategy = str(
            beam_cfg.get('beam_early_stop_strategy', beam_cfg.get('early_stop_strategy', 'dynamic'))
        ).lower()
        self.early_stop_patience = int(beam_cfg.get('beam_early_stop_patience', 2))
        self.early_stop_min_valid_vars = int(beam_cfg.get('beam_early_stop_min_valid_vars', 3))

        self.enable_ast_check = bool(beam_cfg.get('beam_enable_ast_check', True))
        self._warned_missing_analyzer = False

    def _calculate_fitness(self, probs: List[float], original_pred: int) -> float:
        orig_idx = 0 if original_pred == -1 else original_pred
        orig_idx = min(orig_idx, len(probs) - 1)
        orig_prob = max(probs[orig_idx], 1e-9)

        if self.mode == "binary":
            target_idx = 1 if orig_idx == 0 else 0
            target_idx = min(target_idx, len(probs) - 1)
            target_prob = max(probs[target_idx], 1e-9)
            return math.log(target_prob) - math.log(orig_prob)

        other_probs = [p for i, p in enumerate(probs) if i != orig_idx]
        max_other_prob = max(other_probs) if other_probs else 1e-9
        max_other_prob = max(max_other_prob, 1e-9)
        return math.log(max_other_prob) - math.log(orig_prob)

    def _should_stop_after_chunk(self, chunk_best_fitness_gain: float, valid_var_count: int,
                                 bad_chunk_count: int) -> bool:
        strategy = self.early_stop_strategy

        if strategy in {"none", "disabled", "disable", "off", "false", "0"}:
            return False

        if strategy == "gain":
            return chunk_best_fitness_gain >= self.early_stop_delta

        if strategy == "patience":
            if valid_var_count < self.early_stop_min_valid_vars:
                return False
            return bad_chunk_count >= self.early_stop_patience

        # 默认 dynamic：兼容你原来的“有明显提升就停，但变量数太少时不停”。
        if valid_var_count >= self.early_stop_min_valid_vars:
            return chunk_best_fitness_gain >= self.early_stop_delta
        return False

    def run(self, code: str, original_pred: int, target_vars: List[str], subs_pool: Dict[str, List[str]],
            variable_scores: Dict[str, float] = None):

        if variable_scores:
            sorted_vars = sorted(target_vars, key=lambda v: variable_scores.get(v, 0), reverse=True)
        else:
            sorted_vars = target_vars

        query_cache = {}

        def _get_predictions(codes_to_predict: List[str]) -> Tuple[List[List[float]], List[int]]:
            if not codes_to_predict:
                return [], []

            uncached_codes = [c for c in codes_to_predict if c not in query_cache]

            if uncached_codes:
                batch_probs, batch_preds = self.model_zoo.batch_predict(uncached_codes, self.target_model)
                for c, p, pred in zip(uncached_codes, batch_probs, batch_preds):
                    query_cache[c] = (p, pred)

            cached_results = [query_cache[c] for c in codes_to_predict]
            probs_list = [res[0] for res in cached_results]
            preds_list = [res[1] for res in cached_results]
            return probs_list, preds_list

        # 基线预测。
        init_probs, init_preds = _get_predictions([code])
        orig_probs = init_probs[0]
        orig_pred = init_preds[0]

        initial_fitness = self._calculate_fitness(orig_probs, original_pred)

        beam = [(initial_fitness, code, orig_probs, orig_pred)]
        overall_best_fitness = initial_fitness
        overall_best_code = code

        valid_var_count = len([v for v in sorted_vars if subs_pool.get(v, [])])

        for var in sorted_vars:
            candidates = subs_pool.get(var, [])
            if not candidates:
                continue

            new_beam_candidates = []

            for curr_fitness, curr_code, curr_probs, curr_pred in beam:
                new_beam_candidates.append((curr_fitness, curr_code, curr_probs, curr_pred))
                bad_chunk_count = 0

                for i in range(0, len(candidates), self.cand_chunk_size):
                    cand_chunk = candidates[i:i + self.cand_chunk_size]

                    codes_to_predict = []
                    for cand in cand_chunk:
                        if cand == var:
                            continue

                        try:
                            temp_code = self.rename_fn(curr_code, {var: cand})
                            if temp_code:
                                codes_to_predict.append(temp_code)
                        except Exception:
                            continue

                    if not codes_to_predict:
                        if self.early_stop_strategy == "patience":
                            bad_chunk_count += 1
                            if self._should_stop_after_chunk(0.0, valid_var_count, bad_chunk_count):
                                break
                        continue

                    batch_probs, batch_preds = _get_predictions(codes_to_predict)
                    chunk_best_fitness_gain = 0.0

                    for probs, pred, temp_code in zip(batch_probs, batch_preds, codes_to_predict):
                        fitness = self._calculate_fitness(probs, original_pred)
                        fitness_gain = fitness - curr_fitness

                        if fitness_gain > chunk_best_fitness_gain:
                            chunk_best_fitness_gain = fitness_gain

                        if fitness > overall_best_fitness:
                            overall_best_fitness = fitness
                            overall_best_code = temp_code

                        if pred != original_pred and self.run_mode == "attack":
                            verify_probs, verify_preds = _get_predictions([temp_code])
                            if verify_preds[0] != original_pred:
                                return True, temp_code, verify_probs[0], verify_preds[0]

                        new_beam_candidates.append((fitness, temp_code, probs, pred))

                    if self.early_stop_strategy == "patience":
                        if chunk_best_fitness_gain < self.early_stop_delta:
                            bad_chunk_count += 1
                        else:
                            bad_chunk_count = 0

                    if self._should_stop_after_chunk(chunk_best_fitness_gain, valid_var_count, bad_chunk_count):
                        break

            unique_candidates = {}
            for state in new_beam_candidates:
                if state[1] not in unique_candidates or state[0] > unique_candidates[state[1]][0]:
                    unique_candidates[state[1]] = state

            sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[0], reverse=True)
            beam = sorted_candidates[:self.beam_size]

        final_probs_list, final_preds_list = _get_predictions([overall_best_code])
        final_probs = final_probs_list[0]
        final_pred = final_preds_list[0]

        is_success = final_pred != original_pred
        return is_success, overall_best_code, final_probs, final_pred