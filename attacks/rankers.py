import gc
import torch
import heapq

import gc
import torch
import heapq
import random


class RNNS_Ranker:
    def __init__(self, model_zoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    def rank_variables(self, code, variables, subs_pool, reference_label,
                       test_sample_size=10, top_k=10, filter_short_vars=True,
                       guaranteed_head_size=2):  # [新增] 强制保留头部的词汇数量 (对应探针数量)

        oref_idx = 0 if reference_label == -1 else reference_label
        orig_prob = self.model_zoo.predict_label_conf(code, oref_idx, self.target_model)

        valid_vars = [v for v in variables if len(v) > 2] if filter_short_vars else variables
        mutation_tasks = []

        # 串行执行重命名，彻底规避 AST Parser 多线程崩溃风险
        for var in valid_vars:
            all_cands = subs_pool.get(var, [])
            if not all_cands:
                continue

            # =========================================================
            # [核心修复] 智能采样策略：头部绝对保留 + 尾部多样性采样
            # =========================================================
            if len(all_cands) <= test_sample_size:
                candidates = all_cands
            else:
                # 1. 提取头部高优先级词汇 (确保 LLM 探针绝对不被遗漏)
                head_cands = all_cands[:guaranteed_head_size]

                # 2. 从剩余的词汇 (主要是海量 MLM 兜底词) 中随机抽取
                # 这样做比死板地取前 8 个 MLM 词更好，能更全面地探测变量对不同乱码/相似词的敏感度分布
                tail_pool = all_cands[guaranteed_head_size:]
                sample_count = test_sample_size - len(head_cands)

                if len(tail_pool) >= sample_count:
                    tail_cands = random.sample(tail_pool, sample_count)
                else:
                    tail_cands = tail_pool

                candidates = head_cands + tail_cands

            for cand in candidates:
                if cand != var:
                    try:
                        renamed_code = self.rename_fn(code, {var: cand})
                        if renamed_code:
                            mutation_tasks.append((var, cand, renamed_code))
                    except Exception:
                        continue

        if not mutation_tasks:
            return [], {}, {}

        codes_to_predict = [task[2] for task in mutation_tasks]
        all_probs = []
        BATCH_SIZE = 16

        # 分块推理，彻底杜绝 OOM
        for i in range(0, len(codes_to_predict), BATCH_SIZE):
            chunk = codes_to_predict[i:i + BATCH_SIZE]
            chunk_probs, _ = self.model_zoo.batch_predict(chunk, self.target_model)
            all_probs.extend(chunk_probs)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        var_max_drop = {var: -float('inf') for var in valid_vars}
        var_best_cand = {var: var for var in valid_vars}

        for (var, cand, _), probs in zip(mutation_tasks, all_probs):
            prob_drop = orig_prob - probs[oref_idx]

            if prob_drop > var_max_drop[var]:
                var_max_drop[var] = prob_drop
                var_best_cand[var] = cand

        valid_scores = [(var, score) for var, score in var_max_drop.items() if score != -float('inf')]
        sorted_all_vars_with_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)

        ranked_vars = [var for var, _ in sorted_all_vars_with_scores]
        score_dict = {var: score for var, score in sorted_all_vars_with_scores}

        best_seeds = {var: var_best_cand[var] for var in ranked_vars if var_best_cand[var] != var}

        return ranked_vars, score_dict, best_seeds