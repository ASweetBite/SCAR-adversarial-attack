import re
import json
import torch
import torch.nn.functional as F
from typing import Dict, Any, List


class BaseCandidateGenerator:
    """
    统一的候选词生成基类，封装了所有 AST 提取、代码风格匹配、
    语义向量提取 (Embeddings) 以及过滤清洗漏斗机制。
    """

    def __init__(self, embedder, analyzer, config):
        self.embedder = embedder
        self.analyzer = analyzer
        self.config = config

        cg_cfg = self.config.get('candidate_generation', {})
        stats_path = cg_cfg.get('naming_stats_path', 'naming_stats.json')
        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

    def _detect_naming_style(self, name: str) -> str:
        if not name: return 'unknown'
        core_name = name.strip('_')
        if not core_name: return 'unknown'
        if '_' in core_name: return 'SCREAMING_SNAKE' if core_name.isupper() else 'snake_case'
        if core_name.islower(): return 'single_lower'
        if core_name.isupper(): return 'single_upper'
        if core_name[0].islower(): return 'camelCase'
        if core_name[0].isupper(): return 'PascalCase'
        return 'unknown'

    def _matches_style(self, original_style: str, candidate: str) -> bool:
        cand_style = self._detect_naming_style(candidate)
        if cand_style == original_style:
            return True
        if original_style == 'single_lower' and cand_style in ('snake_case', 'camelCase'):
            return True
        if original_style == 'single_upper' and cand_style == 'SCREAMING_SNAKE':
            return True
        if original_style in ('snake_case', 'camelCase', 'PascalCase') and cand_style == 'single_lower':
            return True
        if original_style == 'SCREAMING_SNAKE' and cand_style == 'single_upper':
            return True
        return False

    def _split_identifier(self, name: str):
        if '_' in name: return [p for p in name.split('_') if p], '_'
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
        if not parts or (len(parts) == 1 and parts[0] == name): return [name], ''
        return parts, 'camel'

    def _extract_local_context_ast(self, code_bytes: bytes, target_start: int, target_end: int, tree) -> tuple[
        str, str]:
        node = tree.root_node.descendant_for_byte_range(target_start, target_end)
        if not node:
            line_start = code_bytes.rfind(b'\n', 0, target_start) + 1
            line_end = code_bytes.find(b'\n', target_end)
            if line_end == -1: line_end = len(code_bytes)
            return (code_bytes[line_start:target_start].decode("utf-8", errors="replace"),
                    code_bytes[target_end:line_end].decode("utf-8", errors="replace"))

        statement_node = node
        stop_parent_types = {'compound_statement', 'translation_unit', 'function_definition', 'for_statement',
                             'while_statement', 'if_statement'}
        while statement_node.parent and statement_node.parent.type not in stop_parent_types:
            statement_node = statement_node.parent

        stmt_start = statement_node.start_byte
        stmt_end = statement_node.end_byte
        local_prefix = code_bytes[stmt_start:target_start].decode("utf-8", errors="replace")
        local_suffix = code_bytes[target_end:stmt_end].decode("utf-8", errors="replace")
        return local_prefix, local_suffix

    def _find_best_context_occurrence(self, code_bytes: bytes, occurrences: List[dict], tree) -> int:
        if len(occurrences) <= 1: return 0
        best_idx, max_score = 0, -1.0
        search_limit = min(len(occurrences), 10)
        for i in range(search_limit):
            occ = occurrences[i]
            local_prefix, local_suffix = self._extract_local_context_ast(code_bytes, occ['start'], occ['end'], tree)
            score = len(local_prefix) + len(local_suffix)
            if '(' in local_suffix or ',' in local_suffix: score += 100
            if any(k in local_prefix for k in ['if ', 'while ', 'for ', 'return ']): score += 80
            if re.search(r'=\s*(0|NULL|nullptr|false|true|\{\})\s*;', local_suffix): score -= 150
            if score > max_score:
                max_score = score
                best_idx = i
        return best_idx

    def _verify_ast_single(self, cand: str, ctx: dict) -> str | None:
        if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
            return None
        try:
            from utils.ast_tools import CodeTransformer
            CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'], {ctx['target_name']: cand},
                                               analyzer=self.analyzer)
            return cand
        except Exception:
            return None

    def _get_dynamic_threshold(self, target_name: str, cand: str, base_threshold: float) -> float:
        """轻量级模型中使用的动态阈值计算（结构相似度惩罚）"""
        target_lower = target_name.lower()
        cand_lower = cand.lower()

        if cand_lower.endswith(f"_{target_lower}") or cand_lower.startswith(f"{target_lower}_"):
            return min(0.99, base_threshold + 0.05)
        if target_lower in cand_lower:
            return min(0.99, base_threshold + 0.03)

        import Levenshtein
        if Levenshtein.distance(target_lower, cand_lower) <= 2:
            return min(0.99, base_threshold + 0.07)

        target_parts, target_sep = self._split_identifier(target_name)
        cand_parts, cand_sep = self._split_identifier(cand)

        if len(target_parts) > 1 and len(target_parts) == len(cand_parts) and target_sep == cand_sep:
            identical_count = sum(1 for t, c in zip(target_parts, cand_parts) if t.lower() == c.lower())
            if identical_count > 0:
                overlap_ratio = identical_count / len(target_parts)
                if overlap_ratio >= 0.5:
                    penalty = 0.02 + (overlap_ratio * 0.06)
                    if target_parts[-1].lower() != cand_parts[-1].lower(): penalty += 0.015
                    return min(0.99, base_threshold + penalty)
        return base_threshold

    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 256) -> torch.Tensor:
        """高性能的 Token 向量提取引擎 (兼容 Jina/UniXcoder 等)"""
        all_embeddings = []
        tokenizer = self.embedder.tokenizer
        full_texts = [p + v + s for p, v, s in zip(prefixes, var_names, suffixes)]
        device = self.embedder.device
        dtype = next(self.embedder.model.parameters()).dtype
        MAX_SEQ_LEN = 512

        for i in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[i: i + batch_size]
            batch_prefixes = prefixes[i: i + batch_size]
            batch_vars = var_names[i: i + batch_size]

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN,
                               return_offsets_mapping=True)
            offset_mapping = inputs.pop("offset_mapping")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
                outputs = self.embedder.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]

            for b_idx in range(len(batch_texts)):
                prefix_len, var_len = len(batch_prefixes[b_idx]), len(batch_vars[b_idx])
                target_start_char, target_end_char = prefix_len, prefix_len + var_len
                start_idx, end_idx = -1, -1

                for tok_idx, (char_start, char_end) in enumerate(offset_mapping[b_idx]):
                    if char_start == char_end == 0: continue
                    if start_idx == -1 and char_end > target_start_char:
                        start_idx = tok_idx
                    if char_end >= target_end_char:
                        end_idx = tok_idx + 1
                        break

                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    seq_len = inputs["attention_mask"][b_idx].sum().item()
                    start_idx, end_idx = 1, seq_len - 1

                pooled = last_hidden[b_idx, start_idx:end_idx, :].mean(dim=0)
                all_embeddings.append(pooled.to(torch.float32).cpu())

        return torch.stack(all_embeddings)

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx, use_dynamic_threshold=False,
                           log_prefix=""):
        """统一的漏斗过滤流水线"""
        base_threshold = ctx.get('semantic_threshold', 0.85)
        entity_type = ctx.get('entity_type', 'VARIABLE')
        target_name = ctx['target_name']

        stats = {
            "initial": len(candidate_list), "keyword_or_self": 0, "style": 0,
            "heuristic": 0, "analyzer": 0, "semantic": 0, "ast_verify": 0
        }

        base_cands = []
        for cand in candidate_list:
            if cand in ctx['keywords'] or cand == target_name:
                stats["keyword_or_self"] += 1
                continue
            if ctx['preserve_style'] and not self._matches_style(ctx['original_style'], cand):
                stats["style"] += 1
                continue
            base_cands.append(cand)

        if not base_cands:
            self._print_filter_report(target_name, stats, len(final_candidates), log_prefix)
            return 0

        orig_emb = None
        if base_threshold > 0:
            orig_emb = self._get_variable_token_embeddings([ctx['local_prefix']], [target_name],
                                                           [ctx['local_suffix']]).to(self.embedder.device)

        added = 0
        CHUNK_SIZE = max(50, quota * 2)
        target_parts, _ = self._split_identifier(target_name)
        return_type = ctx.get('return_type', None)

        for i in range(0, len(base_cands), CHUNK_SIZE):
            if added >= quota: break

            chunk = base_cands[i: i + CHUNK_SIZE]
            filtered_chunk, heuristic_bonuses = [], []

            for cand in chunk:
                bonus = 0.0
                if hasattr(self, 'scorer'):
                    cand_parts, _ = self._split_identifier(cand)
                    bonus = self.scorer.calculate_heuristic_score(cand_parts, entity_type, target_parts, return_type)
                if bonus <= -100:
                    stats["heuristic"] += 1
                    continue
                if not self.analyzer.can_rename_to(ctx['code_bytes'], target_name, cand):
                    stats["analyzer"] += 1
                    continue
                filtered_chunk.append(cand)
                heuristic_bonuses.append(bonus)

            if not filtered_chunk: continue

            semantically_valid = []
            if base_threshold > 0:
                prefixes = [ctx['local_prefix']] * len(filtered_chunk)
                suffixes = [ctx['local_suffix']] * len(filtered_chunk)
                cand_embs = self._get_variable_token_embeddings(prefixes, filtered_chunk, suffixes).to(
                    self.embedder.device)
                sims = F.cosine_similarity(orig_emb, cand_embs)

                for cand, sim, bonus in zip(filtered_chunk, sims, heuristic_bonuses):
                    final_score = sim.item() + bonus
                    threshold = self._get_dynamic_threshold(target_name, cand,
                                                            base_threshold) if use_dynamic_threshold else base_threshold
                    if final_score >= threshold:
                        semantically_valid.append((cand, final_score))
                    else:
                        stats["semantic"] += 1
            else:
                semantically_valid = [(cand, 1.0) for cand in filtered_chunk]

            semantically_valid.sort(key=lambda x: x[1], reverse=True)

            for cand, final_score in semantically_valid:
                if added >= quota: break
                valid_cand = self._verify_ast_single(cand, ctx)
                if valid_cand:
                    if valid_cand not in final_candidates:
                        final_candidates.append(valid_cand)
                        added += 1
                else:
                    stats["ast_verify"] += 1
        if log_prefix == "HeavyWeight":
            self._print_filter_report(target_name, stats, added, log_prefix)
        return added

    def _print_filter_report(self, target_name: str, stats: dict, final_count: int, log_prefix: str):
        total_filtered = sum(v for k, v in stats.items() if k != 'initial')
        skipped = max(0, stats['initial'] - total_filtered - final_count)
        prefix_str = f"[{log_prefix}] " if log_prefix else ""

        print(f"\n[*] {prefix_str}Filtering Report for `{target_name}`:")
        print(f"    📥 Initial Candidates Given  : {stats['initial']:>3}")
        if stats['keyword_or_self'] > 0: print(f"    🚫 Filtered (Keyword/Self)   : {stats['keyword_or_self']:>3}")
        if stats['style'] > 0:           print(f"    🚫 Filtered (Style Match)    : {stats['style']:>3}")
        if stats['heuristic'] > 0:       print(f"    🚫 Filtered (Heuristic Rules): {stats['heuristic']:>3}")
        if stats['analyzer'] > 0:        print(f"    🚫 Filtered (Static AST)     : {stats['analyzer']:>3}")
        if stats['semantic'] > 0:        print(f"    🚫 Filtered (Semantic Sim)   : {stats['semantic']:>3}")
        if stats['ast_verify'] > 0:      print(f"    🚫 Filtered (AST Rewrite)    : {stats['ast_verify']:>3}")
        if skipped > 0:                  print(f"    ⏭️  Skipped (Quota Reached)  : {skipped:>3}")
        print(f"    ✅ Final Valid Retained      : {final_count:>3}")