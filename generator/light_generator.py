import itertools
import math
import re
from typing import Dict, Any, List

import torch
import torch.nn.functional as F

from generator.base_generator import BaseCandidateGenerator


class LightweightCandidateGenerator(BaseCandidateGenerator):
    def __init__(self, mlm_engine, embedder, analyzer, config):
        super().__init__(embedder, analyzer, config)
        self.mlm_engine = mlm_engine

    def _get_mutation_pattern(self, target_name: str, cand_name: str) -> str:
        # Extracts the specific mutation pattern of target and candidate identifiers.
        t_parts, t_sep = self._split_identifier(target_name)
        c_parts, c_sep = self._split_identifier(cand_name)

        if len(t_parts) == len(c_parts) and t_sep == c_sep and len(t_parts) > 1:
            pattern = []
            identical_count = 0
            for t, c in zip(t_parts, c_parts):
                if t.lower() == c.lower():
                    pattern.append(t.lower())
                    identical_count += 1
                else:
                    pattern.append('*')

            if identical_count > 0 and identical_count < len(t_parts):
                sep_display = '_' if t_sep == '_' else ''
                return sep_display.join(pattern)

        return '*'

    def _build_masked_string(self, parts: List[str], start: int, end: int, num_masks: int, style: str, mask_token: str,
                             target_name: str) -> str:
        # Synthesizes a code string with mask tokens inserted for MLM prediction.
        mask_list = [mask_token] * num_masks
        new_parts = parts[:start] + mask_list + parts[end:]

        if style == '_':
            return "_".join(new_parts)
        elif style == 'camel':
            res = []
            for j, p in enumerate(new_parts):
                if p == mask_token:
                    res.append(p)
                else:
                    res.append(p.lower() if j == 0 and target_name[0].islower() else p.capitalize())
            return "".join(res).replace(mask_token.capitalize(), mask_token)
        else:
            return mask_token

    def _get_model_logits_batched(self, cropped_codes: List[str]):
        # Runs MLM inference to predict masked token logits for a batch of code inputs.
        if not cropped_codes: return None, []
        inputs = self.mlm_engine.tokenizer(
            cropped_codes, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.mlm_engine.device)
        mask_token_id = self.mlm_engine.tokenizer.mask_token_id

        with torch.no_grad():
            batch_logits = self.mlm_engine.model(**inputs).logits

        batch_mask_indices = [(inputs.input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0] for i in
                              range(batch_logits.size(0))]
        return batch_logits, batch_mask_indices

    def _decode_words(self, mask_logits, top_k, allow_underscore=False, required_length=None):
        # Decodes token predictions from model logits into standard text representations.
        _, top_indices = torch.topk(mask_logits, top_k, dim=-1)
        words = []
        for idx in top_indices:
            w = self.mlm_engine.tokenizer.decode([idx]).strip().replace('Ġ', '').replace('##', '')
            if allow_underscore:
                w = re.sub(r'[^a-zA-Z0-9_]', '', w)
                if not w or (not w[0].isalpha() and w[0] != '_'): continue
            else:
                w = re.sub(r'[^a-zA-Z0-9]', '', w)
                if not w: continue
            if required_length is not None and len(w) != required_length: continue
            words.append(w)
        return words

    def generate_candidates(self, batch_tasks: List[Dict[str, Any]], top_k_mlm: int = 40, top_n_keep: int = 20,
                            ) -> Dict[str, List[str]]:
        # Generates semantic replacement candidates using contextual masked language modeling.
        results = {task["target_name"]: [] for task in batch_tasks}
        mlm_tracking = []
        task_metadata = {}
        mask_token = self.mlm_engine.tokenizer.mask_token
        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.analyzer.language
        for task_idx, task in enumerate(batch_tasks):
            target_name = task["target_name"]

            slice_code_str = task["code_str"]
            slice_code_bytes = slice_code_str.encode("utf-8")
            tree = parser.parse(slice_code_bytes)

            slice_identifiers = self.analyzer.extract_identifiers(slice_code_bytes)

            if target_name not in slice_identifiers:
                continue

            best_occ_idx = self._find_best_context_occurrence(slice_code_bytes, slice_identifiers[target_name], tree)
            target_info = slice_identifiers[target_name][best_occ_idx]
            leading_m = re.match(r'^_+', target_name)
            leading_us = leading_m.group(0) if leading_m else ""
            core_name = target_name[len(leading_us):] if leading_us else target_name

            entity_type = 'BOOLEAN_VAR' if core_name.startswith(('is_', 'has_', 'can_', 'should_')) else (
                'FUNCTION' if target_info.get('entity_type') == 'function' else 'VARIABLE')

            original_style = self._detect_naming_style(target_name)
            parts, style = self._split_identifier(core_name)

            prefix_bytes = slice_code_bytes[:target_info['start']]
            suffix_bytes = slice_code_bytes[target_info['end']:]

            local_prefix = prefix_bytes.decode("utf-8", errors="replace")
            local_suffix = suffix_bytes.decode("utf-8", errors="replace")

            task_metadata[task_idx] = {
                "target_name": target_name, "core_name": core_name, "leading_us": leading_us,
                "parts": parts, "style": style, "n_parts": len(parts),
                "entity_type": entity_type, "original_style": original_style,

                "full_code_str": task["full_code_str"],
                "full_code_bytes": task["full_code_str"].encode("utf-8"),
                "full_identifiers": task.get("full_identifiers", slice_identifiers),

                "local_prefix": local_prefix, "local_suffix": local_suffix,
                "raw_mlm_cands": []
            }

            MAX_CHAR_LIMIT = 2500

            prefix_str = local_prefix[-MAX_CHAR_LIMIT:] if len(local_prefix) > MAX_CHAR_LIMIT else local_prefix
            suffix_str = local_suffix[:MAX_CHAR_LIMIT] if len(local_suffix) > MAX_CHAR_LIMIT else local_suffix

            variations = []
            if len(parts) == 1:
                variations.extend([
                    {'expand_mode': 'none', 'num_masks': 1, 'masked_str': leading_us + mask_token},
                    {'expand_mode': 'prefix', 'num_masks': 1, 'masked_str': leading_us + f"{mask_token}_{core_name}"},
                    {'expand_mode': 'suffix', 'num_masks': 1, 'masked_str': leading_us + f"{core_name}_{mask_token}"}
                ])
            else:
                for i in range(len(parts)):
                    m_str = self._build_masked_string(parts, i, i + 1, 1, style, mask_token, core_name)
                    variations.append({'expand_mode': 'sub', 'start': i, 'end': i + 1, 'num_masks': 1,
                                       'masked_str': leading_us + m_str})
                if len(parts) >= 2:
                    for i in range(len(parts) - 1):
                        m_str = self._build_masked_string(parts, i, i + 2, 2, style, mask_token, core_name)
                        variations.append({'expand_mode': 'sub', 'start': i, 'end': i + 2, 'num_masks': 2,
                                           'masked_str': leading_us + m_str})

            for var in variations:
                mlm_tracking.append({"task_idx": task_idx, "cropped_code": prefix_str + var['masked_str'] + suffix_str,
                                     "variation_info": var})

        if not task_metadata: return results

        all_cropped_codes = [item["cropped_code"] for item in mlm_tracking]
        batch_logits, batch_mask_indices = self._get_model_logits_batched(all_cropped_codes)

        def _join_parts(new_parts, orig_name, st):
            if st == '_':
                return "_".join(new_parts)
            elif st == 'camel':
                return "".join(
                    p.lower() if j == 0 and orig_name[0].islower() else p.capitalize() for j, p in enumerate(new_parts))
            return "".join(new_parts)

        if batch_logits is not None:
            for i, track_info in enumerate(mlm_tracking):
                meta = task_metadata[track_info["task_idx"]]
                var_info = track_info["variation_info"]
                logits = batch_logits[i:i + 1]
                mask_indices = batch_mask_indices[i]
                num_masks = var_info.get('num_masks', 1)

                core_name = meta["core_name"]
                leading_us = meta["leading_us"]

                if len(mask_indices) < num_masks: continue

                if num_masks == 1:
                    words = self._decode_words(logits[0, mask_indices[0], :], top_k_mlm)
                    for w in words:
                        if var_info.get('expand_mode') == 'prefix':
                            meta["raw_mlm_cands"].append(f"{leading_us}{w}_{core_name}")
                        elif var_info.get('expand_mode') == 'suffix':
                            meta["raw_mlm_cands"].append(f"{leading_us}{core_name}_{w}")
                        else:
                            if meta["n_parts"] == 1:
                                meta["raw_mlm_cands"].append(f"{leading_us}{w}")
                            else:
                                joined = _join_parts(
                                    meta["parts"][:var_info['start']] + [w] + meta["parts"][var_info['end']:],
                                    core_name, meta["style"])
                                meta["raw_mlm_cands"].append(f"{leading_us}{joined}")
                elif num_masks == 2:
                    top_k_2holes = min(4, max(2, top_k_mlm // 4))
                    words1 = self._decode_words(logits[0, mask_indices[0], :], top_k_2holes)
                    words2 = self._decode_words(logits[0, mask_indices[1], :], top_k_2holes)
                    for w1, w2 in itertools.product(words1, words2):
                        joined = _join_parts(
                            meta["parts"][:var_info['start']] + [w1, w2] + meta["parts"][var_info['end']:],
                            core_name, meta["style"])
                        meta["raw_mlm_cands"].append(f"{leading_us}{joined}")

        for t_idx, meta in task_metadata.items():
            unique_mlm_cands = list(dict.fromkeys(meta["raw_mlm_cands"]))

            cg_cfg = self.config.get('candidate_generation', {})
            lw_cfg = cg_cfg.get('lightweight', {})


            ctx = {
                'code_bytes': meta["full_code_bytes"],
                'full_code_str': meta["full_code_str"],
                'target_name': meta["target_name"],
                'identifiers': meta["full_identifiers"],
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"],

                'semantic_threshold': lw_cfg.get('semantic_threshold', 0.85),
                'preserve_style': cg_cfg.get('preserve_style', True),
                'ppl_max_ratio': cg_cfg.get('ppl_max_ratio', 1.2),
                'ppl_max_abs': cg_cfg.get('ppl_max_abs', 50.0),

                'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["full_identifiers"].get(meta["target_name"], []) if
                     u.get('return_type')), None),
            }

            final_candidates = []
            self._verify_and_filter(
                candidate_list=unique_mlm_cands,
                quota=top_n_keep,
                final_candidates=final_candidates,
                ctx=ctx,
                use_dynamic_threshold=True,
                log_prefix="LightWeight"
            )
            results[meta["target_name"]] = final_candidates

        return results