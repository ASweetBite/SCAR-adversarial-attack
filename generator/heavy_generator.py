import json
import re
from typing import Dict, Any, List

from generator.base_generator import BaseCandidateGenerator


class HeavyWeightCandidateGenerator(BaseCandidateGenerator):
    def __init__(self, embedder, llm_client, analyzer, config):
        super().__init__(embedder, analyzer, config)
        self.llm_client = llm_client

    def _parse_single_json_response(self, response: str) -> List[str]:
        """智能解析单任务数组返回结构"""
        if not response: return []

        # 1. 移除非法 Markdown 代码块标签
        clean_text = re.sub(r'```[a-zA-Z]*', '', response).replace('```', '').strip()

        # 2. 尝试提取完整的 [...] 结构 (Chat 模型的常见输出)
        start = clean_text.find('[')
        end = clean_text.rfind(']')
        if start != -1 and end != -1 and start < end:
            try:
                parsed = json.loads(clean_text[start:end + 1])
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass

        # 3. 尝试处理“续写模式” (Completion 模型从 '[' 之后开始输出的情况)
        patched = clean_text
        if not patched.startswith('['): patched = '[' + patched
        if not patched.endswith(']'): patched = patched + ']'
        try:
            parsed = json.loads(patched)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass

        # 4. 最终正则兜底：不依赖 JSON，直接提取所有引号内的连贯字符
        cands = re.findall(r'["\']([a-zA-Z0-9_]+)["\']', response)
        if cands:
            return cands

        # 如果依然失败，打印模型返回的原始内容，方便你查错
        print(f"\n[!] JSON Parse Warning (Single Task): Could not extract array. Raw output:\n{response[:250]}\n")
        return []

    def _parse_multi_json_response(self, response: str) -> Dict[str, List[str]]:
        """智能解析多任务字典返回结构"""
        if not response: return {}

        clean_text = re.sub(r'```[a-zA-Z]*', '', response).replace('```', '').strip()

        # 1. 尝试提取完整的 {...} 结构 (Chat 模型的常见输出)
        start = clean_text.find('{')
        end = clean_text.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                parsed = json.loads(clean_text[start:end + 1])
                if isinstance(parsed, dict):
                    return {str(k): [str(x) for x in v] if isinstance(v, list) else [] for k, v in parsed.items()}
            except Exception:
                pass

        # 2. 尝试处理“续写模式” (Completion 模型从 '{' 之后开始输出的情况)
        patched = clean_text
        if not patched.startswith('{'): patched = '{' + patched
        if not patched.endswith('}'): patched = patched + '}'
        try:
            parsed = json.loads(patched)
            if isinstance(parsed, dict):
                return {str(k): [str(x) for x in v] if isinstance(v, list) else [] for k, v in parsed.items()}
        except Exception:
            pass

    def _build_llm_prompt(self, context_code: str, target_name: str, style: str, top_n: int, entity_type: str, n_parts: int) -> str:
        # Formulates a strict, JSON-enforced instructions prompt for candidate generation.
        if style == 'camelCase':
            ex_var = "dataBuffer"
            ex_bool = "'isReady', 'hasData'"
            ex_func = "'getData', 'updateState'"
            ex_short = "'shmInfo', 'memData', 'idx'"
        elif style == 'PascalCase':
            ex_var = "DataBuffer"
            ex_bool = "'IsReady', 'HasData'"
            ex_func = "'GetData', 'UpdateState'"
            ex_short = "'ShmInfo', 'MemData', 'Idx'"
        elif style == 'SCREAMING_SNAKE':
            ex_var = "DATA_BUFFER"
            ex_bool = "'IS_READY', 'HAS_DATA'"
            ex_func = "'GET_DATA', 'UPDATE_STATE'"
            ex_short = "'SHM_INFO', 'MEM_DATA', 'IDX'"
        else:
            ex_var = "data_buffer"
            ex_bool = "'is_ready', 'has_data'"
            ex_func = "'get_data', 'update_state'"
            ex_short = "'shm_info', 'mem_data', 'idx'"

        if entity_type == 'VARIABLE':
            entity_rule = f"Use NOUNS only (e.g., '{ex_var}'). NO verbs."
        elif entity_type == 'BOOLEAN_VAR':
            entity_rule = f"Use BOOLEAN prefixes (e.g., {ex_bool})."
        else:
            entity_rule = f"Use ACTION VERBS (e.g., {ex_func})."

        leading_us_rule = ""
        if target_name.startswith('_'):
            leading_us_rule = "\n- PRESERVE PREFIX: The original name starts with '_'. ALL your suggestions MUST start with '_'."

        if n_parts <= 2:
            max_allowed_parts = n_parts + 1
            strategy_instruction = f"""[Strategy: Short & Concise]
- MAX WORDS: {max_allowed_parts} words per name.
- EXAMPLES: {ex_short}
- Use common C/C++ abbreviations (ptr, buf, mem, val)."""
        else:
            strategy_instruction = """[Strategy: Semantic Refactoring]
- Provide professional synonyms matching the exact system logic.
- Keep the length similar to the original name."""

        return f"""You are an expert C/C++ developer. Suggest exactly {top_n} alternative names for `{target_name}`.

[Context Code]
{context_code}
{strategy_instruction}

[Strict Rules]
{entity_rule}
STYLE: Use {style} naming convention.{leading_us_rule}
NO generic names ("new_var", "temp").

[Task]
Output ONLY a JSON array containing EXACTLY {top_n} strings. Do not explain.
Example format for {top_n} items: ["name1", "name2", "name3", ...]

JSON
["""

    def generate_candidates(self, vulnerable_tasks: List[Dict[str, Any]], target_quota: int = 20) -> Dict[str, List[str]]:
        # Generates deep semantic naming candidates for target entities using LLM and vector constraints.
        results = {task["target_name"]: [] for task in vulnerable_tasks}

        llm_prompts = []
        task_metadata = {}
        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.analyzer.language
        for task_idx, task in enumerate(vulnerable_tasks):
            target_name = task["target_name"]

            slice_code_str = task["code_str"]
            slice_code_bytes = slice_code_str.encode("utf-8")
            tree = parser.parse(slice_code_bytes)
            slice_identifiers = self.analyzer.extract_identifiers(slice_code_bytes)
            if target_name not in slice_identifiers:
                continue

            best_occ_idx = self._find_best_context_occurrence(slice_code_bytes, slice_identifiers[target_name], tree)
            target_info = slice_identifiers[target_name][best_occ_idx]

            raw_entity_type = target_info.get('entity_type', 'variable')
            entity_type = 'BOOLEAN_VAR' if target_name.startswith(('is_', 'has_', 'can_', 'should_')) else (
                'FUNCTION' if raw_entity_type == 'function' else 'VARIABLE')

            original_style = self._detect_naming_style(target_name)
            parts, style = self._split_identifier(target_name)

            prefix_bytes = slice_code_bytes[:target_info['start']]
            suffix_bytes = slice_code_bytes[target_info['end']:]
            local_prefix = prefix_bytes.decode("utf-8", errors="replace")
            local_suffix = suffix_bytes.decode("utf-8", errors="replace")

            MAX_CHAR_LIMIT = 2500
            prefix_str = local_prefix[-MAX_CHAR_LIMIT:] if len(local_prefix) > MAX_CHAR_LIMIT else local_prefix
            suffix_str = local_suffix[:MAX_CHAR_LIMIT] if len(local_suffix) > MAX_CHAR_LIMIT else local_suffix

            task_metadata[task_idx] = {
                "target_name": target_name, "parts": parts, "style": style, "n_parts": len(parts),
                "entity_type": entity_type, "original_style": original_style,

                "full_code_str": task["full_code_str"],
                "full_code_bytes": task["full_code_str"].encode("utf-8"),
                "full_identifiers": task.get("full_identifiers", slice_identifiers),

                "local_prefix": prefix_str, "local_suffix": suffix_str
            }

            prompt = self._build_llm_prompt(slice_code_str, target_name, original_style, int(target_quota * 1.5), entity_type,
                                            len(parts))
            llm_prompts.append(prompt)

        if not llm_prompts: return results

        try:
            llm_responses = self.llm_client.batch_chat(llm_prompts)
        except Exception as e:
            print(f"[!] LLM Batch Chat Failed: {e}")
            llm_responses = [""] * len(llm_prompts)

        for resp_idx, response in enumerate(llm_responses):
            meta = task_metadata[resp_idx]
            parsed_cands = []

            leading_m = re.match(r'^_+', meta["target_name"])
            leading_us = leading_m.group(0) if leading_m else ""

            if response and isinstance(response, str):
                clean_text = response.replace("```json", "").replace("```", "").strip()
                first_quote, last_quote = clean_text.find('"'), clean_text.rfind('"')

                if first_quote != -1 and last_quote != -1 and first_quote != last_quote:
                    patched_json = f"[{clean_text[first_quote:last_quote + 1]}]"
                    try:
                        parsed_cands = json.loads(patched_json)
                        if not isinstance(parsed_cands, list): parsed_cands = [str(parsed_cands)]
                    except Exception:
                        pass

                if not parsed_cands:
                    parsed_cands = re.findall(r'["\']([a-zA-Z0-9_]+)["\']', response)

            valid_cands, oversized_cands = [], []
            for c in parsed_cands:
                if isinstance(c, str) and c.strip():
                    clean_cand = c.strip()

                    if leading_us and not clean_cand.startswith(leading_us):
                        clean_cand = leading_us + clean_cand.lstrip('_')
                    elif not leading_us and clean_cand.startswith('_'):
                        clean_cand = clean_cand.lstrip('_')

                    if clean_cand in valid_cands or clean_cand in oversized_cands: continue

                    cand_parts_list, _ = self._split_identifier(clean_cand)
                    limit = meta["n_parts"] + 1 if meta["n_parts"] <= 2 else meta["n_parts"] + 2

                    if len(cand_parts_list) <= limit:
                        valid_cands.append(clean_cand)
                    else:
                        oversized_cands.append(clean_cand)

            min_threshold = int(target_quota * 0.8)
            if len(valid_cands) < min_threshold and oversized_cands:
                valid_cands.extend(oversized_cands[:min_threshold - len(valid_cands)])

            cg_cfg = self.config.get('candidate_generation', {})
            hw_cfg = cg_cfg.get('heavyweight', {})

            ctx = {
                'code_bytes': meta["full_code_bytes"],
                'full_code_str': meta["full_code_str"],
                'target_name': meta["target_name"],
                'identifiers': meta["full_identifiers"],
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"],

                'semantic_threshold': hw_cfg.get('semantic_threshold', 0.85),
                'preserve_style': cg_cfg.get('preserve_style', True),

                'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["full_identifiers"].get(meta["target_name"], []) if
                     u.get('return_type')), None),
            }

            final_candidates = []
            self._verify_and_filter(
                candidate_list=valid_cands,
                quota=target_quota,
                final_candidates=final_candidates,
                ctx=ctx,
                use_dynamic_threshold=False,
                log_prefix="HeavyWeight"
            )
            self._verify_and_filter(valid_cands, target_quota, final_candidates, ctx)
            results[meta["target_name"]] = final_candidates

        return results