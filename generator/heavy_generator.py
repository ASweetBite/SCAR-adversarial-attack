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

        # 如果完全解析失败，暴露现场日志
        print(f"\n[!] JSON Parse Warning (Multi Task): Could not extract Dict. Raw output:\n{response[:250]}\n")
        return {}

    def _build_multi_llm_prompt(self, meta_group: List[Dict], top_n: int) -> str:
        """为同时预测多个变量构建高级 Prompt（共享完整上下文，平衡缩写与语义，大模型专属）"""
        prompt = (
            "You are an Expert C/C++ Code Refactoring Specialist and Static Analysis Tool.\n"
            f"Your task is to generate EXACTLY {top_n} highly contextual, semantically equivalent alternative names for MULTIPLE identifiers.\n\n"
        )

        shared_full_code = meta_group[0]['full_code_str']
        is_shared_full = all(m['full_code_str'] == shared_full_code for m in meta_group)

        if is_shared_full:
            prompt += f"[Shared Context Code]\n```c\n{shared_full_code}\n```\n\n"
        else:
            # 兜底逻辑：如果全量代码不同（通常批处理时极少发生），回退到判断 AST 折叠片段是否相同
            shared_slice = meta_group[0]['slice_code_str']
            is_shared_slice = all(m['slice_code_str'] == shared_slice for m in meta_group)
            if is_shared_slice:
                prompt += f"[Shared Context Code]\n```c\n{shared_slice}\n```\n\n"

        prompt += "[Tasks & Specific Constraints]\n"
        example_dict = {}

        for idx, meta in enumerate(meta_group, 1):
            target_name = meta['target_name']
            target_parts = meta['parts']
            style = meta['original_style']
            entity_type = meta['entity_type']
            n_parts = meta['n_parts']

            target_first = target_parts[0].lower() if target_parts else ""

            # 探测打分器关心的动词类型
            is_getter = target_first in {'get', 'fetch', 'read', 'query', 'retrieve', 'calc', 'compute', 'find', 'search'}
            is_setter = target_first in {'set', 'write', 'update', 'assign', 'put', 'init', 'clear', 'reset'}
            is_bool = target_first in {'is', 'has', 'can', 'should', 'will', 'was', 'did', 'check', 'allow'}

            # 1. 实体类型与打分器硬约束对齐
            if entity_type == 'VARIABLE':
                entity_rule = "Entity: VARIABLE. MUST start with a NOUN. Strictly NO action verbs at the start (except safe noun-verbs: 'request', 'reply', 'result', 'record', 'state', 'cache', 'count')."
            elif entity_type == 'BOOLEAN_VAR':
                entity_rule = "Entity: BOOLEAN. MUST start with prefix (is, has, can, should) OR end with suffix (flag, ok, status, success, enable). NO getter/setter verbs."
            else:
                func_rules = ["Entity: FUNCTION/METHOD."]
                if is_getter: func_rules.append("MUST start with a GETTER verb (e.g., get, fetch, query, calc). NO setter verbs.")
                elif is_setter: func_rules.append("MUST start with a SETTER verb (e.g., set, write, update, init). NO getter verbs.")
                elif is_bool: func_rules.append("MUST start with a BOOLEAN prefix (e.g., is, has, check).")
                else: func_rules.append("MUST contain at least one valid ACTION VERB.")
                entity_rule = " ".join(func_rules)

            # 2. 前缀约束
            leading_us_rule = ""
            if target_name.startswith('_'):
                us_prefix = re.match(r'^_+', target_name).group(0)
                leading_us_rule = f" MUST strictly preserve leading '{us_prefix}'."

            # 3. 长度与缩写/泛用词策略
            if n_parts <= 2:
                max_allowed_parts = n_parts + 2
                strategy = f"Strategy: Keep under {max_allowed_parts} words. Idiomatic C/C++ generic names (e.g., data, val, ctx, res, tmp, buf) are great. Standard abbreviations or expansions are highly encouraged."
            else:
                strategy = "Strategy: Semantic Synonyms. Keep length similar. Standard abbreviations of parts are encouraged."

            prompt += f"--- Task {idx}: `{target_name}` ---\n"

            # 兜底的兜底：如果连 AST 折叠片段都不共享，只能为每个变量单独附上它自己的 AST片段
            if not is_shared_full and not all(m['slice_code_str'] == meta_group[0]['slice_code_str'] for m in meta_group):
                prompt += f"Context Code:\n```c\n{meta['slice_code_str']}\n```\n"

            prompt += f"- FORMAT: Strictly `{style}`.{leading_us_rule}\n"
            prompt += f"- {entity_rule}\n"
            prompt += f"- {strategy}\n\n"

            example_dict[target_name] = [f"cand{i}" for i in range(1, top_n + 1)]

        example_json = json.dumps(example_dict, indent=2)

        prompt += f"""[Global Anti-Patterns & Refactoring Rules (CRITICAL)]
1. SEMANTIC QUALITY FIRST: The ultimate criterion is how well the generated name preserves the exact business logic and context semantics.
2. ABBREVIATIONS: You may use STANDARD abbreviations (e.g., message -> msg) OR expand existing abbreviations (e.g., idx -> index). These receive heuristic bonuses.
3. NO GARBAGE SPELLING: DO NOT spam weird or unnatural spelling variations (e.g., do NOT generate "indx", "idex", "ndx" to fake an abbreviation). Focus on highly readable C/C++ idioms.
4. NO DUPLICATE WORDS: Never repeat words within the same name (e.g., "data_data" is strictly rejected).
5. NO LAZY SUFFIXES: Do not append arbitrary numbers (e.g., "count1", "index2").

[Output Task]
Output ONLY a valid JSON Object. Keys are original target variable names, values are arrays of EXACTLY {top_n} strings. Do not explain, do not use markdown format (```json).
Example format:
{example_json}

JSON
{{"""
        return prompt


    def _build_llm_prompt(self, context_code: str, target_name: str, target_parts: list, style: str, top_n: int, entity_type: str, n_parts: int) -> str:
        """为单变量预测构建高度迎合启发式打分器的高级 Prompt（平衡缩写加分与语义质量）"""

        target_first = target_parts[0].lower() if target_parts else ""

        # 探测启发式分类器关心的动词类型
        is_getter = target_first in {'get', 'fetch', 'read', 'query', 'retrieve', 'calc', 'compute', 'find', 'search'}
        is_setter = target_first in {'set', 'write', 'update', 'assign', 'put', 'init', 'clear', 'reset'}
        is_bool = target_first in {'is', 'has', 'can', 'should', 'will', 'was', 'did', 'check', 'allow'}

        # 1. 实体类型与词性约束
        if entity_type == 'VARIABLE':
            entity_rule = (
                "Entity: VARIABLE (Data/State).\n"
                "- MUST START WITH A NOUN. It is STRICTLY FORBIDDEN to start a variable name with an action verb (e.g., 'run_data', 'read_buf' are invalid).\n"
                "- Allowed safe verb-nouns at the start: 'request', 'reply', 'result', 'record', 'state', 'cache', 'count'."
            )
        elif entity_type == 'BOOLEAN_VAR':
            entity_rule = (
                "Entity: BOOLEAN VARIABLE.\n"
                "- MUST start with a boolean prefix (e.g., is, has, can, should) OR end with a boolean suffix (e.g., flag, ok, status, success, enable).\n"
                "- DO NOT start with setter/getter verbs."
            )
        else:
            func_rules = ["Entity: FUNCTION/METHOD."]
            if is_getter:
                func_rules.append("- The original function is a GETTER. ALL generated names MUST start with a getter verb (e.g., get, fetch, read, query, retrieve, calc, compute, find, search). DO NOT use setter verbs.")
            elif is_setter:
                func_rules.append("- The original function is a SETTER. ALL generated names MUST start with a setter verb (e.g., set, write, update, assign, put, init, clear, reset). DO NOT use getter verbs.")
            elif is_bool:
                func_rules.append("- The original function checks a boolean state. ALL generated names MUST start with a boolean prefix (e.g., is, has, can, should, check).")
            else:
                func_rules.append("- MUST contain at least one valid ACTION VERB. Do not use pure nouns for functions.")
            entity_rule = "\n".join(func_rules)

        # 2. 格式与前缀约束
        leading_us_rule = ""
        if target_name.startswith('_'):
            us_prefix = re.match(r'^_+', target_name).group(0)
            leading_us_rule = f"\n- PREFIX REQUIREMENT: The original name starts with '{us_prefix}'. ALL your suggestions MUST strictly start with '{us_prefix}'."

        # 3. 核心重构策略 (强调语义质量、双向缩写/展开，并警告垃圾变体)
        if n_parts <= 2:
            max_allowed_parts = n_parts + 2
            strategy_instruction = f"""[Refactoring Strategy: Idiomatic & High Semantic Quality]
- Keep it under {max_allowed_parts} words. Match the naming habits of the original codebase.
- The ULTIMATE criterion for accepting a candidate is its Semantic Quality in the context. 
- Idiomatic C/C++ generic names (e.g., data, val, ctx, res, tmp) are great if they match the semantics perfectly.
- You may use STANDARD abbreviations (e.g., message -> msg) OR expand existing abbreviations (e.g., idx -> index). These receive a slight heuristic bonus.
- WARNING: DO NOT spam weird or unnatural spelling variations (e.g., do NOT generate "indx", "idex", "ndx" just to force an abbreviation for "index"). Focus on highly readable, professional C/C++ code."""
        else:
            strategy_instruction = f"""[Refactoring Strategy: Deep Semantic Synonyms]
- Deeply analyze the exact business logic from the context.
- The ULTIMATE criterion is how well the generated name preserves the original semantic space.
- Provide highly professional, domain-specific synonyms. Generic base nouns ('data', 'buffer', 'value', 'info') combined with context words are fine.
- You may abbreviate or expand parts of the name using STANDARD C/C++ idioms, but do not invent unnatural shortcut names."""

        return f"""You are an Expert C/C++ Code Refactoring Specialist and Static Analysis Tool.
Your task is to generate EXACTLY {top_n} highly contextual, semantically equivalent alternative names for the identifier `{target_name}`.

[Context Code]
{context_code}

{strategy_instruction}

[Strict Naming Constraints]
- FORMAT: MUST strictly adhere to the `{style}` naming convention.{leading_us_rule}
{entity_rule}

[Anti-Patterns (Violating these will crash the system)]
1. NO DUPLICATE WORDS: Never repeat words within the same name (e.g., "data_data", "buffer_buffer" are strictly rejected).
2. NO LAZY SUFFIXES: Do not just lazily append "1", "2" or "_new" to the original name (unless mimicking an existing array/loop index pattern).
3. NO GARBAGE SPELLING: Do not invent meaningless letter combinations just to be different.

[Output Task]
Output ONLY a valid JSON array containing EXACTLY {top_n} strings. Do not explain, do not add markdown format (```json).
Example format for {top_n} items: ["candidate1", "candidate2", "candidate3", ...]

JSON
["""

    def generate_candidates(self, vulnerable_tasks: List[Dict[str, Any]], target_quota: int = 20) -> Dict[
        str, List[str]]:
        results = {task["target_name"]: [] for task in vulnerable_tasks}
        task_metadata = []

        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.analyzer.language

        # 1. 解析任务和提取上下文
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

            task_metadata.append({
                "target_name": target_name, "parts": parts, "style": style, "n_parts": len(parts),
                "entity_type": entity_type, "original_style": original_style,
                "slice_code_str": slice_code_str,  # 保留字符串用于 Prompt 构建

                "full_code_str": task["full_code_str"],
                "full_code_bytes": task["full_code_str"].encode("utf-8"),
                "full_identifiers": task.get("full_identifiers", slice_identifiers),

                "local_prefix": prefix_str, "local_suffix": suffix_str
            })

        if not task_metadata: return results

        # 2. 构建 Prompts (分流逻辑: 多任务合并 vs 单任务)
        raw_candidates_dict = {meta["target_name"]: [] for meta in task_metadata}
        llm_prompts = []

        GROUPING_THRESHOLD = 10
        GROUP_SIZE = 5  # 每 5 个变量打包成 1 次 API 调用，兼顾上下文窗口长度和效率

        if target_quota <= GROUPING_THRESHOLD:
            # === 打包请求模式 (降低 API 调用频率) ===
            grouped_metas = [task_metadata[i:i + GROUP_SIZE] for i in range(0, len(task_metadata), GROUP_SIZE)]
            for group in grouped_metas:
                prompt = self._build_multi_llm_prompt(group, int(target_quota * 1.5))
                llm_prompts.append(prompt)

            if llm_prompts:
                try:
                    llm_responses = self.llm_client.batch_chat(llm_prompts)
                except Exception as e:
                    print(f"[!] LLM Batch Chat Failed: {e}")
                    llm_responses = [""] * len(llm_prompts)

                for resp, group in zip(llm_responses, grouped_metas):
                    # print("\n" + "=" * 60)
                    # print(f"[DEBUG] LLM Raw Output for variables: {[m['target_name'] for m in group]}")
                    # print(f"[{resp}]")
                    # print("=" * 60 + "\n")
                    parsed_dict = self._parse_multi_json_response(resp)
                    for meta in group:
                        t_name = meta["target_name"]
                        # 处理可能的键值大小写偏差
                        raw_cands = parsed_dict.get(t_name)
                        if raw_cands is None:
                            for k, v in parsed_dict.items():
                                if k.lower() == t_name.lower():
                                    raw_cands = v
                                    break
                        raw_candidates_dict[t_name] = raw_cands or []

        else:
            # === 原有高精度单任务模式 ===
            for meta in task_metadata:
                prompt = self._build_llm_prompt(
                    meta["slice_code_str"], meta["target_name"], meta["parts"], meta["original_style"],
                    int(target_quota * 1.5), meta["entity_type"], meta["n_parts"]
                )
                llm_prompts.append(prompt)

            if llm_prompts:
                try:
                    llm_responses = self.llm_client.batch_chat(llm_prompts)
                except Exception as e:
                    print(f"[!] LLM Batch Chat Failed: {e}")
                    llm_responses = [""] * len(llm_prompts)

                for resp, meta in zip(llm_responses, task_metadata):
                    raw_candidates_dict[meta["target_name"]] = self._parse_single_json_response(resp)

        # 3. 后处理与 AST/语义校验 (对这两种模式获取的结果进行统一清洗)
        cg_cfg = self.config.get('candidate_generation', {})
        hw_cfg = cg_cfg.get('heavyweight', {})

        for meta in task_metadata:
            t_name = meta["target_name"]
            parsed_cands = raw_candidates_dict[t_name]

            leading_m = re.match(r'^_+', t_name)
            leading_us = leading_m.group(0) if leading_m else ""

            valid_cands, oversized_cands = [], []
            for c in parsed_cands:
                if isinstance(c, str) and c.strip():
                    clean_cand = c.strip()

                    # 规范化前导下划线
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

            ctx = {
                'code_bytes': meta["full_code_bytes"],
                'full_code_str': meta["full_code_str"],
                'target_name': t_name,
                'identifiers': meta["full_identifiers"],
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"],

                'semantic_threshold': hw_cfg.get('semantic_threshold', 0.85),
                'preserve_style': cg_cfg.get('preserve_style', True),

                'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["full_identifiers"].get(t_name, []) if
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
            results[t_name] = final_candidates

        return results