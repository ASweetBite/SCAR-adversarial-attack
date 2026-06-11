import json
import os
import re

import nltk
from nltk import pos_tag


class StatisticalNamingScorer:
    """
    结合统计数据与 NLTK 词性标注的启发式打分器。
    引入函数语义护城河：严格把控 Getter, Setter, Bool 的转换。
    """

    def __init__(self, stats_file_path: str = 'naming_stats.json'):
        self.stats_file_path = stats_file_path
        self.stats = {}

        self._load_stats()
        self._warmup_nltk()

        # 动词/名词盲区
        self.VERB_BLIND_SPOTS = {
            'hash', 'run', 'read', 'write', 'load', 'save', 'init',
            'log', 'build', 'parse', 'bind', 'start', 'stop', 'cast',
            'add', 'fix', 'del', 'rm', 'calc', 'cmp', 'update', 'check',
            'alloc', 'free', 'pop', 'push', 'lock', 'unlock', 'clear', 'reset'
        }
        self.NOUN_BLIND_SPOTS = {
            'log', 'hash', 'state', 'cache', 'count', 'size', 'len',"txt","num",
            'ptr', 'idx', 'buf', 'tmp', 'str', 'ret', 'val', 'msg', 'req', 'res'
        }

        # 函数语义动作分类库
        self.GETTER_VERBS = {'get', 'fetch', 'read', 'query', 'retrieve', 'calc', 'compute', 'find', 'search'}
        self.SETTER_VERBS = {'set', 'write', 'update', 'assign', 'put', 'init', 'clear', 'reset'}
        self.BOOL_PREFIXES = {'is', 'has', 'can', 'should', 'will', 'was', 'did', 'check', 'allow'}

        # ==========================================
        # [新增] 极短词白名单 (保护合法编程常识)
        # ==========================================
        self.SHORT_WHITELIST = {
            'id', 'key', 'map', 'set', 'log', 'max', 'min', 'row', 'col',
            'pos', 'end', 'sum', 'num', 'val', 'ret', 'ptr', 'buf', 'tag',
            'msg', 'req', 'res', 'ctx', 'src', 'dst', 'len', 'idx', 'tmp',
            'out', 'in', 'err', 'obj', 'arg', 'env', 'url', 'uri', 'fd', 'ip'
        }

    def _load_stats(self):
        if os.path.exists(self.stats_file_path):
            try:
                with open(self.stats_file_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except Exception:
                self.stats = {}
        else:
            self.stats = {}

    def _warmup_nltk(self):
        try:
            pos_tag(["warmup"])
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            pos_tag(["warmup"])

    def _is_abbreviation(self, w1: str, w2: str, long_w_parts: list = None) -> bool:
        """
        检查 w1 和 w2 是否互为高质量的缩写关系。
        引入【首字母锚定规则】：缩写词的首字母，必须是长词中某个单词的首字母。
        """
        if not w1 or not w2: return False
        w1, w2 = w1.lower(), w2.lower()
        if w1 == w2: return False

        # 确定长短词
        if len(w1) < len(w2):
            short_w, long_w = w1, w2
        else:
            short_w, long_w = w2, w1
            long_w_parts = None

        if long_w_parts:
            valid_start_chars = {p[0].lower() for p in long_w_parts if p}
        else:
            valid_start_chars = {long_w[0]}

        if short_w[0] not in valid_start_chars:
            return False

        if len(long_w) - len(short_w) < 2:
            return False

        if len(short_w) == 1:
            return True

        if long_w.startswith(short_w):
            return True

        it = iter(long_w)
        return all(c in it for c in short_w)

    def calculate_heuristic_score(self, cand_parts: list, entity_type: str, target_parts: list = None,
                                  return_type: str = None) -> float:
        if not cand_parts:
            return 0.0

        score = 0.0
        cand_first = cand_parts[0].lower()
        cand_last = cand_parts[-1].lower()
        target_first = target_parts[0].lower() if target_parts else ""

        if len(set(cand_parts)) != len(cand_parts):
            return -999.0

        # 提前加载统计特征，用于短词豁免和后续校验
        entity_stats = self.stats.get(entity_type, {})
        known_prefixes = entity_stats.get('prefixes', {})
        known_suffixes = entity_stats.get('suffixes', {})

        cand_full = "".join(cand_parts).lower()
        target_full = "".join(target_parts).lower() if target_parts else ""

        type_full = ""
        type_parts = []
        if return_type:
            clean_type = re.sub(r'^(struct|class|enum|union)\s+', '', return_type.strip())
            clean_type_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_type)
            if clean_type_name:
                type_parts, _ = self._split_identifier(clean_type_name)
                type_full = "".join(type_parts).lower()

        is_abbrev = False

        if len(cand_parts) == 1:
            if target_parts:
                is_abbrev = self._is_abbreviation(cand_full, target_full, target_parts)
            if not is_abbrev and type_parts:
                is_abbrev = self._is_abbreviation(cand_full, type_full, type_parts)

        if is_abbrev:
            score += 0.08
        else:
            if len(cand_parts) == 1 and len(cand_full) < 4:
                # 结合真实统计数据：高频短词享受白名单待遇
                is_frequent_in_stats = (known_prefixes.get(cand_full, 0) >= 0.001 or
                                        known_suffixes.get(cand_full, 0) >= 0.001)

                if cand_full not in self.SHORT_WHITELIST and not is_frequent_in_stats:
                    score -= 0.09
                else:
                    score -= 0.02

        if entity_type == 'BOOLEAN_VAR':
            if cand_first in self.BOOL_PREFIXES or cand_last in ['flag', 'ok', 'status', 'success', 'enable',
                                                                 'disable']:
                score += 0.005
            elif len(cand_parts) > 1 and cand_first in self.VERB_BLIND_SPOTS and cand_first not in ['set', 'get']:
                score -= 0.1

        elif entity_type == 'FUNCTION':
            is_target_getter = target_first in self.GETTER_VERBS
            is_target_setter = target_first in self.SETTER_VERBS
            is_target_bool = target_first in self.BOOL_PREFIXES

            is_cand_getter = cand_first in self.GETTER_VERBS
            is_cand_setter = cand_first in self.SETTER_VERBS
            is_cand_bool = cand_first in self.BOOL_PREFIXES

            if (is_target_getter and is_cand_setter) or (is_target_setter and is_cand_getter):
                return -999.0

            if is_target_bool and not (is_cand_bool or is_cand_getter):
                score -= 0.2

            if is_cand_bool and not is_target_bool:
                if return_type and return_type.lower() not in ['bool', 'boolean', 'int', '_bool']:
                    return -999.0

            if return_type:
                return_type_lower = return_type.lower()
                is_void = (return_type_lower == 'void')

                if is_void and is_cand_getter and not is_target_getter:
                    return -999.0
                if not is_void and return_type_lower not in ['bool', 'int'] and is_cand_setter and not is_target_setter:
                    return -999.0

            has_verb = False
            has_unknown_term = False
            tagged = pos_tag(cand_parts)

            for word, tag in tagged:
                w_lower = word.lower()
                if w_lower in self.VERB_BLIND_SPOTS or tag.startswith('VB'):
                    has_verb = True
                    break
                is_rare_in_stats = (known_prefixes.get(w_lower, 0) < 0.001 and known_suffixes.get(w_lower, 0) < 0.001)
                if is_rare_in_stats:
                    has_unknown_term = True

            if not has_verb and not has_unknown_term:
                score -= 0.08

        elif entity_type == 'VARIABLE':
            is_first_word_verb = False
            if cand_first in self.VERB_BLIND_SPOTS:
                is_first_word_verb = True
            elif cand_first not in self.NOUN_BLIND_SPOTS:
                tagged = pos_tag(cand_parts)
                if tagged and tagged[0][1].startswith('VB'):
                    is_first_word_verb = True

            safe_verb_nouns = {'request', 'reply', 'result', 'record', 'return', 'state', 'cache', 'count', 'limit'}
            if cand_first in safe_verb_nouns:
                is_first_word_verb = False

            if is_first_word_verb:
                return -999.0

        return score

    def _split_identifier(self, name: str):
        """
        [新增] 将标识符拆分为构成单词的列表，并返回风格。
        从 Generator 迁移过来的高优切分引擎。
        """
        if not name:
            return [], ''

        if '_' in name:
            # 过滤掉连续下划线导致的空字符串
            return [p for p in name.split('_') if p], '_'
        else:
            # 经典的正则：完美切分驼峰、帕斯卡以及连续大写缩写 (如 HTTPResponse -> HTTP, Response)
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
            if not parts or (len(parts) == 1 and parts[0] == name):
                return [name], ''
            return parts, 'camel'