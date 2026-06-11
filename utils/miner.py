import os
import re
import json
from collections import defaultdict
import pandas as pd
# 假设 IdentifierAnalyzer 可以在这里被导入
from utils.ast_tools import IdentifierAnalyzer


class NamingDataMiner:
    def __init__(self, analyzer: IdentifierAnalyzer):
        """复用主程序的 AST Analyzer，确保判定标准绝对一致"""
        self.analyzer = analyzer
        self.stats = {
            'FUNCTION': {'prefixes': defaultdict(int), 'suffixes': defaultdict(int)},
            'VARIABLE': {'prefixes': defaultdict(int), 'suffixes': defaultdict(int)},
            'BOOLEAN_VAR': {'prefixes': defaultdict(int), 'suffixes': defaultdict(int)}
        }

    def _split_identifier(self, name: str):
        if '_' in name:
            return name.split('_')
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
        return parts if parts else [name]

    def mine_code(self, code_bytes: bytes):
        try:
            identifiers = self.analyzer.extract_identifiers(code_bytes)
        except Exception:
            return

        for name, occurrences in identifiers.items():
            if not name or not occurrences: continue

            parts = [p.lower() for p in self._split_identifier(name) if p]
            if not parts: continue

            first_word = parts[0]
            last_word = parts[-1]  # 获取后缀

            ent_type = occurrences[0].get("entity_type", "variable")

            # 启发式布尔判定
            is_bool = False
            if first_word in ['is', 'has', 'can', 'should', 'will', 'was', 'did'] or \
                    last_word in ['flag', 'ok', 'status', 'success', 'enable', 'disable']:
                is_bool = True

            if ent_type == "function":
                self.stats['FUNCTION']['prefixes'][first_word] += 1
                self.stats['FUNCTION']['suffixes'][last_word] += 1
            elif is_bool:
                self.stats['BOOLEAN_VAR']['prefixes'][first_word] += 1
                self.stats['BOOLEAN_VAR']['suffixes'][last_word] += 1
            else:
                self.stats['VARIABLE']['prefixes'][first_word] += 1
                self.stats['VARIABLE']['suffixes'][last_word] += 1

    def mine_parquet(self, filepath: str):
        """直接从 Parquet 数据集中挖掘"""
        print(f"[*] Mining dataset '{filepath}' for naming statistics... (This might take a moment)")
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            print(f"[!] Failed to read parquet file for mining: {e}")
            return

        if 'func' not in df.columns:
            print("[!] 'func' column missing, cannot mine.")
            return

        # 为了速度，随机抽取最多 20000 条代码进行词频统计即可，足以获得稳定的分布
        sample_df = df if len(df) <= 100000 else df.sample(n=100000, random_state=42)

        total = len(sample_df)
        for idx, code in enumerate(sample_df['func'].dropna()):
            if idx % 5000 == 0 and idx > 0:
                print(f"    -> Mined {idx}/{total} snippets...")
            self.mine_code(code.encode('utf-8', errors='ignore'))

    def export_json(self, output_path: str, min_count: int = 5, min_prob: float = 0.0015, top_k: int = 150):
        """
        归一化并导出统计数据，加入多重阈值过滤以消除长尾噪音。

        :param output_path: JSON 导出路径
        :param min_count: 绝对频次阈值（出现次数少于此值的直接丢弃，防笔误/孤立词）
        :param min_prob: 相对概率阈值（占比低于 0.1% 的词汇丢弃，防长尾噪音）
        :param top_k: 兜底限制，最多只保留前 N 个最高频的词（防文件体积爆炸）
        """
        normalized_stats = {}
        for entity_type, categories in self.stats.items():
            normalized_stats[entity_type] = {}

            for pos_type, counts in categories.items():  # pos_type 是 prefixes 或 suffixes
                total = sum(counts.values())
                if total == 0:
                    continue

                # ==========================================
                # 1. 绝对频次过滤 (Min Count Threshold)
                # ==========================================
                filtered_counts = {w: c for w, c in counts.items() if c >= min_count}

                # ==========================================
                # 2. 排序并执行兜底 Top-K 截断
                # ==========================================
                top_items = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

                # ==========================================
                # 3. 计算相对概率，并执行概率过滤 (Min Prob Threshold)
                # ==========================================
                final_items = {}
                for word, count in top_items:
                    prob = count / total
                    if prob >= min_prob:
                        final_items[word] = prob

                normalized_stats[entity_type][pos_type] = final_items

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_stats, f, indent=2)

        print(f"[+] 高质量统计数据已导出至 {output_path}")
        print(f"    - 过滤条件: 出现次数 >= {min_count}, 且占比 >= {min_prob * 100}%")