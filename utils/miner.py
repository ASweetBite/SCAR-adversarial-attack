import re
import json
from collections import defaultdict
import pandas as pd
from utils.ast_tools import IdentifierAnalyzer


class NamingDataMiner:
    def __init__(self, analyzer: IdentifierAnalyzer):
        # Initializes the naming data miner with an AST analyzer.
        self.analyzer = analyzer
        self.stats = {
            'FUNCTION': {'prefixes': defaultdict(int), 'suffixes': defaultdict(int)},
            'VARIABLE': {'prefixes': defaultdict(int), 'suffixes': defaultdict(int)},
            'BOOLEAN_VAR': {'prefixes': defaultdict(int), 'suffixes': defaultdict(int)}
        }

    def _split_identifier(self, name: str):
        # Splits camelCase or snake_case identifiers into individual parts.
        if '_' in name:
            return name.split('_')
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
        return parts if parts else [name]

    def mine_code(self, code_bytes: bytes):
        # Extracts and classifies prefix/suffix statistics from source code bytes.
        try:
            identifiers = self.analyzer.extract_identifiers(code_bytes)
        except Exception:
            return

        for name, occurrences in identifiers.items():
            if not name or not occurrences: continue

            parts = [p.lower() for p in self._split_identifier(name) if p]
            if not parts: continue

            first_word = parts[0]
            last_word = parts[-1]

            ent_type = occurrences[0].get("entity_type", "variable")

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
        # Mines naming statistics directly from a Parquet dataset.
        print(f"[*] Mining dataset '{filepath}' for naming statistics... (This might take a moment)")
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            print(f"[!] Failed to read parquet file for mining: {e}")
            return

        if 'func' not in df.columns:
            print("[!] 'func' column missing, cannot mine.")
            return

        sample_df = df if len(df) <= 100000 else df.sample(n=100000, random_state=42)

        total = len(sample_df)
        for idx, code in enumerate(sample_df['func'].dropna()):
            if idx % 5000 == 0 and idx > 0:
                print(f"    -> Mined {idx}/{total} snippets...")
            self.mine_code(code.encode('utf-8', errors='ignore'))

    def export_json(self, output_path: str, min_count: int = 5, min_prob: float = 0.0015, top_k: int = 150):
        # Normalizes and exports the mined statistics to a JSON file.
        normalized_stats = {}
        for entity_type, categories in self.stats.items():
            normalized_stats[entity_type] = {}

            for pos_type, counts in categories.items():
                total = sum(counts.values())
                if total == 0:
                    continue

                filtered_counts = {w: c for w, c in counts.items() if c >= min_count}

                top_items = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

                final_items = {}
                for word, count in top_items:
                    prob = count / total
                    if prob >= min_prob:
                        final_items[word] = prob

                normalized_stats[entity_type][pos_type] = final_items

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_stats, f, indent=2)

        print(f"[+] High-quality statistics successfully exported to {output_path}")
        print(f"    - Filter criteria: appearance count >= {min_count}, and proportion >= {min_prob * 100}%")