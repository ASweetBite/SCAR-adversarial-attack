import argparse
import os
import random
import numpy as np
import torch
import yaml

from generator.heavy_generator import HeavyWeightCandidateGenerator
from attacks.scar_attacker import SCARAttacker
from generator.light_generator import LightweightCandidateGenerator
from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.dataset_loader import DatasetLoader
from utils.llm_loader import LocalLLMClient
from utils.miner import NamingDataMiner
from utils.mlm_engine import MLMEngine
from utils.model_zoo import ModelZoo


def main(args, config):
    # Coordinates model execution, candidate generation, and the attack evaluation process.
    lang = config['global'].get('lang', 'cpp')
    analyzer = IdentifierAnalyzer(lang=lang)

    cg_config = config.get('candidate_generation', {})
    stats_path = cg_config.get('naming_stats_path', 'naming_stats.json')
    dataset_path = config['run_params']['dataset']

    if not os.path.exists(stats_path):
        print(f"\n[!] Heuristic naming statistics dictionary '{stats_path}' not found.")
        print(f"[*] Launching offline data miner (based on dataset: {dataset_path})...")
        miner = NamingDataMiner(analyzer)
        miner.mine_parquet(dataset_path)
        miner.export_json(stats_path)
    else:
        print(f"\n[*] Existing naming statistics dictionary found: {stats_path}. Skipping mining phase.")

    if 'candidate_generation' not in config:
        config['candidate_generation'] = {}
    config['candidate_generation']['naming_stats_path'] = stats_path

    print("\n[*] Loading Engines and Models...")
    mlm_engine_name = config['models'].get('mlm_engine', 'microsoft/codebert-base-mlm')
    mlm_engine = MLMEngine(mlm_engine_name)

    llm_name = config['models'].get('llm_generator', 'models/qwen2.5-1.5b-code')
    llm_client = LocalLLMClient(model_name=llm_name)

    lightweight_generator = LightweightCandidateGenerator(
        mlm_engine=mlm_engine,
        analyzer=analyzer,
        config=config,
        llm_client=llm_client,
    )

    heavyweight_generator = HeavyWeightCandidateGenerator(
        embedder=mlm_engine,
        llm_client=llm_client,
        analyzer=analyzer,
        config=config
    )

    model_configs = config['models'].get('target_models', {})
    model_zoo = ModelZoo(
        model_configs=model_configs,
        eval_mode=args.mode,
        config=config,
    )
    transformer = CodeTransformer()

    def get_all_identifiers_fn(code_str: str) -> list:
        # Extracts non-main identifiers from the given code string.
        data = analyzer.extract_identifiers(code_str.encode("utf-8"))
        return [name for name in data.keys() if name != "main"]

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        # Renames identifiers within the code string according to a target renaming map.
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map, analyzer=analyzer)

    config['run_params']['algorithm'] = config['attack'].get('algorithm', 'beam')
    config['run_params']['iterations'] = config['attack'].get('iterations', 25)

    evaluator = SCARAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        mlm_gen=lightweight_generator,
        llm_gen=heavyweight_generator,
        rename_fn=rename_fn,
        mode=args.mode,
        config=config
    )

    loader = DatasetLoader()
    print(f"\n[*] Loading dataset in {args.mode} mode...")
    run_params = config['run_params']
    dataset = loader.load_parquet_dataset(
        filepath=run_params['dataset'],
        mode=args.mode,
        max_samples=run_params['samples'],
        label_map_path=run_params.get('label_map'),
        random_seed=config['global'].get('random_seed', 42)
    )

    evaluator.attack(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial sample generation attack tool")

    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary",
                        help="Select run mode: binary or multi")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="System configuration file path (YAML format)")

    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        parser.error(f"❌ Configuration file not found: {args.config}. Please ensure the file exists!")

    run_params = config.get('run_params', {})
    if args.mode == "multi" and run_params.get('label_map') is None:
        parser.error("❌ When --mode=multi, the label_map must be provided")

    seed = config.get('global', {}).get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(args, config)