# SCAR-adversarial-attack
## Dataset

For the fine-tuning process, the dataset must meet specific structural and format requirements to ensure compatibility with the training script and the underlying Transformer models.

1. **Format:** The dataset **must** be in the `.parquet` format. Parquet is highly efficient for storing columnar data and provides faster read times compared to CSV or JSON.
2. **Required Columns:** Your `.parquet` file must contain the following columns exactly as named:
* **`func` (String):** This column contains the raw, complete source code of the function. It serves as the primary input feature for the model.
* **`vul` (Integer or Boolean):** This column represents the ground truth label. It must be mapped to integer values during processing:
* `0`: Indicates that the function is **safe** (non-vulnerable).
* `1`: Indicates that the function contains a **vulnerability**.


3. **Data Quality Constraints:**
* **Length:** The training script automatically filters out code snippets longer than 4,000 characters. To avoid data loss, ensure that the majority of your functions fit within this limit, or pre-process them into manageable chunks.
* **Balance:** While the script processes the data as-is, for optimal training results on binary classification tasks, it is highly recommended to maintain a reasonable balance between positive (vulnerable) and negative (safe) samples.

## Fine-turn Model
We use full train data for fine-tuning CodeBERT and valid data for evaluating.
```bash
python train_model.py \
    --train_data_file data/train.parquet \
    --eval_data_file data/valid.parquet \
    --output_dir ./models \
    --model_name CodeBERT \
    --model_name_or_path microsoft/codebert-base \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 1 \
    --num_epochs 3 \
    --seed 42

```

## Attack

All detailed configurations for the attack process—including candidate generation quotas (MLM/LLM), optimizer algorithms (Beam Search, Greedy, GA), target models, and file paths—are managed via the YAML configuration file. Please refer to `config/config.yaml` for specific details and customizable parameters.

### Binary classification

To execute the adversarial attack in binary classification mode (evaluating whether a function is simply safe or vulnerable), use the `--mode binary` flag along with your configuration file.

```bash
python main.py \
    --mode binary \
    --config config/config.yaml
```

In this mode, the system only relies on the `func` and `vul` columns in your dataset.

### Multi classification

To execute the adversarial attack in multi-class classification mode (evaluating specific vulnerability types or CWEs), use the `--mode multi` flag:

```bash
python main.py \
    --mode multi \
    --config config/config.yaml
```

**Additional Requirements for Multi-class Mode:**

1. **Extra Dataset Column (`cwe`):** 
   In addition to the `func` and `vul` columns, your `.parquet` dataset **must** include a `cwe` column (String type). This column specifies the exact vulnerability identifier (e.g., `CWE-79`, `CWE-89`) for vulnerable samples. For safe samples, this column can be empty, `"safe"`, or `None`.
2. **Label Map File (`label_map.json`):** 
   You must provide a label mapping file to tell the framework which integer ID corresponds to which CWE class in your specific target model. 
   * This path must be specified under `run_params: label_map:` in your `config.yaml`.
   * The JSON file should look something like this:
     ```json
     {
       "id2label": {
         "0": "Safe",
         "1": "CWE-119",
         "2": "CWE-120",
         "3": "CWE-79"
       }
     }
     ```
## Analyze result

Use `analyze_rename_stealthiness.py` after you run the attack program:
```bash
python analyze_rename_stealthiness.py \
    --input_json "results/attack_results.jsonl" \
    --config "config/config.yaml" \
    --out "evaluation_results" \
    --top-k 20
```
`--top-k`: How many sets of data are included in the visual chart

## Contact

Feel free to contact Ruizhe Ren (lwdcfxy22401@163.com), Hongbo Qu (2267606106@qq.com), Chengfeng Ren (rcf@stu.edu.ouc.cn) if you have any further questions.