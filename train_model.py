import os
import gc
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import recall_score, precision_score, f1_score

# =============== Environment Configuration ===============
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# =============== Dataset Definition ===============

class VulnerabilityDataset(Dataset):
    def __init__(self, tokenizer, parquet_path, max_len=512):
        self.examples = []
        logger.info(f"Loading dataset file at {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df['label'] = df['vul'].astype(int)

        # Basic length truncation filtering
        df = df[df['func'].str.len() <= 4000].copy()

        funcs = df['func'].tolist()
        labels = df['label'].tolist()

        for func, label in tqdm(zip(funcs, labels), total=len(funcs), desc="Tokenizing"):
            encoded = tokenizer(
                func,
                truncation=True,
                max_length=max_len,
                padding='max_length'
            )
            self.examples.append({
                "input_ids": encoded['input_ids'],
                "attention_mask": encoded['attention_mask'],
                "label": label
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        return (
            torch.tensor(example['input_ids'], dtype=torch.long),
            torch.tensor(example['attention_mask'], dtype=torch.long),
            torch.tensor(example['label'], dtype=torch.long)
        )


# =============== Evaluation & Training Logic ===============

def evaluate(model, eval_dataset, args):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    logits_list = []
    y_trues = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        inputs = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        labels = batch[2].to(args.device)

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.mean().item()
            logits_list.append(outputs.logits.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

        nb_eval_steps += 1

    logits = np.concatenate(logits_list, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = np.argmax(logits, axis=-1)

    recall = recall_score(y_trues, y_preds, average='macro', zero_division=0)
    precision = precision_score(y_trues, y_preds, average='macro', zero_division=0)
    f1 = f1_score(y_trues, y_preds, average='macro', zero_division=0)

    return {
        "eval_loss": eval_loss / nb_eval_steps,
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }


def train(model, train_dataset, eval_dataset, args):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    logger.info(f"***** Running training for {args.model_name} *****")

    best_f1 = 0.0
    model.zero_grad()

    for epoch in range(int(args.num_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(bar):
            model.train()
            inputs = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            labels = batch[2].to(args.device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            bar.set_postfix({"loss": round(loss.item() * args.gradient_accumulation_steps, 4)})

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Evaluate at the end of each epoch
        results = evaluate(model, eval_dataset, args)
        logger.info(
            f"  Epoch {epoch + 1} Results: F1: {results['eval_f1']:.4f} | Recall: {results['eval_recall']:.4f} | Prec: {results['eval_precision']:.4f}")

        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            output_dir = os.path.join(args.output_dir, f"{args.model_name}_best_f1")
            os.makedirs(output_dir, exist_ok=True)

            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            logger.info(f"  [+] New best F1 ({best_f1:.4f})! Model saved to {output_dir}")


# =============== Main Control Flow ===============

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Code Vulnerability Detection")

    # Core data and path parameters
    parser.add_argument("--train_data_file", type=str, required=True, help="Path to training Parquet file")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to validation Parquet file")
    parser.add_argument("--output_dir", type=str, default="./models", help="Model output directory")

    # Model architecture selection
    parser.add_argument("--model_name", type=str, default="UniXcoder",
                        help="Model alias (used for naming the output folder)")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/unixcoder-base",
                        help="HuggingFace model path or local path")

    # Hyperparameter settings
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    logger.info("\n" + "=" * 50)
    logger.info(f"🚀 Initializing and starting training for: {args.model_name} ({args.model_name_or_path})")
    logger.info("=" * 50)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, bos_token="<s>", eos_token="</s>", sep_token="</s>",
        cls_token="<s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        additional_special_tokens=[]
    )

    # 2. Preprocess Data
    train_dataset = VulnerabilityDataset(tokenizer, args.train_data_file)
    eval_dataset = VulnerabilityDataset(tokenizer, args.eval_data_file)

    # 3. Load Model & Inject LoRA
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2,
                                                               trust_remote_code=True)
    model = get_peft_model(model, peft_config)
    model.to(args.device)

    # 4. Execute Training
    train(model, train_dataset, eval_dataset, args)

    # 5. Save Tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, f"{args.model_name}_best_f1"))

    # 6. Clean up Memory
    del model, tokenizer, train_dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"✅ {args.model_name} training completed, GPU memory cleared.\n")


if __name__ == "__main__":
    main()