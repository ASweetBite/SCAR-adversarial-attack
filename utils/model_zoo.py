import logging
import os
import json
import re
from typing import List, Tuple

from torch import nn

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer

logger = logging.getLogger(__name__)

LOCAL_BASE_MODELS_MAP = {
    "graphcodebert": "./models/graphcodebert-base",
    "unixcoder": "./models/unixcoder-base",
    "codebert": "./models/codebert-base"
}


class ModelZooQueryTracker:
    """
    黑盒查询拦截器：利用代理模式透明地包装 ModelZoo。
    严格记录针对特定大模型的所有预测查询开销（包含单步预测和批处理预测）。
    """
    def __init__(self, model_zoo):
        self._model_zoo = model_zoo
        self._query_count = 0

    def reset_counter(self):
        self._query_count = 0

    def get_query_count(self):
        return self._query_count

    def predict(self, *args, **kwargs):
        self._query_count += 1
        return self._model_zoo.predict(*args, **kwargs)

    def batch_predict(self, codes, *args, **kwargs):
        self._query_count += len(codes)
        return self._model_zoo.batch_predict(codes, *args, **kwargs)

    def predict_label_conf(self, *args, **kwargs):
        self._query_count += 1
        return self._model_zoo.predict_label_conf(*args, **kwargs)

    def __getattr__(self, name):
        # 将其他所有未重写的方法/属性（如 model_names）透明转发给底层的 model_zoo
        return getattr(self._model_zoo, name)


import os
import json
import torch
import numpy as np
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from gensim.models import KeyedVectors
import torch.nn.functional as F
import pydot
import networkx as nx
feature_num = 6

class TextCNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(TextCNN, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        classifier_dropout = 0.1
        self.convs = nn.ModuleList(
            [nn.Conv2d(feature_num, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(4 * self.num_filters * len(self.filter_sizes), num_classes)
        self.vector_weights = nn.Parameter(torch.ones(4))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out_list = []
        for i, vec in enumerate(x):
            out = vec.float()
            hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
            hidden_state = hidden_state * self.vector_weights[i]
            out_list.append(hidden_state)

        combined_hidden_state = torch.cat(out_list, dim=1)
        out = self.dropout(combined_hidden_state)
        out = self.fc(out)
        return out, combined_hidden_state

class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict):
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.eval_mode = eval_mode
        self.max_seq_len = run_cfg.get('max_seq_len', 512)

        if self.eval_mode == "binary":
            self.num_classes = 2
            print("[*] ModelZoo running in BINARY mode (Forcing num_classes = 2)")
        else:
            self.num_classes = run_cfg.get('num_classes', 16)
            print(f"[*] ModelZoo running in MULTI mode (num_classes = {self.num_classes})")

        self.models = {}
        self.model_names = list(model_configs.keys())

        # =====================================================================
        # 1. 动态加载 DFG 特征提取器
        # =====================================================================
        self.analyzer = None
        if any("vulcnn" in name.lower() for name in self.model_names):
            print("[*] Detected VulCNNPlus. Initializing Word2Vec Extractor...")
            self.vulcnn_w2v = KeyedVectors.load_word2vec_format("./dataset/data_model.bin", binary=True)
        if any("graphcodebert" in name.lower() for name in self.model_names):
            print("[*] Detected GraphCodeBERT in targets. Initializing DFG Extractor...")
            # 假设 IdentifierAnalyzer 已经处理好离线问题
            self.analyzer = IdentifierAnalyzer(lang="cpp")

        for name, path in model_configs.items():
            print(f"\n[*] Loading Model[{name}] from {path}...")

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[!] CRITICAL: Target path {path} not found for model '{name}'. Aborting init.")

            try:
                adapter_config_path = os.path.join(path, "adapter_config.json")
                name_lower = name.lower()
                if "vulcnn" in name_lower:
                    print(f"    |- Loading VulCNNPlus TextCNN Architecture...")
                    model = TextCNN(hidden_size=128)
                    # path 应该指向你重新训练得到的 _best_model.pth
                    state_dict = torch.load(path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.models[name] = {"type": "cnn", "tokenizer": None, "model": model}
                    print(f"[+] Successfully loaded {name} to {self.device}")
                    continue
                # 先确定基础模型路径，用来加载 Tokenizer 和 Base Model
                if os.path.exists(adapter_config_path):
                    print(f"    |- Detected LoRA adapter. Parsing config...")
                    # =========================================================
                    # 纯离线推断基础模型路径
                    # =========================================================
                    if "graphcodebert" in name_lower:
                        base_model_path = LOCAL_BASE_MODELS_MAP["graphcodebert"]
                    elif "unixcoder" in name_lower:
                        base_model_path = LOCAL_BASE_MODELS_MAP["unixcoder"]
                    else:
                        base_model_path = LOCAL_BASE_MODELS_MAP["codebert"]

                    if not os.path.exists(base_model_path):
                        raise FileNotFoundError(f"本地基础模型缺失，请检查路径: {base_model_path}")

                    print(f"    |- [*] Offline mapping to local base model: {base_model_path}")

                    # [*] 加载 Tokenizer (从本地基础模型加载)
                    # 添加 local_files_only=True 防止任何联网尝试
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_path,
                        trust_remote_code=True,
                        use_fast=True,
                        local_files_only=True
                    )

                    # [*] Loading Standard HF Classifier Skeleton...
                    print(f"    |- Loading Base Model Skeleton from: {base_model_path}")
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_path,
                        num_labels=self.num_classes,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                        local_files_only=True  # 强制离线
                    )

                    # [*] Loading LoRA Adapters...
                    print(f"    |- Injecting LoRA Adapters from {path}...")
                    # PeftModel 默认从本地 path 读取
                    model = PeftModel.from_pretrained(base_model, path)
                    print("    |- ✅ LoRA weights and Custom Classifier successfully injected.")

                else:
                    # =========================================================
                    # 兼容非 LoRA 模型的普通加载逻辑 (假设 path 里就是完整的离线模型)
                    # =========================================================
                    print(f"    |- Loading standard offline HF model...")

                    tokenizer = AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=True,
                        use_fast=True,
                        local_files_only=True
                    )

                    model = AutoModelForSequenceClassification.from_pretrained(
                        path,
                        num_labels=self.num_classes,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                        local_files_only=True  # 强制离线
                    )
                    print("    |- ✅ Standard offline model loaded successfully.")

                model.to(self.device)
                model.eval()
                self.models[name] = {"type": "transformer", "tokenizer": tokenizer, "model": model}
                print(f"[+] Successfully loaded {name} to {self.device}")

            except Exception as e:
                import traceback
                print("\n" + "=" * 50)
                print(f"🚨 FAILED TO LOAD MODEL: {name}")
                print(f"Path: {path}")
                print("Error Traceback:")
                traceback.print_exc()
                print("=" * 50 + "\n")
                raise RuntimeError(f"Failed to load model '{name}'. Execution halted.") from e

    def _generate_image_from_dot_string(self, dot_text: str):
        graphs = pydot.graph_from_dot_data(dot_text)
        graph = nx.drawing.nx_pydot.from_pydot(graphs[0])

        labels_dict = nx.get_node_attributes(graph, 'label')
        labels_code = dict()
        for label, all_code in labels_dict.items():
            if not isinstance(all_code, str): continue
            all_code = re.sub(r'<SUB>\d+</SUB>', '', all_code)[1:-1]
            try:
                code = all_code[all_code.index(",") + 1:-1].split("\\n")[0]
            except ValueError:
                code = all_code
            code = code.replace("static void", "void")
            labels_code[label] = code

        degree_cen_dict = nx.degree_centrality(graph)
        closeness_cen_dict = nx.closeness_centrality(graph)
        harmonic_cen_dict = nx.harmonic_centrality(graph)

        G = nx.DiGraph()
        G.add_nodes_from(graph.nodes())
        G.add_edges_from(graph.edges())

        try:
            katz_cen_dict = nx.katz_centrality(G, max_iter=500, tol=1e-4)
        except:
            katz_cen_dict = {}
        try:
            eigenvector_cen_dict = nx.eigenvector_centrality(G, max_iter=500, tol=1e-4)
        except:
            eigenvector_cen_dict = {}
        try:
            betweenness_cen_dict = nx.betweenness_centrality(graph)
        except:
            betweenness_cen_dict = nx.degree_centrality(graph)

        degree_ch, closeness_ch, betweenness_ch = [], [], []
        eigenvector_ch, harmonic_ch, katz_ch = [], [], []

        for label, code in labels_code.items():
            words = code.strip().split()
            vecs = [self.vulcnn_w2v[w] for w in words if w in self.vulcnn_w2v.key_to_index]
            line_vec = np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(self.vulcnn_w2v.vector_size)

            degree_ch.append(degree_cen_dict.get(label, 0) * line_vec)
            closeness_ch.append(closeness_cen_dict.get(label, 0) * line_vec)
            betweenness_ch.append(betweenness_cen_dict.get(label, 0) * line_vec)
            eigenvector_ch.append(eigenvector_cen_dict.get(label, 0) * line_vec)
            harmonic_ch.append(harmonic_cen_dict.get(label, 0) * line_vec)
            katz_ch.append(katz_cen_dict.get(label, 0) * line_vec)

        return [degree_ch, closeness_ch, katz_ch, betweenness_ch, eigenvector_ch, harmonic_ch]

    def _fast_vulcnn_encode(self, sample_id: str, rename_mapping: dict):
        # 注意：这里需要指向你存原版 4 种 dot 文件的根目录
        base_dir = "./my_dataset/processed/graphs"
        graph_types = ['pdg', 'cfg', 'ddg', 'ast']
        all_views_features = []

        for g_type in graph_types:
            dot_path = os.path.join(base_dir, g_type, f"{sample_id}.dot")
            if not os.path.exists(dot_path):
                raise FileNotFoundError(f"找不到原始 DOT 文件: {dot_path}")

            with open(dot_path, "r", encoding="utf-8") as f:
                dot_text = f.read()

            # 正则替换标识符
            if rename_mapping:
                for old_var, new_var in rename_mapping.items():
                    dot_text = re.sub(r'\b' + re.escape(old_var) + r'\b', new_var, dot_text)

            view_channels = self._generate_image_from_dot_string(dot_text)
            all_views_features.append(view_channels)

        return all_views_features

    def _encode_graphcodebert(self, code: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code_bytes = code.encode('utf-8')
        dfg_nodes, dfg_to_code_chars, dfg_to_dfg = self.analyzer.extract_dataflow(code_bytes)

        # ====================================================================
        # 🌟 关键对齐：严格匹配训练脚本 GraphCodeBERTVulDataset 的硬编码长度
        # ====================================================================
        args_code_length = 384
        args_dfg_length = self.max_seq_len - args_code_length  # 128

        # 1. 截断图节点 (对齐训练集)
        dfg_nodes = dfg_nodes[:args_dfg_length]
        dfg_to_code_chars = dfg_to_code_chars[:args_dfg_length]
        dfg_to_dfg = [[e for e in edges if e < args_dfg_length] for edges in dfg_to_dfg[:args_dfg_length]]

        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("[!] 必须加载 Fast 版本的 Tokenizer！")

        # 2. 文本截断必须锁定 384 (对齐训练集)
        encoded = tokenizer(
            code,
            truncation=True,
            max_length=args_code_length,
            return_offsets_mapping=True
        )
        text_ids = encoded['input_ids']
        offsets = encoded['offset_mapping']
        text_len = len(text_ids)

        # 3. 构建 Subwords 映射
        dfg_to_subwords = []
        for (start_char, end_char) in dfg_to_code_chars:
            subword_indices = []
            for idx, (o_start, o_end) in enumerate(offsets):
                if o_start == o_end: continue
                if o_start < end_char and o_end > start_char:
                    subword_indices.append(idx)
            dfg_to_subwords.append(subword_indices)

        # 4. 组装 IDs
        input_ids = text_ids + [tokenizer.unk_token_id] * len(dfg_nodes)
        position_ids = [i + tokenizer.pad_token_id + 1 for i in range(text_len)] + [0] * len(dfg_nodes)

        # 5. Padding
        pad_len = self.max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        position_ids += [tokenizer.pad_token_id] * pad_len

        # 6. 构建 Attention Mask
        attn_mask = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.bool_)

        attn_mask[:text_len, :text_len] = True

        for idx, token_id in enumerate(input_ids):
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                attn_mask[idx, :text_len + len(dfg_nodes)] = True
                attn_mask[:text_len + len(dfg_nodes), idx] = True

        for dfg_idx, subword_idxs in enumerate(dfg_to_subwords):
            matrix_dfg_idx = text_len + dfg_idx
            for sub_idx in subword_idxs:
                attn_mask[matrix_dfg_idx, sub_idx] = True
                attn_mask[sub_idx, matrix_dfg_idx] = True

        for dfg_idx, edges in enumerate(dfg_to_dfg):
            matrix_dfg_idx = text_len + dfg_idx
            for source_dfg_idx in edges:
                matrix_source_idx = text_len + source_dfg_idx
                attn_mask[matrix_dfg_idx, matrix_source_idx] = True
                attn_mask[matrix_source_idx, matrix_dfg_idx] = True

        # 7. 转换 4D Float 掩码
        float_mask = np.where(attn_mask, 0.0, -10000.0).astype(np.float32)
        float_mask_4d = np.expand_dims(float_mask, axis=(0, 1))

        return (
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor(float_mask_4d, dtype=torch.float32),
            torch.tensor([position_ids], dtype=torch.long)
        )

    def _encode_unixcoder(self, code: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """为 UniXcoder 构建特征：强制注入 <encoder-only> 控制符"""
        tokens = tokenizer.tokenize(code)
        tokens = tokens[:self.max_seq_len - 4]

        mode_token = "<encoder-only>"
        source_tokens = [tokenizer.bos_token, mode_token, tokenizer.eos_token] + tokens + [tokenizer.eos_token]
        input_ids = tokenizer.convert_tokens_to_ids(source_tokens)

        padding_length = self.max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * (self.max_seq_len - padding_length) + [0] * padding_length

        return (
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor([attention_mask], dtype=torch.long)
        )

    # =========================================================================
    # 推断接口动态路由 (Prediction Dispatcher)
    # =========================================================================

    # [修改签名] 加上 **kwargs
    def predict(self, code: str, target_model: str, **kwargs) -> Tuple[List[float], int]:
        m = self.models.get(target_model)
        if m is None:
            return [1.0, 0.0], -1

        model = m["model"]
        model_name_lower = target_model.lower()

        with torch.no_grad():
            # =====================================================
            # [新增] VulCNNPlus 推理通道
            # =====================================================
            if m.get("type") == "cnn" and "vulcnn" in model_name_lower:
                sample_id = kwargs.get("sample_id")
                rename_mapping = kwargs.get("rename_mapping", {})

                if not sample_id:
                    raise ValueError("VulCNN 推理需要通过 kwargs 传入 'sample_id' 定位原始 dot 图。")

                # 1. 生成 4个视图的原始/变异特征
                all_views = self._fast_vulcnn_encode(sample_id, rename_mapping)

                # 2. 补齐与对齐 Dataset 形状 -> 4 x (1, 6, 100, 128)
                max_len, hidden_size = 100, 128
                batched_vectors_tuple = ()
                for view_channels in all_views:
                    vectors = np.zeros(shape=(6, max_len, hidden_size))
                    for j in range(6):
                        nodes_count = len(view_channels[j])
                        for i in range(min(nodes_count, max_len)):
                            vectors[j][i] = view_channels[j][i]
                    t = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0).to(self.device)
                    batched_vectors_tuple += (t, )

                # 3. 推理预测
                outputs, _ = model(batched_vectors_tuple)
                # TextCNN 的输出可能不是 SequenceClassifierOutput
                logits = outputs if not hasattr(outputs, "logits") else outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().tolist()

            # =====================================================
            # [原逻辑保留] GraphCodeBERT
            # =====================================================
            elif "graphcodebert" in model_name_lower:
                tokenizer = m["tokenizer"]
                input_ids, attn_mask, position_ids = self._encode_graphcodebert(code, tokenizer)
                outputs = model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn_mask.to(self.device),
                    position_ids=position_ids.to(self.device)
                )
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy().tolist()

            # =====================================================
            # [原逻辑保留] UniXCoder
            # =====================================================
            elif "unixcoder" in model_name_lower:
                tokenizer = m["tokenizer"]
                input_ids, attn_mask = self._encode_unixcoder(code, tokenizer)
                outputs = model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn_mask.to(self.device)
                )
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy().tolist()

            # =====================================================
            # [原逻辑保留] CodeBERT / 兜底
            # =====================================================
            else:
                tokenizer = m["tokenizer"]
                inputs = tokenizer(
                    code, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding="max_length"
                ).to(self.device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy().tolist()

            # --- 公共后处理 ---
            pred_label = int(np.argmax(probs))
            if self.eval_mode == "binary" and pred_label == 0:
                pred_label = -1

        return probs, pred_label

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32, **kwargs) -> Tuple[
        List[List[float]], List[int]]:
        """安全的 Batch Predict: 兼容各种非标准架构的特征重组"""
        m = self.models.get(target_model)
        if m is None:
            return [[1.0, 0.0]] * len(codes), [-1] * len(codes)

        tokenizer = m.get("tokenizer")
        model = m["model"]
        model_name_lower = target_model.lower()
        model_type = m.get("type", "transformer")

        # =================================================================
        # [新增] 针对 VulCNNPlus Batch 处理的参数校验与提取
        # =================================================================
        sample_ids = kwargs.get("sample_ids", [])
        rename_mappings = kwargs.get("rename_mappings", [])

        if model_type == "cnn" and "vulcnn" in model_name_lower:
            # 必须保证传入的 sample_ids 列表长度与代码列表一致
            if not sample_ids or len(sample_ids) != len(codes):
                raise ValueError(
                    f"[!] 对于 VulCNNPlus 批量预测，必须在 kwargs 中传入等长的 'sample_ids' 列表！(期待 {len(codes)} 个)")

            # 如果没传 rename_mappings，就用空字典补齐 (说明是单纯测原版数据)
            if not rename_mappings:
                rename_mappings = [{}] * len(codes)
            elif len(rename_mappings) != len(codes):
                raise ValueError("[!] 'rename_mappings' 列表长度必须与 codes 列表相等！")

        all_probs, all_preds = [], []

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]

            with torch.no_grad():
                # =========================================================
                # [新增] VulCNNPlus 的 Batch 组装分支
                # =========================================================
                if model_type == "cnn" and "vulcnn" in model_name_lower:
                    b_sample_ids = sample_ids[i:i + batch_size]
                    b_rename_mappings = rename_mappings[i:i + batch_size]

                    # 准备 4 个视图的容器，后续转为 (Batch, 6, 100, 128)
                    b_view_0, b_view_1, b_view_2, b_view_3 = [], [], [], []
                    max_len, hidden_size = 100, 128

                    # 遍历 Batch 里的每一条数据
                    for s_id, r_map in zip(b_sample_ids, b_rename_mappings):
                        # 调用我们写的极速引擎，获取 4 个视图
                        all_views = self._fast_vulcnn_encode(s_id, r_map)

                        # 执行 Padding
                        padded_views = []
                        for view_channels in all_views:
                            vectors = np.zeros(shape=(6, max_len, hidden_size))
                            for j in range(6):
                                nodes_count = len(view_channels[j])
                                for idx in range(min(nodes_count, max_len)):
                                    vectors[j][idx] = view_channels[j][idx]
                            padded_views.append(vectors)

                        # 分别存入 4 个视图的 Batch 列表中
                        b_view_0.append(padded_views[0])
                        b_view_1.append(padded_views[1])
                        b_view_2.append(padded_views[2])
                        b_view_3.append(padded_views[3])

                    # 转为 FloatTensor (形状: [BatchSize, 6, 100, 128])
                    t0 = torch.tensor(np.array(b_view_0), dtype=torch.float32).to(self.device)
                    t1 = torch.tensor(np.array(b_view_1), dtype=torch.float32).to(self.device)
                    t2 = torch.tensor(np.array(b_view_2), dtype=torch.float32).to(self.device)
                    t3 = torch.tensor(np.array(b_view_3), dtype=torch.float32).to(self.device)

                    # 组装为包含 4 个 Tensor 的 Tuple 喂给 TextCNN
                    batched_vectors_tuple = (t0, t1, t2, t3)
                    outputs, _ = model(batched_vectors_tuple)

                    # TextCNN 返回的第一个元素就是 logits
                    logits = outputs

                # =========================================================
                # [原有] GraphCodeBERT
                # =========================================================
                elif "graphcodebert" in model_name_lower:
                    b_input_ids, b_attn_mask, b_position_ids = [], [], []
                    for c in batch_codes:
                        i_ids, a_mask, p_ids = self._encode_graphcodebert(c, tokenizer)
                        b_input_ids.append(i_ids)
                        b_attn_mask.append(a_mask)
                        b_position_ids.append(p_ids)

                    outputs = model(
                        input_ids=torch.cat(b_input_ids, dim=0).to(self.device),
                        attention_mask=torch.cat(b_attn_mask, dim=0).to(self.device),
                        position_ids=torch.cat(b_position_ids, dim=0).to(self.device)
                    )
                    logits = outputs.logits

                # =========================================================
                # [原有] UniXCoder
                # =========================================================
                elif "unixcoder" in model_name_lower:
                    b_input_ids, b_attn_mask = [], []
                    for c in batch_codes:
                        i_ids, a_mask = self._encode_unixcoder(c, tokenizer)
                        b_input_ids.append(i_ids)
                        b_attn_mask.append(a_mask)

                    outputs = model(
                        input_ids=torch.cat(b_input_ids, dim=0).to(self.device),
                        attention_mask=torch.cat(b_attn_mask, dim=0).to(self.device)
                    )
                    logits = outputs.logits

                # =========================================================
                # [原有] CodeBERT 等标准架构
                # =========================================================
                else:
                    inputs = tokenizer(
                        batch_codes, return_tensors="pt", truncation=True, max_length=self.max_seq_len,
                        padding="max_length"
                    ).to(self.device)
                    outputs = model(**inputs)
                    logits = outputs.logits

                # =========================================================
                # [公共处理] 统一处理 Softmax 与标签预测
                # =========================================================
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_list = probs.tolist() if probs.ndim == 2 else [probs.tolist()]
                preds_list = [int(np.argmax(p)) for p in probs_list]

                if self.eval_mode == "binary":
                    preds_list = [1 if p == 1 else -1 for p in preds_list]

                all_probs.extend(probs_list)
                all_preds.extend(preds_list)

        return all_probs, all_preds

    def predict_label_conf(self, code: str, label: int, target_model: str, **kwargs) -> float:
        probs, _ = self.predict(code, target_model, **kwargs)
        return probs[label] if label < len(probs) else 0.0