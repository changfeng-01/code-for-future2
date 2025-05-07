import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
import re


# -------------------- 配置参数 --------------------
class Config:
    # 模型参数
    max_length = 1024
    embed_dim = 128
    num_heads = 8
    ff_dim = 512
    num_layers = 6
    dropout = 0.1

    # 训练参数
    batch_size = 32
    lr = 3e-5
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- 数据预处理 --------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        sequences: list of tuples (sequence, seq_type)
        labels: list of tuples (type_label, enzyme_label)
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = HybridTokenizer()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, seq_type = self.sequences[idx]
        type_label, enzyme_label = self.labels[idx]

        # 编码序列
        input_ids = self.tokenizer.encode(seq, seq_type)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "seq_type": torch.tensor(seq_type, dtype=torch.long),
            "labels": {
                "type": torch.tensor(type_label, dtype=torch.long),
                "enzyme": torch.tensor(enzyme_label, dtype=torch.float)
            }
        }


class HybridTokenizer:
    def __init__(self):
        # DNA: 0-3, Protein: 4-23, Special: 24-27
        self.char_dict = {
            # DNA编码
            'A': 0, 'T': 1, 'C': 2, 'G': 3,
            # Protein编码
            'M': 4, 'E': 5, 'L': 6, 'K': 7, 'Q': 8, 'R': 9, 'S': 10,
            'V': 11, 'D': 12, 'N': 13, 'P': 14, 'F': 15, 'W': 16,
            'H': 17, 'I': 18, 'Y': 19, 'G': 20, 'A': 21, 'T': 22, 'C': 23,
            # 特殊字符
            '<pad>': 24, '<cls>': 25, '<sep>': 26, '<unk>': 27
        }

    def encode(self, sequence, seq_type):
        # 序列标准化
        cleaned = self._clean_sequence(sequence, seq_type)
        # 截断/填充
        encoded = [self.char_dict.get(c, 27) for c in cleaned[:Config.max_length]]
        if len(encoded) < Config.max_length:
            encoded += [24] * (Config.max_length - len(encoded))
        return encoded

    def _clean_sequence(self, seq, seq_type):
        """根据序列类型进行清洗"""
        seq = seq.upper()
        if seq_type == 0:  # DNA
            return re.sub(r'[^ATCG]', '', seq)
        else:  # Protein
            return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)


# -------------------- 模型架构 --------------------
class BioFusionLayer(nn.Module):
    """生物特征融合层"""

    def __init__(self):
        super().__init__()
        self.dna_conv = nn.Conv1d(4, 32, kernel_size=9, padding=4)
        self.prot_conv = nn.Conv1d(20, 32, kernel_size=9, padding=4)
        self.attention = nn.MultiheadAttention(32, 4)

    def forward(self, x, seq_type):
        # DNA特征提取
        if seq_type == 0:
            x = F.one_hot(x.long(), num_classes=4).float().permute(0, 2, 1)
            x = self.dna_conv(x)
        # Protein特征提取
        else:
            x = F.one_hot(x.long(), num_classes=20).float().permute(0, 2, 1)
            x = self.prot_conv(x)

        # 注意力机制
        x = x.permute(2, 0, 1)  # [seq_len, batch, features]
        attn_out, _ = self.attention(x, x, x)
        return attn_out.mean(dim=0)


class MetaProClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取
        self.fusion = BioFusionLayer()

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=Config.ff_dim,
            dropout=Config.dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, Config.num_layers)

        # 多任务头
        self.type_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.enzyme_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, seq_type):
        # 生物特征融合
        x = self.fusion(input_ids, seq_type)

        # Transformer处理
        x = self.transformer(x.unsqueeze(1)).squeeze(1)

        # 分类任务
        type_logits = self.type_classifier(x)
        enzyme_prob = self.enzyme_classifier(x)

        return type_logits, enzyme_prob


# -------------------- 训练流水线 --------------------
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs, labels):
        type_logits, enzyme_prob = outputs
        type_loss = self.ce_loss(type_logits, labels['type'])

        # 仅对蛋白质样本计算酶损失
        enzyme_mask = (labels['type'] == 1)
        if torch.sum(enzyme_mask) > 0:
            enzyme_loss = self.bce_loss(
                enzyme_prob[enzyme_mask],
                labels['enzyme'][enzyme_mask]
            )
        else:
            enzyme_loss = 0.0

        return type_loss + 0.5 * enzyme_loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        inputs = batch['input_ids'].to(device)
        seq_types = batch['seq_type'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        # 前向传播
        type_logits, enzyme_prob = model(inputs, seq_types)

        # 计算损失
        loss = criterion((type_logits, enzyme_prob), labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# -------------------- 推理示例 --------------------
def predict_sequence(model, sequence):
    tokenizer = HybridTokenizer()

    # 第一阶段：序列类型判断
    dna_test = re.sub(r'[^ATCG]', '', sequence.upper())
    is_dna = len(dna_test) / len(sequence) > 0.9 if sequence else False

    # 编码序列
    seq_type = 0 if is_dna else 1
    input_ids = tokenizer.encode(sequence, seq_type)
    input_tensor = torch.tensor([input_ids], device=Config.device)

    # 模型预测
    with torch.no_grad():
        type_logits, enzyme_prob = model(input_tensor, torch.tensor([seq_type]))

    # 解析结果
    type_pred = torch.argmax(type_logits).item()
    result = {
        "sequence_type": "DNA" if type_pred == 0 else "Protein",
        "confidence": torch.softmax(type_logits, dim=1)[0][type_pred].item()
    }

    if type_pred == 1:
        result["is_enzyme"] = enzyme_prob.item() > 0.5
        result["enzyme_probability"] = enzyme_prob.item()

    return result


# -------------------- 初始化与执行 --------------------
if __name__ == "__main__":
    # 示例数据
    dna_sample = ("ATCGATCGATCG", 0)
    protein_sample = ("MADEEVQRE", 1)
    labels = [(0, -1), (1, 1)]

    # 创建数据集
    dataset = SequenceDataset([dna_sample, protein_sample], labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 初始化模型
    model = MetaProClassifier().to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    criterion = CustomLoss()

    # 训练循环
    for epoch in range(Config.epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, Config.device)
        print(f"Epoch {epoch + 1} Loss: {loss:.4f}")