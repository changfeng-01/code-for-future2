import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# 加载数据
df = pd.read_csv("D:\统计数模大赛\copy.csv")

# 对 object 类型数据进行编码
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 处理缺失值，这里用均值填充
df.fillna(df.mean(), inplace=True)

# 划分特征和目标变量，假设目标变量为 'Depression'
X = df.drop('Depression', axis=1)
y = df['Depression']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据平衡处理
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # 添加序列维度，这里假设序列长度为 1
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)  # 调整 Dropout 率
        self.dropout = nn.Dropout(0.3)  # 在全连接层前添加 Dropout
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取序列的最后一个输出
        out = self.dropout(out)  # 应用 Dropout
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# 模型参数，使用修改后的结构
input_size = X_train.shape[1]
hidden_size = 128
num_layers = 3
output_size = 1
learning_rate = 0.001
num_epochs = 100  # 增加训练轮数

# 初始化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 调整损失函数权重，处理不平衡数据
pos_weight = torch.tensor([y_train_tensor.size(0) / y_train_tensor.sum()])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)  # L2 正则化减少过拟合
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)  # 学习率调度器

# 删除旧的模型文件
model_path = 'lstm_model.pth'
if os.path.exists(model_path):
    os.remove(model_path)

# 训练模型
train_losses = []
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    scheduler.step(loss)  # 更新学习率

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), model_path)
        print(f"在第 {epoch + 1} 个 epoch 保存了当前最优模型。")

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

# 绘制训练损失曲线
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 加载最优模型
model.load_state_dict(torch.load(model_path))

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_tensor)
    y_pred = (y_pred_proba > 0.5).float()  # 可以尝试调整阈值来提高召回率，这里先使用 0.5

    accuracy = accuracy_score(y_test_tensor, y_pred)
    precision = precision_score(y_test_tensor, y_pred)
    recall = recall_score(y_test_tensor, y_pred)
    f1 = f1_score(y_test_tensor, y_pred)

    print(f'模型准确率：{accuracy:.4f}')
    print(f'模型精确率：{precision:.4f}')
    print(f'模型召回率：{recall:.4f}')
    print(f'模型 F1 分数：{f1:.4f}')

    # 生成ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test_tensor, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.title("中文标题")
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROS曲线')
    plt.legend(loc="lower right")
    plt.show()

    # 生成混淆矩阵图
    y_test_np = y_test_tensor.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()
    ConfusionMatrixDisplay.from_predictions(y_test_np, y_pred_np)
    plt.title('混淆矩阵')
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")

    plt.show()