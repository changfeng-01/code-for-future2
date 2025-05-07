import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 加载数据
df = pd.read_csv("D:\统计数模大赛\copy.csv")

# 对 object 类型数据进行编码
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 处理缺失值，这里用均值填充
df.fillna(df.mean(), inplace=True)

# 划分特征和目标变量
X = df.drop(['id', 'Depression'], axis=1)
y = df['Depression']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义更复杂的全连接神经网络
class ImprovedNet(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 初始化模型
input_size = X_train_tensor.shape[1]
model = ImprovedNet(input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 早停策略参数
best_val_loss = float('inf')
patience = 10
early_stopping_counter = 0

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}')

    # 早停策略
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print('Early stopping!')
            break

    scheduler.step()

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# 获取每个特征的影响权重（从第一层线性层获取）
weights = model.fc1.weight.data.numpy().mean(axis=0)
feature_weights = pd.Series(weights, index=X.columns)

# 打印每个特征的影响权重
print("每个特征对抑郁症的影响权重：")
print(feature_weights)

# 绘制特征权重柱状图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.title("中文标题")

plt.figure(figsize=(10, 6))
feature_weights.plot(kind='bar')
plt.title("不同原因对抑郁症成因的影响")
plt.xlabel('特征')
plt.ylabel('影响权重')
plt.xticks(rotation=45)
plt.show()