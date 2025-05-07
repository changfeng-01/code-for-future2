# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from imblearn.over_sampling import SMOTE  # 处理极端不平衡数据
#
# # 加载数据
# data = pd.read_csv("scores.csv")
#
# # 定义特征类型（示例）
# numeric_features = ['age', 'sleep_hours', 'stress_score', 'physical_activity']
# categorical_features = ['gender', 'marital_status', 'education_level']
#
# # 高级预处理流水线
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(drop='if_binary'),  # 二分类特征直接0/1编码
#         categorical_features)
#     ], remainder='passthrough'  # 保留未定义的其他特征（如果有）
# )
#
# # 处理缺失值（分类型和数值型分别处理）
# data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
# for col in categorical_features:
#     data[col] = data[col].fillna(data[col].mode()[0])
#
# # 分割数据
# X = data.drop('depression', axis=1)
# y = data['depression']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#
# # 应用预处理
# X_train_processed = preprocessor.fit_transform(X_train)
# X_test_processed = preprocessor.transform(X_test)
#
# # 处理极端不平衡（如阳性样本<5%时使用SMOTE）
# if np.mean(y_train) < 0.05:
#     smote = SMOTE(sampling_strategy=0.3, k_neighbors=5)
#     X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
#
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
#     from tensorflow.keras.regularizers import l2
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.callbacks import EarlyStopping
#
#     # 获取预处理后的特征维度
#     input_dim = X_train_processed.shape[1]
#
#     # 构建DNN架构
#     model = Sequential([
#         Dense(128, activation='relu', input_shape=(input_dim,),
#               kernel_regularizer=l2(0.01)),  # L2正则化
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
#         BatchNormalization(),
#         Dropout(0.2),
#         Dense(32, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#
#     # 自定义优化器与早停
#     optimizer = Adam(learning_rate=0.001, clipvalue=0.5)  # 梯度裁剪防止爆炸
#     early_stop = EarlyStopping(monitor='val_auc', patience=10, mode='max',
#                                restore_best_weights=True)
#
#     # 编译模型
#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics=['AUC', 'Precision', 'Recall'])
#
#     # 动态类别权重计算（替代固定权重）
#     from sklearn.utils.class_weight import compute_class_weight
#
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
#
#     # 训练模型
#     history = model.fit(
#         X_train_processed, y_train,
#         epochs=100,
#         batch_size=64,
#         validation_split=0.15,
#         class_weight=class_weight_dict,
#         callbacks=[early_stop],
#         verbose=2
#     )
#
#     # 性能评估
#     test_pred = (model.predict(X_test_processed) > 0.5).astype("int32")
#     test_proba = model.predict(X_test_processed)
#
#     from sklearn.metrics import classification_report, roc_auc_score
#
#     print("测试集性能:")
#     print(classification_report(y_test, test_pred))
#     print(f"AUC-ROC: {roc_auc_score(y_test, test_proba):.3f}")
#
#     # 训练过程可视化
#     import matplotlib.pyplot as plt
#
#     plt.plot(history.history['auc'], label='Train AUC')
#     plt.plot(history.history['val_auc'], label='Validation AUC')
#     plt.title('模型训练过程监控')
#     plt.xlabel('Epoch')
#     plt.ylabel('AUC')
#     plt.legend()
#     plt.show()
#
#     import shap
#
#     # 创建背景数据集（随机抽取100个样本）
#     background = X_train_processed[np.random.choice(X_train_processed.shape[0], 100, replace=False)]
#
#     # 初始化解释器
#     explainer = shap.DeepExplainer(model, background)
#     shap_values = explainer.shap_values(X_test_processed[:50])  # 解释前50个测试样本
#
#     # 可视化全局特征重要性
#     shap.summary_plot(shap_values[0], X_test_processed[:50],
#                       feature_names=preprocessor.get_feature_names_out())

# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.impute import SimpleImputer
#
# # 对 object 类型数据进行编码
# categorical_cols = df.select_dtypes(include=['object']).columns
# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#
# # 使用均值填充缺失值
# imputer = SimpleImputer(strategy='mean')
# df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#
# # 划分特征和目标变量
# X = df.drop(['id', 'Depression'], axis=1)
# y = df['Depression']
#
# # 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # 构建逻辑回归模型
# logreg = LogisticRegression(max_iter=1000)
# logreg.fit(X_train, y_train)
# logreg_pred = logreg.predict(X_test)
#
# # 构建随机森林模型
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)
#
# # 构建支持向量机模型
# svm = SVC()
# svm.fit(X_train, y_train)
# svm_pred = svm.predict(X_test)
#
# # 构建多层感知器（MLP）深度学习模型
# model = Sequential()
# model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # 编译模型
# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
#
# # 训练模型
# model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
#
# # 在测试集上进行预测
# mlp_pred_proba = model.predict(X_test)
# mlp_pred = (mlp_pred_proba > 0.5).astype(int).flatten()
#
# # 评估模型
# def evaluate_model(y_test, y_pred, model_name):
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     print(f'{model_name} 评估结果:')
#     print(f'准确率: {accuracy:.4f}')
#     print(f'精确率: {precision:.4f}')
#     print(f'召回率: {recall:.4f}')
#     print(f'F1分数: {f1:.4f}')
#     print('-' * 30)
#
# evaluate_model(y_test, logreg_pred, '逻辑回归')
# evaluate_model(y_test, rf_pred, '随机森林')
# evaluate_model(y_test, svm_pred, '支持向量机')
# evaluate_model(y_test, mlp_pred, '多层感知器（MLP）')

# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.impute import SimpleImputer
#
# # 加载数据
# df = p)
#
# # 对 object 类型数据进行编码
# categorical_cols = df.select_dtypes(include=['object']).columns
# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#
# # 使用均值填充缺失值
# imputer = SimpleImputer(strategy='mean')
# df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#
# # 划分特征和目标变量
# X = df.drop(['id', 'Depression'], axis=1)
# y = df['Depression']
#
# # 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # 将数据转换为 PyTorch 张量
# X_train_tensor = torch.FloatTensor(X_train)
# y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
# X_test_tensor = torch.FloatTensor(X_test)
# y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)
#
#
# # 定义一个简单的全连接神经网络模型
# class DepressionModel(nn.Module):
#     def __init__(self, input_size):
#         super(DepressionModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         out = self.sigmoid(out)
#         return out
#
#
# # 初始化模型
# input_size = X_train_tensor.shape[1]
# model = DepressionModel(input_size)
#
# # 定义损失函数和优化器
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# num_epochs = 10
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 1 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 在测试集上进行预测
# with torch.no_grad():
#     model.eval()
#     y_pred_proba = model(X_test_tensor)
#     y_pred = (y_pred_proba > 0.5).float()
#
# # 评估模型
# accuracy = accuracy_score(y_test_tensor, y_pred)
# precision = precision_score(y_test_tensor, y_pred)
# recall = recall_score(y_test_tensor, y_pred)
# f1 = f1_score(y_test_tensor, y_pred)
#
# print('PyTorch 模型评估结果:')
# print(f'准确率: {accuracy:.4f}')
# print(f'精确率: {precision:.4f}')
# print(f'召回率: {recall:.4f}')
# print(f'F1分数: {f1:.4f}')


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from missingno import matrix  # 用于缺失值可视化（需安装：pip install missingno）
#
# # 假设已加载数据集（替换为你的数据路径）
# df = pd.read_csv("D:\机器学习笔记\Student Depression Dataset.csv")
#
# # 设置中文字体（避免中文乱码）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # ====================== 1. 目标变量分布（抑郁症比例） ======================
# plt.figure(figsize=(8, 4))
# sns.countplot(x='Depression', data=df)
# plt.title('目标变量（抑郁症）分布')
# plt.xlabel('是否患有抑郁症（0=否，1=是）')
# plt.ylabel('样本数量')
# plt.show()
#
# # ====================== 2. 数值特征分布（以年龄、压力值为例） ======================
# numeric_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Work/Study Hours']  # 替换为实际数值特征
# plt.figure(figsize=(15, 12))
# for i, col in enumerate(numeric_features):
#     plt.subplot(3, 2, i+1)
#     sns.histplot(df[col], kde=True, bins=20)
#     plt.title(f'{col} 分布')
# plt.tight_layout()
# plt.show()
#
# # ====================== 3. 分类特征分布（以性别、学位为例） ======================
# categorical_features = ['Gender', 'Degree', 'Sleep Duration', 'Dietary Habits']  # 替换为实际分类特征
# plt.figure(figsize=(15, 8))
# for i, col in enumerate(categorical_features):
#     plt.subplot(2, 2, i+1)
#     df[col].value_counts().plot(kind='bar')
#     plt.title(f'{col} 类别分布')
#     plt.xlabel('类别')
#     plt.ylabel('样本数量')
# plt.tight_layout()
# plt.show()
#
# # ====================== 4. 缺失值可视化（矩阵图） ======================
# plt.figure(figsize=(12, 6))
# matrix(df, color=(0.6, 0.2, 0.8), figsize=(12, 6))  # 缺失值矩阵图
# plt.title('缺失值分布矩阵')
# plt.show()


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv("D:\机器学习笔记\Student Depression Dataset.csv")

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf.predict(X_test)

# 评估模型
accuracy2 = accuracy_score(y_test, y_pred)
precision2 = precision_score(y_test, y_pred)
recall2 = recall_score(y_test, y_pred)
f12 = f1_score(y_test, y_pred)

print('随机森林模型评估结果:')
print(f'准确率: {accuracy2:.4f}')
print(f'精确率: {precision2:.4f}')
print(f'召回率: {recall2:.4f}')
print(f'F1分数: {f12:.4f}')

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv("D:\机器学习笔记\Student Depression Dataset.csv")

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建朴素贝叶斯模型
nb = GaussianNB()
nb.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = nb.predict(X_test)

# 评估模型
accuracy3 = accuracy_score(y_test, y_pred)
precision3 = precision_score(y_test, y_pred)
recall3 = recall_score(y_test, y_pred)
f13 = f1_score(y_test, y_pred)

print('朴素贝叶斯模型评估结果:')
print(f'准确率: {accuracy3:.4f}')
print(f'精确率: {precision3:.4f}')
print(f'召回率: {recall3:.4f}')
print(f'F1分数: {f13:.4f}')

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader

# 加载数据
df = pd.read_csv("D:\机器学习笔记\Student Depression Dataset.csv")

# 对 object 类型数据进行编码
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 使用均值填充缺失值
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 划分特征和目标变量
X = df.drop(['id', 'Depression'], axis=1)
y = df['Depression']

# 特征选择
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# 数据平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 划分训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义一个更复杂的全连接神经网络模型
class DepressionModel(nn.Module):
    def __init__(self, input_size):
        super(DepressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# 初始化模型
input_size = X_train_tensor.shape[1]
model = DepressionModel(input_size)

# 定义损失函数和优化器，添加 L2 正则化
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 早停策略参数
best_val_loss = float('inf')
patience = 10
early_stopping_counter = 0

# 训练模型
num_epochs = 50
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

# 在测试集上进行预测
with torch.no_grad():
    model.eval()
    y_pred_proba = model(X_test_tensor)
    y_pred = (y_pred_proba > 0.5).float()

# 评估模型
accuracy1 = accuracy_score(y_test_tensor, y_pred)
precision1 = precision_score(y_test_tensor, y_pred)
recall1 = recall_score(y_test_tensor, y_pred)
f11 = f1_score(y_test_tensor, y_pred)

print('优化后 PyTorch 模型评估结果:')
print(f'准确率: {accuracy1:.4f}')
print(f'精确率: {precision1:.4f}')
print(f'召回率: {recall1:.4f}')
print(f'F1分数: {f11:.4f}')

import matplotlib.pyplot as plt

# 定义模型名称和各指标数据
models = ['优化后 PyTorch', '随机森林', '朴素贝叶斯']
accuracy = [accuracy1,accuracy2,accuracy3]
precision = [precision1,precision2,precision3]
recall = [recall1,recall2,recall3]
f1 = [f11,f12,f13]

# 柱子宽度与位置参数
width = 0.2
x = range(len(models))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.title("中文标题")

# 绘制各指标柱状图
plt.bar([i - width for i in x], accuracy, width, label='准确率', color='skyblue')
plt.bar(x, precision, width, label='精确率', color='lightgreen')
plt.bar([i + width for i in x], recall, width, label='召回率', color='orange')
plt.bar([i + 2 * width for i in x], f1, width, label='F1分数', color='purple')

# 设置图表细节
plt.xticks(x, models, rotation=45)  # 旋转标签避免重叠
plt.xlabel('模型')
plt.ylabel('分数')
plt.title('三种模型评估指标对比')
plt.legend()
plt.tight_layout()  # 优化布局
plt.show()

