import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
df = pd.read_csv(r"D:\统计数模大赛\copy.csv")

# 对 object 类型数据进行编码
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 处理缺失值，使用 KNN 算法填充
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 划分特征和目标变量，假设目标变量为 'Depression'
X = df.drop('Depression', axis=1)
y = df['Depression']

# 数据标准化：使用 Z - score 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 特征工程：主成分分析（PCA）降维
pca = PCA(n_components=0.95)  # 保留 95% 的方差
X_pca = pca.fit_transform(X_scaled)

# 特征选择：基于相关性的特征选择
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_pca, y)

# 数据平衡处理
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 训练支持向量机（SVM）模型
svm = SVC(random_state=42)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring='f1')
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
svm_pred = best_svm.predict(X_test)
svm_f1 = f1_score(y_test, svm_pred)

# 训练 Adaboost 模型
ada = AdaBoostClassifier(random_state=42)
param_grid_ada = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 1]
}
grid_search_ada = GridSearchCV(ada, param_grid_ada, cv=3, scoring='f1')
grid_search_ada.fit(X_train, y_train)
best_ada = grid_search_ada.best_estimator_
ada_pred = best_ada.predict(X_test)
ada_f1 = f1_score(y_test, ada_pred)

# 训练神经网络模型
mlp = MLPClassifier(random_state=42, max_iter=300)  # 例如设为300
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
   'max_iter': [300]  # 如果在GridSearchCV中调优，也需包含此参数
}
grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=3, scoring='f1')
grid_search_mlp.fit(X_train, y_train)
best_mlp = grid_search_mlp.best_estimator_
mlp_pred = best_mlp.predict(X_test)
mlp_f1 = f1_score(y_test, mlp_pred)

# 模型集成：加权平均法
weights = [svm_f1, ada_f1, mlp_f1]
weights = np.array(weights) / np.sum(weights)
ensemble_pred_proba = (weights[0] * best_svm.decision_function(X_test) +
                       weights[1] * best_ada.predict_proba(X_test)[:, 1] +
                       weights[2] * best_mlp.predict_proba(X_test)[:, 1])
ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_precision = precision_score(y_test, ensemble_pred)
ensemble_recall = recall_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred)

print(f'支持向量机模型 F1 分数：{svm_f1:.4f}')
print(f'Adaboost 模型 F1 分数：{ada_f1:.4f}')
print(f'神经网络模型 F1 分数：{mlp_f1:.4f}')

print(f'集成模型准确率：{ensemble_accuracy:.4f}')
print(f'集成模型精确率：{ensemble_precision:.4f}')
print(f'集成模型召回率：{ensemble_recall:.4f}')
print(f'集成模型 F1 分数：{ensemble_f1:.4f}')

# 绘制 PCA 方差贡献率图
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.show()