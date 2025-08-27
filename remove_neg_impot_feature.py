import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

seed = 45
fold = 8
ker = "rbf"

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=fold, scoring="accuracy", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=seed
    )

    # 計算平均與標準差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 畫圖
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std,
                     alpha=0.2, color="blue")

    plt.plot(train_sizes, test_mean, label="Cross-validation score", color="green")
    plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std,
                     alpha=0.2, color="green")

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="best")
    plt.show()

# ==================== 1. 載入和準備資料 ====================


df = pd.read_csv("data_standardized_complete_v2.csv")  #'age', 'HbA1c_LOCF', 'GNB3G'('HAMD_V4_LOCF', 'oxytocin_LOCF','pers_err_LOCF', 'SSS.iib_s', 'whoqol28_ov')

# 特徵 (X) 和 目標變數 (y1, y2)
X = df.drop(['age', 'HbA1c_LOCF', 'GNB3G', 'res_no_1', 'remissionv41_LOCF', 'responderv41_LOCF'], axis=1)
y = df[['remissionv41_LOCF', 'responderv41_LOCF']]

# ==================== 2. 資料分割 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=seed, stratify=y['remissionv41_LOCF']
)

# ==================== 3. 建立和訓練 SVM 模型 (MultiOutput) ====================
base_model = make_pipeline(MinMaxScaler(), SVC(kernel=ker, random_state=seed, probability=True))
#'estimator__svc__C': 105, 'estimator__svc__gamma': 0.00952

param_grid = {
    'estimator__svc__C': [180],
    'estimator__svc__gamma': ['scale', 0.008]
}

multi_clf = MultiOutputClassifier(base_model)

grid = GridSearchCV(multi_clf, param_grid, cv=fold, scoring='accuracy', verbose=1, return_train_score=True)
grid.fit(X_train, y_train)

print("最佳參數:", grid.best_params_)
print("交叉驗證最佳分數:", grid.best_score_)
best_model = grid.best_estimator_

best_idx = grid.best_index_

mean_score = grid.cv_results_['mean_test_score'][best_idx]
std_score = grid.cv_results_['std_test_score'][best_idx]

clf_rem = best_model.estimators_[0]
clf_res = best_model.estimators_[1]

y_rem = y['remissionv41_LOCF'].values
y_res = y['responderv41_LOCF'].values

cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)

# 改成 accuracy
scores_rem = cross_val_score(clf_rem, X, y_rem, cv=cv, scoring="accuracy", n_jobs=-1)
scores_res = cross_val_score(clf_res, X, y_res, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"Remission Accuracy:  {scores_rem.mean():.4f} ± {scores_rem.std(ddof=1):.4f} (n={len(scores_rem)})")
print(f"Responder Accuracy: {scores_res.mean():.4f} ± {scores_res.std(ddof=1):.4f} (n={len(scores_res)})")

print(f"Accuracy: {mean_score:.4f} ± {std_score:.4f}")
# ==================== 4. 使用最佳模型進行預測 ====================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

plot_learning_curve(best_model.estimators_[0], X, y['remissionv41_LOCF'],
                    "Learning Curve (SVM) - remissionv41_LOCF")

# 對 responder 畫學習曲線
plot_learning_curve(best_model.estimators_[1], X, y['responderv41_LOCF'],
                    "Learning Curve (SVM) - responderv41_LOCF")

clf_rem = best_model.estimators_[0]
clf_res = best_model.estimators_[1]

# y 需是 DataFrame，內含 0/1
y_rem = y['remissionv41_LOCF'].values
y_res = y['responderv41_LOCF'].values

# 用 decision_function（不需要 probability=True）
auc_scorer = make_scorer(roc_auc_score, needs_threshold=True)

cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)

scores_rem = cross_val_score(clf_rem, X, y_rem, cv=cv, scoring="roc_auc", n_jobs=-1)
scores_res = cross_val_score(clf_res, X, y_res, cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"Remission AUC: {scores_rem.mean():.4f} ± {scores_rem.std(ddof=1):.4f}")
print(f"Responder AUC: {scores_res.mean():.4f} ± {scores_res.std(ddof=1):.4f}")
# ==================== 5. Confusion Matrix 視覺化 ====================

for i, col in enumerate(y.columns):
    cm = confusion_matrix(y_test[col], y_pred[:, i])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["False", "True"], yticklabels=["False", "True"])
    
    acc = accuracy_score(y_test[col], y_pred[:, i])
    print(f"{col} - Accuracy: {acc:.4f}")

    '''
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {col}")
    plt.show()
    '''