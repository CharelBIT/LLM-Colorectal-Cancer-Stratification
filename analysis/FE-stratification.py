from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
offline_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_offline_merge_encode.xlsx', engine='openpyxl')
online_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online_merge_encode.xlsx', engine='openpyxl') 
risk_stratification_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/副本直肠癌分险分层医师金标准2.xlsx', index_col=0, engine='openpyxl')
for index in risk_stratification_df.index:
    risk_str = risk_stratification_df.loc[index, '分险分层医师金标准（低/中/高分险）']
    if '低' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 0
    elif '中' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 1
    elif '高' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 2
data_df = pd.concat([offline_df, online_df], axis=0, ignore_index=True)

def marcro_ovsr_roc_curve(y, pred_proba):
    y = np.array(y)
    classes = np.unique(y)
    fprs, tprs, roc_aucs = {}, {}, {}
    for i, class_ in enumerate(classes):
        y_bin = np.where(y == class_, 1, 0)
        fpr, tpr, _ = metrics.roc_curve(y_bin, pred_proba[:, i])
        fprs[class_] = fpr
        tprs[class_] = tpr
        roc_auc = metrics.auc(fpr, tpr)
        roc_aucs[class_] = roc_auc
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for class_ in classes:
        mean_tpr += np.interp(mean_fpr, fprs[class_], tprs[class_])
    mean_tpr /= len(classes)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    return mean_fpr, mean_tpr, mean_auc
feature_cols = ['CRM受累情况', 'EMVI受累情况', '直肠系膜内淋巴结评估', '直肠系膜外淋巴结评估', '肿瘤浸润深度', '肿瘤的位置', '肿瘤累及长度']

plt.rcParams['figure.figsize'] = (10.0, 10.0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_names = data_df['model_name'].unique().tolist()
num_rows = num_cols = int(math.ceil(math.sqrt(len(model_names))))
_, axes = plt.subplots(num_rows, num_cols)
mean_fpr = np.linspace(0, 1, 100)
for i, model_name in enumerate(model_names):
    X = data_df[data_df['model_name'] == model_name][feature_cols]
    y = risk_stratification_df.loc[X.index, 'Label']
    tprs, aucs = [], []
    for j, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = LogisticRegression(class_weight='balanced', penalty='l2', max_iter=300, random_state=1234, multi_class='ovr')
        model.fit(X.iloc[train_index], y.iloc[train_index])
        test_pred_proba = model.predict_proba(X.iloc[test_index])
        fpr, tpr, roc_auc = marcro_ovsr_roc_curve(y.iloc[test_index], test_pred_proba)
        
        cr = metrics.classification_report(y.iloc[test_index], model.predict(X.iloc[test_index]), output_dict=True, zero_division=0)
        axes[i // num_cols, i % num_cols].plot(
            fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold {} (AUC = {:.3f})'.format(j, roc_auc)
        )
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    axes[i // num_cols, i % num_cols].fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    axes[i // num_cols, i % num_cols].plot((0, 1), (0, 1), linestyle='--', color='grey', lw=2)
    axes[i // num_cols, i % num_cols].plot(
        mean_fpr, mean_tpr, color='blue', lw=2,
        label='Mean ROC (AUC = {:.2f})'.format(np.mean(aucs))
    )
    axes[i // num_cols, i % num_cols].set(xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    # axes.ylim(-0.05, 1.05)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axes[i // num_cols, i % num_cols].legend(loc="lower right", fontsize=15)
    axes[i // num_cols, i % num_cols].set_title(f"{model_name}")
    axes[i // num_cols, i % num_cols].set_xlabel('1-Specificity')
    axes[i // num_cols, i % num_cols].set_ylabel('Sensitivity')
plt.tight_layout()
plt.savefig('/gruntdata/workDir/dataset/jed_error_report/DD/各模型风险分层ROC曲线.svg', dpi=300)
