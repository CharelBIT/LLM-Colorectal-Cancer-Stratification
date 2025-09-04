import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
def weighted_cohen_kappa_with_pvalue(y_true, y_pred, weights='linear'):
    """
    Compute weighted Cohen's kappa and its p-value using an approximate variance.
    
    Parameters:
    - y_true: array-like of true labels
    - y_pred: array-like of predicted labels
    - weights: 'linear' or 'quadratic'
    
    Returns:
    - kappa: weighted kappa coefficient
    - se: standard error
    - z: z-score
    - p: two-sided p-value
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    N = cm.sum()
    k = len(labels)
    
    # Create weight matrix
    idx = np.arange(k)
    if weights == 'linear':
        w = 1 - np.abs(np.subtract.outer(idx, idx)) / (k - 1)
    elif weights == 'quadratic':
        w = 1 - (np.subtract.outer(idx, idx) / (k - 1))**2
    else:
        raise ValueError("weights must be 'linear' or 'quadratic'")
    
    # Observed agreement with weights
    Po_w = np.sum(w * cm) / N
    
    # Marginal probabilities
    p_true = cm.sum(axis=1) / N
    p_pred = cm.sum(axis=0) / N
    
    # Expected agreement with weights
    Pe_w = np.sum(w * np.outer(p_true, p_pred))
    
    # Weighted kappa
    kappa = (Po_w - Pe_w) / (1 - Pe_w)
    
    # Variance approximation
    var_k = (Po_w * (1 - Po_w)) / ((1 - Pe_w)**2 * N)
    se = np.sqrt(var_k)
    
    # z-score and p-value
    z = kappa / se
    p = 2 * norm.sf(abs(z))
    
    return kappa, se, z, p
auto_stratification_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_merge_encode_flatten.xlsx', index_col=0, engine='openpyxl')
auto_stratification_df.set_index('Unnamed: 0.1', inplace=True)
auto_stratification_df['model_name'] = auto_stratification_df['model_name'].apply(lambda x: x.replace('deepseek-r1:latest', 'DeepSeek-R1-8B').replace('qwen3:8b', 'Qwen3-8B').replace('deepseek-reasoner', 'DeepSeek-Reasoner'))
auto_stratification_df = auto_stratification_df[['风险分级', 'model_name', 'stratification_model_name']]
auto_stratification_df['ModelPaired'] = list(map(lambda x, y: f"{x.split('_')[-1]}_{y.split('_')[-1]}", auto_stratification_df['model_name'], auto_stratification_df['stratification_model_name']))
auto_stratification_df['ModelPaired'] = auto_stratification_df['ModelPaired'].apply(lambda x: x.replace(':', '-'))
print(auto_stratification_df['ModelPaired'].unique())


risk_stratification_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/副本直肠癌分险分层医师金标准2.xlsx', index_col=0, engine='openpyxl')
for index in risk_stratification_df.index:
    risk_str = risk_stratification_df.loc[index, '分险分层医师金标准（低/中/高分险）']
    if '低' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 0
    elif '中' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 1
    elif '高' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 2

kappa_metrics = pd.DataFrame()
for model_1 in auto_stratification_df['ModelPaired'].unique().tolist():
    for model_2 in auto_stratification_df['ModelPaired'].unique().tolist():
        df_1 = auto_stratification_df[auto_stratification_df['ModelPaired'] == model_1]
        df_2 = auto_stratification_df[auto_stratification_df['ModelPaired'] == model_2]
        df_1.dropna(subset=['风险分级'], inplace=True, axis=0)
        df_2.dropna(subset=['风险分级'], inplace=True, axis=0)
        common_patient_ids = list(set(df_1.index) & set(df_2.index))
        df_1 = df_1.loc[common_patient_ids]
        df_2 = df_2.loc[common_patient_ids]
        x = df_1['风险分级'].astype(int)
        y = df_2['风险分级'].astype(int)
        kappa, se, z, p = weighted_cohen_kappa_with_pvalue(x, y, weights='linear')
        kappa_metrics.loc[model_1, model_2] = kappa
for model_name in auto_stratification_df['ModelPaired'].unique().tolist():
    df = auto_stratification_df[auto_stratification_df['ModelPaired'] == model_name]
    failed_num = pd.isna(df['风险分级']).sum() / df.shape[0]
    print(f"模型 {model_name} 自动分层失败比例: {failed_num:.2%}")
    df.dropna(subset=['风险分级'], inplace=True, axis=0)
    common_patient_ids = list(set(df.index) & set(risk_stratification_df.index))
    df = df.loc[common_patient_ids]
    y = risk_stratification_df.loc[common_patient_ids, 'Label']
    y_pred = df['风险分级'].astype(int)
    kappa, se, z, p = weighted_cohen_kappa_with_pvalue(y, y_pred, weights='linear')
    kappa_metrics.loc['Radiologist', model_name] = kappa
    kappa_metrics.loc[model_name, 'Radiologist'] = kappa
kappa_metrics.loc['Radiologist', 'Radiologist'] = 1.0  # Radiologist vs Radiologist is always perfect agreement
fig, ax = plt.subplots(figsize=(11, 11))


kappa_array = kappa_metrics.values
n = len(kappa_metrics.columns)
cmap = plt.cm.get_cmap().copy()
cmap.set_bad(alpha=0)
mask = np.triu(np.ones((n, n), dtype=bool), k=1)
A_masked = np.ma.array(kappa_array, mask=mask)
im = ax.imshow(A_masked, aspect="equal", vmin=0, vmax=1, cmap=cmap)
# plt.colorbar(label="Kappa value")
plt.xticks(range(n), kappa_metrics.columns, rotation=45, ha="right")
plt.yticks(range(n), kappa_metrics.columns)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)   # 只去掉刻度线，保留刻度文字（想全关可用 ax.axis('off')）

# 色条（可选）
cbar = fig.colorbar(im, ax=ax)
cbar.outline.set_visible(False)   # 去掉色条外框
# Annotate only the lower triangle (including diagonal)
rows, cols = np.tril_indices(n, k=0)   # 若想连对角线也不标，改成 k=-1
for i, j in zip(rows, cols):
    val = kappa_array[i, j]
    
    if not np.isnan(val):
        plt.text(j, i, f"{val:.2f}", ha="center", va="center")

plt.title("Kappa Matrix")
plt.tight_layout()
# plt.savefig("/mnt/data/kappa_lower_heatmap.png", dpi=200, bbox_inches="tight")

plt.savefig('kappa_lower_heatmap.svg', dpi=300, bbox_inches="tight")