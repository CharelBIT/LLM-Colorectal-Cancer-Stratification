

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt
import math
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
font = FontProperties(fname='/gruntdata/workDir/dataset/xnyy_mri_baby/simhei.ttf', size=15)
plt.rcParams['figure.figsize'] = (10.0, 10.0)
auto_stratification_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_merge_encode_flatten.xlsx', index_col=0, engine='openpyxl')
auto_stratification_df.set_index('Unnamed: 0.1', inplace=True)
auto_stratification_df['model_name'] = auto_stratification_df['model_name'].apply(lambda x: x.replace('deepseek-r1:latest', 'DeepSeek-R1-8B').replace('qwen3:8b', 'Qwen3-8B').replace('deepseek-reasoner', 'DeepSeek-Reasoner'))
auto_stratification_df = auto_stratification_df[['风险分级', 'model_name', 'stratification_model_name']]
auto_stratification_df['ModelPaired'] = list(map(lambda x, y: f"{x.split('_')[-1]}_{y.split('_')[-1]}", auto_stratification_df['model_name'], auto_stratification_df['stratification_model_name']))
auto_stratification_df['ModelPaired'] = auto_stratification_df['ModelPaired'].apply(lambda x: x.replace(':', '-'))
print(auto_stratification_df['ModelPaired'].unique())

offline_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_offline_merge_encode.xlsx', engine='openpyxl')
online_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online_merge_encode.xlsx', engine='openpyxl') 

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
from collections import defaultdict
boostrap_metrics = {}
for model_name in data_df['model_name'].unique().tolist():
    boostrap_metrics[model_name] = defaultdict(list)
feature_cols = ['CRM受累情况', 'EMVI受累情况', '直肠系膜内淋巴结评估', '直肠系膜外淋巴结评估', '肿瘤浸润深度', '肿瘤的位置', '肿瘤累及长度']
risk_stratification_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/副本直肠癌分险分层医师金标准2.xlsx', index_col=0, engine='openpyxl')
for index in risk_stratification_df.index:
    risk_str = risk_stratification_df.loc[index, '分险分层医师金标准（低/中/高分险）']
    if '低' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 0
    elif '中' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 1
    elif '高' in risk_str:
        risk_stratification_df.loc[index, 'Label'] = 2
for seed in range(200):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model_names = data_df['model_name'].unique().tolist()
    # num_rows = num_cols = int(math.ceil(math.sqrt(len(model_names))))
    # _, axes = plt.subplots(num_rows, num_cols)
    # mean_fpr = np.linspace(0, 1, 100)
    for i, model_name in enumerate(model_names):
        X = data_df[data_df['model_name'] == model_name][feature_cols]
        y = risk_stratification_df.loc[X.index, 'Label']
        # tprs, aucs = [], []
        for j, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model = LogisticRegression(class_weight='balanced', penalty='l2', max_iter=300, random_state=1234, multi_class='ovr')
            model.fit(X.iloc[train_index], y.iloc[train_index])
            test_pred_proba = model.predict_proba(X.iloc[test_index])
            fpr, tpr, roc_auc = marcro_ovsr_roc_curve(y.iloc[test_index], test_pred_proba)
            boostrap_metrics[model_name]['AUC'].append(roc_auc)
            cr = metrics.classification_report(y.iloc[test_index], model.predict(X.iloc[test_index]), output_dict=True, zero_division=0)
            boostrap_metrics[model_name]['Precision'].append(cr['macro avg']['precision'])
            boostrap_metrics[model_name]['Recall'].append(cr['macro avg']['recall'])
            boostrap_metrics[model_name]['F1-score'].append(cr['macro avg']['f1-score'])
            boostrap_metrics[model_name]['Accuracy'].append(cr['accuracy'])

common_patient_ids = set(auto_stratification_df.index)
cn2label = {
    '低': 0,
    '中': 1,
    '高': 2,
}
en2label = {
    'low': 0,
    'intermediate': 1,
    'high': 2,
}
junior_struct_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/副本低年资直肠癌结构化报告的风险等级评估.xlsx', engine='openpyxl', index_col=0)
junior_struct_df = junior_struct_df.loc[common_patient_ids]
junior_struct_df['风险分级'] = junior_struct_df['（分险分层）医师2'].apply(lambda x: en2label[x])
junior_struct_df['ModelPaired'] = 'Junior-Structured_Report'

junior_org_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/副本低年资直肠癌原始报告的风险等级评估..xlsx', engine='openpyxl', index_col=0)
junior_org_df = junior_org_df.loc[common_patient_ids]
junior_org_df['风险分级'] = junior_org_df['（分险分层）医师2'].apply(lambda x: en2label[x])
junior_org_df['ModelPaired'] = 'Junior-Original_Report'

senior_struct_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/副本高年资医师直肠癌结构化报告的风险等级评估.xlsx', engine='openpyxl', index_col=0)
senior_struct_df = senior_struct_df.loc[common_patient_ids]
senior_struct_df['风险分级'] = senior_struct_df['医师1风险等级评估'].apply(lambda x: cn2label[x])
senior_struct_df['ModelPaired'] = 'Senior-Structured_Report'

senior_org_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/副本高年资医师直肠癌原始报告的风险等级评估..xlsx', engine='openpyxl', index_col=0)
senior_org_df = senior_org_df.loc[common_patient_ids]
senior_org_df['风险分级'] = senior_org_df['医师1风险等级评估'].apply(lambda x: cn2label[x])
senior_org_df['ModelPaired'] = 'Senior-Original_Report'

doctor_df = pd.concat([junior_struct_df, junior_org_df, senior_struct_df, senior_org_df], axis=0)
doctor_df = doctor_df[['风险分级', 'ModelPaired']]

auto_stratification_df = pd.concat([auto_stratification_df, doctor_df], axis=0)
for model_name in auto_stratification_df['ModelPaired'].unique().tolist():
    df = auto_stratification_df[auto_stratification_df['ModelPaired'] == model_name]
    failed_num = pd.isna(df['风险分级']).sum() / df.shape[0]
    print(f"模型 {model_name} 自动分层失败比例: {failed_num:.2%}")
    df.dropna(subset=['风险分级'], inplace=True, axis=0)
    common_patient_ids = list(set(df.index) & set(risk_stratification_df.index))
    df = df.loc[common_patient_ids]
    y = risk_stratification_df.loc[common_patient_ids, 'Label']
    for seed in range(200):
        spliiter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(spliiter.split(df, y)):
            y_test = y.iloc[test_index]
            y_pred = df.iloc[test_index]['风险分级'].astype(int)
            cr = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            if model_name not in boostrap_metrics:
                boostrap_metrics[model_name] = defaultdict(list)
            boostrap_metrics[model_name]['Precision'].append(cr['macro avg']['precision'])
            boostrap_metrics[model_name]['Recall'].append(cr['macro avg']['recall'])
            boostrap_metrics[model_name]['F1-score'].append(cr['macro avg']['f1-score'])
            boostrap_metrics[model_name]['Accuracy'].append(cr['accuracy'])

            model = LogisticRegression(class_weight='balanced', penalty='l2', max_iter=300, random_state=1234, multi_class='ovr')
            model.fit(df.iloc[train_index][['风险分级']], y.iloc[train_index])
            test_pred_proba = model.predict_proba(df.iloc[test_index][['风险分级']])
            fpr, tpr, roc_auc = marcro_ovsr_roc_curve(y.iloc[test_index], test_pred_proba)
            boostrap_metrics[model_name]['AUC'].append(roc_auc)
from numpy import mean


mertircs_df = pd.DataFrame()

def mean_95_ci(data):
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    ci = 1.96 * (std / np.sqrt(n))  # 95% CI
    return "{:3f}({:3f}, {:3f})".format(mean, mean - ci, mean + ci)

import numpy as np

def compare_paired_bootstrap(seA_boot, seB_boot, ci=0.95):
    seA_boot = np.asarray(seA_boot)
    seB_boot = np.asarray(seB_boot)
    assert len(seA_boot) == len(seB_boot), "成对样本需要等长"

    delta = seA_boot - seB_boot
    alpha = 1 - ci
    lo, hi = np.quantile(delta, [alpha/2, 1-alpha/2])
    # 近似双侧 p 值（与百分位CI一致的经验法）
    B = len(delta)
    k = np.sum(delta <= 0)
    p = 2 * min((k+1)/(B+1), (B-k+1)/(B+1))
    return "{:3f}({:3f}, {:3f})[{:.3f}]".format(float(delta.mean()), float(lo), float(hi), p)
    # return dict(delta_mean=float(delta.mean()), ci=(float(lo), float(hi)), p_value=float(min(1.0, p)))


for model_name, metrics in boostrap_metrics.items():
    mertircs_df.loc[model_name, 'Precision'] = mean_95_ci(metrics['Precision'])
    mertircs_df.loc[model_name, 'Recall'] =  mean_95_ci(metrics['Recall'])
    mertircs_df.loc[model_name, 'F1-score'] = mean_95_ci(metrics['F1-score'])
    mertircs_df.loc[model_name, 'Accuracy'] = mean_95_ci(metrics['Accuracy'])
    mertircs_df.loc[model_name, 'AUC'] = mean_95_ci(metrics['AUC'])
    if model_name != 'FE-Gold-Standard':
        mertircs_df.loc[model_name, 'Precision vs FE-Gold-Standard'] = compare_paired_bootstrap(metrics['Precision'], boostrap_metrics['FE-Gold-Standard']['Precision'])
        mertircs_df.loc[model_name, 'Recall vs FE-Gold-Standard'] = compare_paired_bootstrap(metrics['Recall'], boostrap_metrics['FE-Gold-Standard']['Recall'])
        mertircs_df.loc[model_name, 'F1-score vs FE-Gold-Standard'] = compare_paired_bootstrap(metrics['F1-score'], boostrap_metrics['FE-Gold-Standard']['F1-score'])
        mertircs_df.loc[model_name, 'Accuracy vs FE-Gold-Standard'] = compare_paired_bootstrap(metrics['Accuracy'], boostrap_metrics['FE-Gold-Standard']['Accuracy'])
        mertircs_df.loc[model_name, 'AUC vs FE-Gold-Standard'] = compare_paired_bootstrap(metrics['AUC'], boostrap_metrics['FE-Gold-Standard']['AUC'])
mertircs_df