from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import os
import matplotlib.pyplot as plt
save_path = '/gruntdata/workDir/dataset/jed_error_report/DD/paper_materials_2'
os.makedirs(save_path, exist_ok=True)
for model_name in auto_stratification_df['ModelPaired'].unique().tolist():
    df = auto_stratification_df[auto_stratification_df['ModelPaired'] == model_name]
    failed_num = pd.isna(df['风险分级']).sum() / df.shape[0]
    print(f"模型 {model_name} 自动分层失败比例: {failed_num:.2%}")
    df.dropna(subset=['风险分级'], inplace=True, axis=0)
    common_patient_ids = list(set(df.index) & set(risk_stratification_df.index))
    df = df.loc[common_patient_ids]
    y = risk_stratification_df.loc[common_patient_ids, 'Label']
    cm = metrics.confusion_matrix(y, df['风险分级'].astype(int))
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap='Blues')
    display.ax_.set_title(f'{model_name}')
    display.ax_.set_xlabel('Prediction')
    display.ax_.set_ylabel('Ground Truth')
    display.figure_.set_size_inches(8, 6)
    plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.svg'), dpi=300)
    plt.clf()

    