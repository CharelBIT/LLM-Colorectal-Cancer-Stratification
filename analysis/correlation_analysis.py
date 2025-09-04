import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Generate some sample data
def correlation_analysis(x, y, save_file):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Create the plot
    fig, axScatter = plt.subplots(figsize=(10, 8))

    # Scatter plot with regression line
    sns.regplot(x=x, y=y, ax=axScatter, scatter_kws={'s': 50}, line_kws={"color": "blue"})

    # Create the histograms on the margins
    divider = make_axes_locatable(axScatter)
    axHistTop = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
    axHistRight = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

    # Make some labels invisible
    axHistTop.xaxis.set_tick_params(labelbottom=False)
    axHistRight.yaxis.set_tick_params(labelleft=False)

    # Plot histograms
    axHistTop.hist(x, bins=20, color='green', edgecolor='black')
    axHistRight.hist(y, bins=20, orientation='horizontal', color='orange', edgecolor='black')

    # Annotate the plot with statistics
    axScatter.text(0.5, 0.95, f'$t_{{Student}}(47) = {stats.t.ppf(0.975, df=len(x)-1):.2f}, p = {p_value:.2e}$\n'
                            f'$r_{{Pearson}} = {r_value:.2f}, CI_{{95\%}} = [{r_value - 1.96*std_err:.2f}, {r_value + 1.96*std_err:.2f}]$\n'
                            f'$n_{{pairs}} = {len(x)}$', 
                transform=axScatter.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

    plt.xlabel('AI Volume')
    plt.ylabel('Y Value')
    plt.savefig(save_file)

for model_name in model_names:
    pred_df = data_df[data_df['model_name'] == model_name]
    
    common_patient_ids = pred_df.index.intersection(gt_df.index)
    pred_df = pred_df.loc[common_patient_ids]
    gt_df = gt_df.loc[common_patient_ids]
    for feature in ['肿瘤累及长度',]:
        correlation_analysis(pred_df[feature], gt_df[feature], f'/gruntdata/workDir/dataset/jed_error_report/DD/{model_name}_{feature}_correlation_analysis.svg')