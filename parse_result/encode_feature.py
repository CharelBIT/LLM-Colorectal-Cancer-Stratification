import pandas as pd
data_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online_merge.xlsx')

jinrun = {
    'T1': 1,
    'T2': 2,
    'T3': 3,
    'T4': 4,
}

# 上段，中段，下段，中上段，中下段，直肠全段，直肠乙状结肠交界区；
rectum_loc = {
    '上段': 1,
    '中段': 2,
    '下段': 3,
    '中上段': 4,
    '中下段': 5,
    '直肠全段': 6,
    '交界区': 7
}
loc = {

}

for index in data_df.index:
    if pd.isna(data_df.loc[index, '直肠系膜内淋巴结评估']) or '无' in data_df.loc[index, '直肠系膜内淋巴结评估']:
        data_df.loc[index, '直肠系膜内淋巴结评估'] = 0
    else:
        data_df.loc[index, '直肠系膜内淋巴结评估'] = 1

    if pd.isna(data_df.loc[index, '直肠系膜外淋巴结评估']) or '无' in data_df.loc[index, '直肠系膜外淋巴结评估']:
        data_df.loc[index, '直肠系膜外淋巴结评估'] = 0
    else:
        data_df.loc[index, '直肠系膜外淋巴结评估'] = 1

    if pd.isna(data_df.loc[index, '肿瘤浸润深度']) or data_df.loc[index, '肿瘤浸润深度'] not in jinrun.keys():
        data_df.loc[index, '肿瘤浸润深度'] = 0
    else:
        data_df.loc[index, '肿瘤浸润深度'] = jinrun[data_df.loc[index, '肿瘤浸润深度']]

    if pd.isna(data_df.loc[index, '肿瘤的位置']) or data_df.loc[index, '肿瘤的位置'] not in rectum_loc.keys():
        data_df.loc[index, '肿瘤的位置'] = 0
    else:
        for key in rectum_loc.keys():
            if key in data_df.loc[index, '肿瘤的位置']:
                data_df.loc[index, '肿瘤的位置'] = rectum_loc[key]
                break
    
    if pd.isna(data_df.loc[index, '肿瘤累及长度']):
        data_df.loc[index, '肿瘤累及长度'] = 0
    else:
        if 'mm' in data_df.loc[index, '肿瘤累及长度']:
            mm = data_df.loc[index, '肿瘤累及长度'][:data_df.loc[index, '肿瘤累及长度'].find('mm')]
            data_df.loc[index, '肿瘤累及长度'] = float(mm)
        elif 'cm' in data_df.loc[index, '肿瘤累及长度']:
            cm = data_df.loc[index, '肿瘤累及长度'][:data_df.loc[index, '肿瘤累及长度'].find('cm')]
            data_df.loc[index, '肿瘤累及长度'] = float(cm) * 10

    if pd.isna(data_df.loc[index, 'CRM受累情况']) or '无' in data_df.loc[index, 'CRM受累情况']:
        data_df.loc[index, 'CRM受累情况'] = 0
    else:
        data_df.loc[index, 'CRM受累情况'] = 1

    if pd.isna(data_df.loc[index, 'EMVI受累情况']) or '无' in data_df.loc[index, 'EMVI受累情况']:
        data_df.loc[index, 'EMVI受累情况'] = 0
    else:
        data_df.loc[index, 'EMVI受累情况'] = 1

data_df.to_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online_merge_encode.xlsx', index=False)
# import pandas as pd
# data_df = pd.read_excel('/datadisk/zjh/cjwork/dataset/jde_llm_project/DD/parse_input_offline_merge_encode.xlsx', engine='openpyxl')
# data_df = pd.concat([data_df, pd.read_excel('/datadisk/zjh/cjwork/dataset/jde_llm_project/DD/parse_input_online_merge_encode.xlsx', engine='openpyxl')], axis=0, ignore_index=True)
# data_df.to_excel('/datadisk/zjh/cjwork/dataset/jde_llm_project/DD/parse_input_merge_encode.xlsx', index=False)
