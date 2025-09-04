

import pandas as pd
from functools import reduce

# rename_dict = {
#     'CRM受累情况': ['CRM受累'],
#     'EMVI受累情况': ['EMVI受累'],
#     '直肠系膜内淋巴结评估': ['直肠系膜内淋巴结'],
#     '直肠系膜外淋巴结评估': ['直肠系膜外淋巴结']
# }

# data_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online.xlsx')
# for index in data_df.index:
#     for key in rename_dict.keys():
#         if not pd.isna(data_df.loc[index, key]):
#             continue
#         for sub_key in rename_dict[key]:
#             if not pd.isna(data_df.loc[index, sub_key]):
#                 data_df.loc[index, key] = data_df.loc[index, sub_key]
#                 break
# delete_cols = reduce(lambda x, y: x + y, rename_dict.values())
# data_df.drop(delete_cols, axis=1, inplace=True)
# data_df.to_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online_merge.xlsx')

# rename_dict = {
#     'CRM受累情况': ['CRM受累'],
#     'EMVI受累情况': ['EMVI受累'],
#     '直肠系膜内淋巴结评估': ['直肠系膜内淋巴结'],
#     '直肠系膜外淋巴结评估': ['直肠系膜外淋巴结']
# }
# data_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/feature_gold_standard.xlsx')
# for index in data_df.index:
#     for key in rename_dict.keys():
#         if not pd.isna(data_df.loc[index, key]):
#             continue
#         for sub_key in rename_dict[key]:
#             if not pd.isna(data_df.loc[index, sub_key]):
#                 data_df.loc[index, key] = data_df.loc[index, sub_key]
#                 break
# delete_cols = reduce(lambda x, y: x + y, rename_dict.values())
# data_df.drop(delete_cols, axis=1, inplace=True)
# data_df.to_excel('/gruntdata/workDir/dataset/jed_error_report/DD/feature_gold_standard_merge.xlsx', index=False)

# import pandas as pd
# offline_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_offline_merge.xlsx')
# online_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online_merge.xlsx')
# total_df = pd.concat([offline_df, online_df], axis=0, ignore_index=True)
# total_df.to_excel('/gruntdata/workDir/dataset/jed_error_report/DD/total_data.xlsx')


rename_dict = {
    'CRM受累情况': ['CRM受累'],
    'EMVI受累情况': ['EMVI受累'],
    '直肠系膜内淋巴结评估': ['直肠系膜内淋巴结'],
    '直肠系膜外淋巴结评估': ['直肠系膜外淋巴结']
}
data_df = pd.read_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online.xlsx')
for index in data_df.index:
    for key in rename_dict.keys():
        if not pd.isna(data_df.loc[index, key]):
            continue
        for sub_key in rename_dict[key]:
            if not pd.isna(data_df.loc[index, sub_key]):
                data_df.loc[index, key] = data_df.loc[index, sub_key]
                break
delete_cols = reduce(lambda x, y: x + y, rename_dict.values())
data_df.drop(delete_cols, axis=1, inplace=True)
data_df.to_excel('/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online_merge.xlsx', index=False)