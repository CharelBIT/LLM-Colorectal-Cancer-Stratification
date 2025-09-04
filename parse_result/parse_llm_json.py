import pandas as pd
import json
def parse_offline(data_str):
    data_str_strip = data_str[data_str.find('</think>\n') + 9:]
    data_json_str = data_str_strip[data_str_strip.rfind('{'): data_str_strip.rfind('}') + 1]
    data_json_str = data_json_str.replace("None", "null").replace("'", '"').replace('\n\n', '\n').replace('：', ":")
    try:
        data_json = json.loads(data_json_str)
    except:
        return {}
    return data_json

def parse_online(data_str):
    data_str_strip = data_str
    data_json_str = data_str_strip[data_str_strip.rfind('{'): data_str_strip.rfind('}') + 1]
    data_json_str = data_json_str.replace("None", "null").replace("'", '"')
    data_json = json.loads(data_json_str)
    return data_json


def parse_gt(data_str):
    data_str_strip = data_str
    data_json_str = data_str_strip[data_str_strip.rfind('{'): data_str_strip.rfind('}') + 1]
    data_json_str = data_json_str.replace("None", "null").replace("'", '"')
    data_json = json.loads(data_json_str)
    return data_json


def extract_json_from_str(data_file, model_names, save_file):
    data_df = pd.read_excel(data_file)
    save_data_df = pd.DataFrame()
    for index in data_df.index:
        print(index, data_df.loc[index, model_names[0]])
        for model_name in model_names:
            if 'reasoner' in model_name:
                json_info = parse_online(data_df.loc[index, model_name])
            else:
                json_info = parse_gt(data_df.loc[index, model_name])
            json_info.update(data_df.loc[index].to_dict())
            json_info['model_name'] = model_name
            save_data_df = save_data_df.append(json_info, ignore_index=True)
    save_data_df.to_excel(save_file)

# data_file = '/gruntdata/workDir/dataset/jed_error_report/DD/副本input1.xlsx'
# model_names = ['20250722073446_qwen3:8b', '20250723083603_deepseek-r1:latest']
# save_file = '/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_offline.xlsx'
# extract_json_from_str(data_file, model_names, save_file)

# data_file = '/gruntdata/workDir/dataset/jed_error_report/DD/关键特征提取医师金标准.xlsx'
# model_names = ['医师金标准']
# save_file = '/gruntdata/workDir/dataset/jed_error_report/DD/feature_gold_standard.xlsx'
# extract_json_from_str(data_file, model_names, save_file)
data_file = '/gruntdata/workDir/dataset/jed_error_report/DD/副本input1_online.xlsx'
model_names = ['20250721191521_deepseek-reasoner']
save_file = '/gruntdata/workDir/dataset/jed_error_report/DD/parse_input_online.xlsx'
extract_json_from_str(data_file, model_names, save_file)