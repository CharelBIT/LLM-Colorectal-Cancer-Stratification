from codecs import ignore_errors
import os

import requests
import json
import copy
import pandas as pd
from tqdm import tqdm
import time
api_token = 'sk-**************************'  # replace with your DeepSeek API token

url = 'https://api.deepseek.com/chat/completions'
headers = {
    'Authorization': f'Bearer {api_token}',
    'Content-Type': 'application/json'
}
def chat_with_txt(report_root, prompt_file, model_name, ouput_file):
    prompt_str = open(prompt_file, 'r', encoding='utf-8-sig').read()
    data_df = pd.DataFrame()
    system_info = {"role": "system", "content": prompt_str}
    chat_model_name = model_name
    time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    for file in tqdm(os.listdir(report_root)):
        if not file.endswith('txt'):
            continue
        file_path = os.path.join(report_root, file)
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            report_str = f.read()
        report_message = {"role": "user", "content": report_str}
        message = [system_info, report_message]
       
        response = requests.post(url, headers=headers, json=dict(model=chat_model_name, messages=message, temperature=0.0, stream=False))
       
        print(response)
        response_str = response.json()
        print(response_str)
        data_df.loc[file, "Report"] = report_str
        data_df.loc[file, f"{time_stamp}_{chat_model_name}"] = response_str['choices'][0]['message']['content']
    data_df.to_excel(ouput_file)
def chat_with_excel(report_file, prompt_file, model_name, output_file):
    prompt_str = open(prompt_file, 'r', encoding='utf-8-sig').read()
    if report_file.endswith('.csv'):
        data_df = pd.read_csv(report_file)
    elif report_file.endswith('.xlsx'):
        data_df = pd.read_excel(report_file)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    system_info = {"role": "system", "content": prompt_str}
    chat_model_name = model_name
    time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    for index in tqdm(data_df.index):
        if data_df.loc[index, 'Report'] == 'nan' or pd.isna(data_df.loc[index, 'Report']):
            print('reportp nan')
            continue
        report_message = {"role": "user", "content": data_df.loc[index, 'Report']}
        message = [system_info, report_message]
        try:
            response = requests.post(url, headers=headers, json=dict(model=chat_model_name, messages=message, temperature=0.0, stream=False))
        except:
            continue
        print(response)
        response_str = response.json()
        data_df.loc[index, f"{time_stamp}_{chat_model_name}"] =  response_str['choices'][0]['message']['content']
    data_df.to_excel(output_file, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chat with txt or excel report files.")
    parser.add_argument('--input_path', type=str, help='Path to the input report file (txt or excel).')
    parser.add_argument('--model_name', type=str, help='Name of the chat model to use.')
    parser.add_argument('--output_file', type=str, help='File to save the output results.')
    parser.add_argument('--prompt_file', type=str, help='File containing the prompt for the chat model.')
    args = parser.parse_args()
    print("Input Path:", args.input_path)
    if os.path.isdir(args.input_path):
        print("Processing directory:", args.input_path)
        chat_with_txt(args.input_path, args.prompt_file, args.model_name, args.output_file)
    elif args.input_path.endswith('.csv') or args.input_path.endswith('.xlsx'):
        chat_with_excel(args.input_path, args.prompt_file, args.model_name, args.output_file)
    else:
        raise ValueError("Unsupported file format. Please provide a TXT PATH, CSV, or Excel file.")  