# LLM-Colorectal-Cancer-Stratification

基于大语言模型 (LLM) 的直肠癌报告信息抽取与风险分层分析管线。集成：  
1) LLM结构化关键信息抽取  
2) 风险分层预测 (自动 vs 医师)  
3) 多指标统计评估（Bootstrap、ROC、Kappa、一致性、混淆矩阵）  
4) 可重复环境 (Docker)  

## 研究流程图

待补充

## 目录结构
```
LLM-Colorectal-Cancer-Stratification/
├── ChatWithLLM/                # 与LLM交互与调用脚本
│   ├── chat_offline.py         # 本地/离线模型调用
│   ├── chat_deepseek-reasoner.py # 带推理链模型调用
│   ├── prompt_for_feature_extraction.txt  # 影像/报告要素抽取提示词
│   └── prompt_for_stratification.txt      # 风险分层提示词
├── parse_result/
│   ├── parse_llm_json.py       # 解析LLM输出JSON
│   ├── merge_json_key.py       # 多批次字段合并
│   └── encode_feature.py       # 类别变量编码
├── analysis/
│   ├── boostrap_analysis.py    # Bootstrap + 交叉验证性能估计
│   ├── confusion_matrix.py     # 混淆矩阵与可视化
│   ├── correlation_analysis.py # 指标/特征相关性
│   ├── kappa_analysis.py       # Kappa一致性分析
│   └── FE-stratification.py    # 基于抽取特征的传统模型训练
├── docker/
│   ├── Dockerfile
│   ├── compose.yaml
│   ├── entrypoint.sh
│   └── README.md
└── README.md
```

## 核心流程概览
1. 准备原始文本/结构化报告  
2. 使用 ChatWithLLM 提示词抽取字段 → 生成JSON  
3. parse_result 模块清洗 + 标准化 + 编码  
4. analysis 模块进行：  
   - 传统特征模型 (Logistic / 多分类 OVR)  
   - 自动 vs 医师风险分层比较  
   - Bootstrap 置信区间 / p 值 (成对比较)  
5. 产出指标表与图形 (ROC / 混淆矩阵 / 相关性 / Kappa)  

## ChatWithLLM 模块
| 文件 | 说明 |
|------|------|
| chat_offline.py | 调用本地/容器内已部署模型（如 qwen / deepseek），批量处理 |
| chat_deepseek-reasoner.py | 使用带 reasoning 的模型，保留推理链，便于审计 |
| prompt_for_feature_extraction.txt | 定义需抽取的语义字段（解剖、浸润、淋巴、EMVI 等） |
| prompt_for_stratification.txt | 依据抽取字段 + 规则/指南进行风险等级建议 |



## parse_result 模块
- parse_llm_json.py：校验键、补缺、省略字段填 None  
- merge_json_key.py：不同批次/模型结果对齐合并  
- encode_feature.py：将中文/英文类别映射为数值标签；输出统一 DataFrame 供建模  
- json_key_validator.py：对生成JSON应用严格键集合校验 
输出：`*_merge_encode.xlsx` / `csv`

## analysis 模块详解

### 1. boostrap_analysis.py
功能：  
- 读取线上/线下 LLM 抽取与人工标注  
- 将医师评估与自动模型统一映射到 Label (0=低,1=中,2=高)  
- 5 折 StratifiedKFold × 200 随机种子循环  
- 训练逻辑回归 (OVR) 生成多分类概率  
- 自定义宏平均 ROC (一对多) → mean AUC  
- 记录 Precision / Recall / F1 / Accuracy / AUC  
- 计算每模型指标：均值 + 95% CI (正态近似)  
- 与金标准 (FE-Gold-Standard) 进行成对 Bootstrap 差异：Δ(下限,上限)[p]  

关键函数：  
- marcro_ovsr_roc_curve：自定义宏平均 ROC 聚合  
- mean_95_ci：均值 + 1.96 * SE  
- compare_paired_bootstrap：差值分布 → CI + 经验 p 值  


### 2. confusion_matrix.py
- 多分类混淆矩阵 + 归一化显示  
- 支持中文标签渲染（设置 SimHei 字体）  

### 3. correlation_analysis.py
- 计算特征 / 模型分层输出之间相关系数（Pearson / Spearman）  
- 可筛选显著性阈值 (p < 0.05)  

### 4. kappa_analysis.py
- 计算自动模型 vs 医师 / 医师间一致性 (Cohen's / Fleiss)  
- 输出 Kappa 值 + 解释级别 (轻度/一般/中度/良好/几乎完全)  

### 5. FE-stratification.py
- 使用编码后的特征构建规则或 ML 模型输出风险分层  
- 结果与 LLM 分层结果进行对照  

## Docker 运行
快速复现环境。

### 构建
```bash
docker build -t llm-rc-stratification -f docker/Dockerfile .
```

### 使用 compose
```bash
docker compose -f docker/compose.yaml up -d
```

### 进入容器
```bash
docker exec -it llm-rc /bin/bash
```


## 数据字段 (核心特征示例)
- CRM受累情况  
- EMVI受累情况  
- 直肠系膜内/外淋巴结评估  
- 肿瘤浸润深度 / 位置 / 累及长度  
- LLM 推理生成的风险等级与医师等级  

## 指标说明
| 指标 | 含义 |
|------|------|
| Precision | 宏平均精确率 |
| Recall | 宏平均召回率 |
| F1-score | 宏平均F1 |
| Accuracy | 总体正确率 |
| AUC | 宏平均多分类 ROC AUC |
| Δ vs FE-Gold-Standard | 差值(95%CI)[p] |
