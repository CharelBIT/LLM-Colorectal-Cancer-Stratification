import json, re, sys, argparse, pathlib
from typing import List, Dict, Any
import pandas as pd

MANDATORY_KEYS = {
    "CRM受累情况",
    "EMVI受累情况",
    "直肠系膜内淋巴结评估",
    "直肠系膜外淋巴结评估",
    "肿瘤浸润深度",
    "肿瘤的位置",
    "肿瘤累及长度",
    "风险分级"   # 若风险分级是另一步生成，可移到 OPTIONAL_KEYS
}

OPTIONAL_KEYS = {
    "备注",
    "分级理由",
    "分段长度单位"
}

ENUM_MAP = {
    "风险分级": {
        "低": 0, "低危": 0, "low": 0,
        "中": 1, "中危": 1, "intermediate": 1, "moderate": 1,
        "高": 2, "高危": 2, "high": 2
    }
}

def extract_first_json_block(text: str) -> str:
    """
    从含解释/代码块的 LLM 输出中提取第一个 JSON 对象或数组。
    """
    # 去除 markdown 代码围栏
    text = re.sub(r"```[jJ]?[sS]?[oO]?[nN]?", "", text).strip("` \n")
    # 尝试对象
    obj_match = re.search(r"\{.*\}", text, flags=re.S)
    arr_match = re.search(r"\[.*\]", text, flags=re.S)
    cand = None
    if arr_match:
        cand = arr_match.group(0)
    if obj_match and (cand is None or obj_match.start() < arr_match.start()):
        cand = obj_match.group(0)
    if not cand:
        raise ValueError("未找到 JSON 结构")
    # 简单截断匹配首尾大括号/中括号平衡
    return cand

def normalize_value(k: str, v: Any):
    if isinstance(v, str):
        v_strip = v.strip()
        if v_strip == "" or v_strip.lower() in {"na", "none", "null"}:
            return None
        # 枚举映射
        if k in ENUM_MAP:
            key_lower = v_strip.lower()
            # 先直接映射原值，再尝试统一小写
            return ENUM_MAP[k].get(v_strip, ENUM_MAP[k].get(key_lower, v_strip))
        return v_strip
    return v

def validate_record(rec: Dict[str, Any]):
    keys = set(rec.keys())
    missing = MANDATORY_KEYS - keys
    extra = keys - (MANDATORY_KEYS | OPTIONAL_KEYS)
    cleaned = {}
    # 填入必需键
    for k in MANDATORY_KEYS:
        cleaned[k] = normalize_value(k, rec.get(k, None))
    # 填入可选键（若存在）
    for k in OPTIONAL_KEYS:
        if k in rec:
            cleaned[k] = normalize_value(k, rec[k])
    return cleaned, missing, extra

def parse_and_validate(raw_text: str):
    json_str = extract_first_json_block(raw_text)
    data = json.loads(json_str)
    if isinstance(data, dict):
        data = [data]
    cleaned_rows = []
    reports = []
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            reports.append({"index": idx, "error": "非字典记录"})
            continue
        cleaned, missing, extra = validate_record(rec)
        reports.append({
            "index": idx,
            "missing": ";".join(missing) if missing else "",
            "extra": ";".join(extra) if extra else ""
        })
        cleaned_rows.append(cleaned)
    return cleaned_rows, reports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="包含 LLM 原始输出的文本/JSON 文件")
    ap.add_argument("--outfile", required=True, help="清洗后的表格 (xlsx/csv)")
    ap.add_argument("--report", required=True, help="校验报告 (csv)")
    args = ap.parse_args()

    raw_text = pathlib.Path(args.infile).read_text(encoding="utf-8")
    rows, reports = parse_and_validate(raw_text)
    if not rows:
        print("无有效记录", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(rows)
    if args.outfile.endswith(".xlsx"):
        df.to_excel(args.outfile, index=False)
    else:
        df.to_csv(args.outfile, index=False)

    pd.DataFrame(reports).to_csv(args.report, index=False)
    print(f"Done. valid={len(rows)} report={args.report}")

if __name__ == "__main__":
    main()