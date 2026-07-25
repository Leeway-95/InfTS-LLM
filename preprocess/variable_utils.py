import pandas as pd
import os

# 数据处理配置
EXCLUDE_COLUMNS = ['date', 'time', 'Date', 'Time', 'label', 'Label', 'News']
EMPTY_VARIABLE_DATASETS = ["TSQA", "AIOps", "NAB", "Oracle", "WeatherQA", "ChatTS"]

def get_dataset_variables(dataset_name, dataset_path=None):
    dataset_variables = {}
    
    # 对于有预定义变量的数据集，直接返回
    if dataset_name in dataset_variables:
        return dataset_variables[dataset_name]
    
    # 对于其他数据集，尝试从文件读取列标题
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            # 排除常见的非变量列
            exclude_cols = EXCLUDE_COLUMNS
            variables = [col for col in df.columns if col not in exclude_cols]
            return ", ".join(variables)
        except Exception:
            pass
    
    # 对于特定模式的数据集，返回空字符串
    if dataset_name in EMPTY_VARIABLE_DATASETS:
        return ""
    
    # 对于以特定前缀开头的数据集，尝试从文件读取
    if dataset_name.startswith(('Weather_', 'Finance_', 'Healthcare_')):
        if dataset_path and os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                exclude_cols = EXCLUDE_COLUMNS
                variables = [col for col in df.columns if col not in exclude_cols]
                return ", ".join(variables)
            except Exception:
                pass

    return ""