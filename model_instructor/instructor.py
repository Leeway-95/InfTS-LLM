import sys
import time
import json
import pandas as pd
import ast
from datetime import datetime
from tqdm import tqdm
from memory_pool import MemoryPool, update_memory_pool
from llm_api import callLLM, parse_llm_output
from file_io import save_full_response, save_log_entry, save_memory_state, load_memory_state, update_csv
from prompt_builder import build_pcot_prompt
from utils.config import *
from utils.common import process_series, process_result
from utils.parser import safe_parse_series
from utils.tools import valid_call
from utils.logging import get_project_logger, Tee
logger = get_project_logger(__name__)
from utils.config import HistLen
from utils.config import PreLen

DEFAULT_HIST_LEN = HistLen[0]
DEFAULT_PRED_LEN = PreLen[0]

def generate_predict_compare_csv(id_val, pred_labels, pred_series_str, impact_scores_str, dataset_name, method,
                                 task_type, hist_len=None, pred_len=None):
    """
    生成predict_compare.csv文件，包含预测结果与真实值的比较

    Args:
        id_val: 数据ID
        pred_labels: 预测标签
        pred_series_str: 预测序列字符串
        impact_scores_str: 影响分数字符串
        dataset_name: 数据集名称
        method: 方法名称
        task_type: 任务类型
        hist_len: 历史长度 (可选)
        pred_len: 预测长度 (可选)
    """
    try:

        # 检查数据集是否属于当前任务类型 - 与update_csv函数完全一致
        if task_type == "UNDERSTANDING" and dataset_name not in DATASET_UNDERSTANDING:
            # logger.info(f"Skipping dataset {dataset_name} for UNDERSTANDING task")
            return
        elif task_type == "FORECASTING_NUM" and dataset_name not in DATASET_FORECASTING_NUM:
            # logger.info(f"Skipping dataset {dataset_name} for FORECASTING_NUM task")
            return
        elif task_type == "FORECASTING_EVENT" and dataset_name not in DATASET_FORECASTING_EVENT:
            # logger.info(f"Skipping dataset {dataset_name} for FORECASTING_EVENT task")
            return
        elif task_type == "REASONING" and dataset_name not in DATASET_REASONING:
            # logger.info(f"Skipping dataset {dataset_name} for REASONING task")
            return

        # 检查方法是否属于当前任务类型 - 只对Baseline方法进行过滤，OUR_Method不需要过滤
        if method not in OUR_Method:  # 只对非OUR_Method的方法进行过滤
            if task_type == "UNDERSTANDING" and method not in BASELINE_UNDERSTANDING:
                logger.info(f"Skipping method {method} for UNDERSTANDING task")
                return
            elif task_type == "FORECASTING_NUM" and method not in BASELINE_FORECASTING_NUM:
                # logger.info(f"Skipping method {method} for FORECASTING_NUM task")
                return
            elif task_type == "FORECASTING_EVENT" and method not in BASELINE_FORECASTING_EVENT:
                # logger.info(f"Skipping method {method} for FORECASTING_EVENT task")
                return
            elif task_type == "REASONING" and method not in BASELINE_REASONING:
                # logger.info(f"Skipping method {method} for REASONING task")
                return

        # 从DATASET_PATHS中读取路径信息 - 与update_csv函数完全一致
        if dataset_name in DATASET_PATHS:
            dataset_path = DATASET_PATHS[dataset_name]
        elif dataset_name in DATASET_MERGE_PATHS:
            dataset_path = DATASET_MERGE_PATHS[dataset_name]
        else:
            raise KeyError(f"Dataset path not found for {dataset_name} in either DATASET_PATHS or DATASET_MERGE_PATHS")

        dirname = os.path.dirname(dataset_path)
        filename = os.path.splitext(os.path.basename(dataset_path))[0]
        output_dir = os.path.join(dirname, f"predict-{filename}")
        os.makedirs(output_dir, exist_ok=True)

        # 处理HistLen和PredLen参数
        # 对于 UNDERSTANDING 和 REASONING 任务，如果没有传入参数，需要从数据中计算
        if hist_len is None or pred_len is None:
            # 读取stream_summary.csv获取实际数据
            stream_detection_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")
            if os.path.exists(stream_detection_path):
                try:
                    stream_df = pd.read_csv(stream_detection_path)
                    if id_val in stream_df.index:
                        stream_row = stream_df.loc[id_val]

                        # 获取完整序列长度作为 HistLen
                        if hist_len is None and task_type in ["UNDERSTANDING", "REASONING"]:
                            if 'Series' in stream_row:
                                try:
                                    full_series = json.loads(stream_row['Series'])
                                    hist_len = len(full_series)
                                except (json.JSONDecodeError, TypeError):
                                    try:
                                        # 如果json.loads失败，尝试使用ast.literal_eval解析
                                        full_series = ast.literal_eval(stream_row['Series'])
                                        hist_len = len(full_series)
                                    except (ValueError, SyntaxError):
                                        # 如果包含NaN值，尝试手动处理
                                        try:
                                            import numpy as np
                                            series_str = str(stream_row['Series']).replace('nan', 'null')
                                            full_series = json.loads(series_str)
                                            hist_len = len(full_series)
                                        except:
                                            logger.warning(
                                                f"Failed to parse Series data for hist_len calculation, using default value 288")
                                            hist_len = DEFAULT_HIST_LEN  # 使用默认值
                            else:
                                hist_len = DEFAULT_HIST_LEN  # 使用默认值
                        elif hist_len is None:
                            hist_len = HistLen[0] if isinstance(HistLen, list) else HistLen

                        # 获取预测标签长度作为 PredLen
                        if pred_len is None and task_type in ["UNDERSTANDING", "REASONING"]:
                            if 'Labels' in stream_row and stream_row['Labels'] and stream_row['Labels'].strip():
                                try:
                                    pred_labels_data = json.loads(stream_row['Labels'])
                                    pred_len = len(pred_labels_data)
                                except (json.JSONDecodeError, TypeError):
                                    try:
                                        pred_labels_data = ast.literal_eval(stream_row['Labels'])
                                        pred_len = len(pred_labels_data)
                                    except (ValueError, SyntaxError):
                                        pred_len = DEFAULT_PRED_LEN  # 使用默认值
                            else:
                                pred_len = DEFAULT_PRED_LEN  # 使用默认值
                        elif pred_len is None:
                            pred_len = PreLen[0] if isinstance(PreLen, list) else PreLen
                except Exception as e:
                    logger.warning(f"Error reading stream data for parameter calculation: {e}")
                    # 使用配置文件中的默认值
                    if hist_len is None:
                        hist_len = HistLen[0] if isinstance(HistLen, list) else HistLen
                    if pred_len is None:
                        pred_len = PreLen[0] if isinstance(PreLen, list) else PreLen
            else:
                # 如果文件不存在，使用配置文件中的默认值
                if hist_len is None:
                    hist_len = HistLen[0] if isinstance(HistLen, list) else HistLen
                if pred_len is None:
                    pred_len = PreLen[0] if isinstance(PreLen, list) else PreLen

        # 确保输出文件路径
        predict_compare_path = os.path.join(output_dir, 'predict_compare.csv')
        stream_detection_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")

        # 读取stream_summary.csv获取真实标签和Series
        stream_df = pd.read_csv(stream_detection_path)

        # 检查id_val是否存在于stream_df中
        if id_val not in stream_df.index:
            logger.warning(f"ID {id_val} does not exist in stream_summary.csv")
            # 使用空值作为默认
            stream_row = pd.Series()
            true_labels = []
            full_series = []
        else:
            stream_row = stream_df.loc[id_val]
            true_labels = []
            if 'Labels' in stream_row and stream_row['Labels'] and stream_row['Labels'].strip():
                try:
                    # 首先尝试使用json.loads解析
                    true_labels = json.loads(stream_row['Labels'])
                except json.JSONDecodeError:
                    try:
                        # 如果json.loads失败，尝试使用ast.literal_eval解析
                        true_labels = ast.literal_eval(stream_row['Labels'])
                    except (ValueError, SyntaxError):
                        logger.warning(f"Unable to parse Labels field: {stream_row['Labels'][:50]}...")
                        true_labels = []
            # 安全解析Series数据，处理NaN值
            if 'Series' in stream_row and stream_row['Series'] and stream_row['Series'].strip():
                try:
                    # 首先尝试使用json.loads解析
                    full_series = json.loads(stream_row['Series'])
                except json.JSONDecodeError:
                    try:
                        # 如果json.loads失败，尝试使用ast.literal_eval解析
                        full_series = ast.literal_eval(stream_row['Series'])
                    except (ValueError, SyntaxError):
                        logger.warning(f"Unable to parse Series field: {stream_row['Series'][:50]}...")
                        # 如果包含NaN值，尝试手动处理
                        try:
                            import numpy as np
                            # 将NaN替换为None或0，然后解析
                            series_str = str(stream_row['Series']).replace('nan', 'null')
                            full_series = json.loads(series_str)
                            # 将null转换为0或适当的数值
                            full_series = [0 if x is None else x for x in full_series]
                        except:
                            logger.warning(f"Failed to parse Series data, using empty list")
                            full_series = []
            else:
                full_series = []

        # 解析预测标签和序列
        pred_labels_list = pred_labels if isinstance(pred_labels, list) else (
            json.loads(pred_labels) if pred_labels else [])
        # 处理pred_series_str，可能是字符串或已经是列表
        if isinstance(pred_series_str, list):
            pred_series_list = pred_series_str
        elif isinstance(pred_series_str, str) and pred_series_str:
            try:
                pred_series_list = json.loads(pred_series_str)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f" Failed to parse pred_series_str as JSON: {e}")
                pred_series_list = []
        else:
            pred_series_list = []

        # 首先确定真实序列的长度（根据pred_len参数或full_series长度）
        if pred_len and pred_len > 0:
            # 使用配置的预测长度作为真实序列长度
            target_series_len = pred_len
        else:
            # 如果没有配置预测长度，使用整个真实序列长度
            target_series_len = len(full_series)

        # 获取真实序列的最后target_series_len个值
        if target_series_len > 0 and len(full_series) >= target_series_len:
            true_series_tail = full_series[-target_series_len:]
        elif target_series_len > 0 and len(full_series) < target_series_len:
            # 如果真实序列长度不足，使用整个真实序列
            true_series_tail = full_series
        else:
            true_series_tail = []

        if task_type == "UNDERSTANDING" or task_type == "REASONING" or task_type == "FORECASTING_EVENT":
            true_series_tail = []

        # 根据真实序列长度调整预测序列
        true_series_len = len(true_series_tail)

        if len(pred_series_list) > true_series_len:
            # 如果预测序列长度大于真实序列长度，截取预测序列
            pred_series_list = pred_series_list[:true_series_len]
        elif len(pred_series_list) < true_series_len:
            # 如果预测序列长度小于真实序列长度，用最后一个值补全或用0补全
            if len(pred_series_list) > 0:
                # 用预测序列的最后一个值补全
                last_value = pred_series_list[-1]
                pred_series_list.extend([last_value] * (true_series_len - len(pred_series_list)))
            else:
                # 如果预测序列为空，用0补全
                pred_series_list = [0.0] * true_series_len

        if task_type == "UNDERSTANDING" or task_type == "REASONING" or task_type == "FORECASTING_EVENT":
            pred_series_list = []

        # 重新计算预测序列长度，确保与真实序列长度一致
        pred_series_len = len(pred_series_list)

        # 计算标签相似度
        labels_accuracy = 0.0
        if pred_labels_list and true_labels:
            # 简化的标签匹配逻辑
            if len(pred_labels_list) == len(true_labels):
                # 计算完全匹配的比例
                matches = sum(1 for p, t in zip(pred_labels_list, true_labels) if p == t)
                labels_accuracy = matches / len(true_labels)
            else:
                # 如果长度不同，计算预测值与真实值中对应位置的匹配比例
                min_len = min(len(pred_labels_list), len(true_labels))
                matches = sum(1 for i in range(min_len) if pred_labels_list[i] == true_labels[i])
                labels_accuracy = matches / len(true_labels) if true_labels else 0.0

        # 计算序列的平均绝对误差(MAE) - 对于FORECASTING_EVENT任务不计算
        series_mae = 0.0
        if task_type != "FORECASTING_EVENT" and pred_series_list and true_series_tail:
            min_len = min(len(pred_series_list), len(true_series_tail))
            if min_len > 0:
                # 计算平均绝对误差(MAE)
                absolute_errors = [abs(float(pred_series_list[i]) - float(true_series_tail[i])) for i in range(min_len)]
                series_mae = sum(absolute_errors) / min_len

        # 计算事件预测的F1和AUC指标（仅对FORECASTING_EVENT任务）
        pred_labels_f1 = 0.0
        pred_labels_auc = 0.0

        if task_type == "FORECASTING_EVENT" and pred_labels_list and true_labels:
            # 导入计算事件指标的函数
            try:
                # 动态导入以避免循环导入
                import sys
                import os as os_module
                sys.path.append(os_module.path.join(os_module.path.dirname(__file__), '..', 'postprocess'))
                from postprocess.analyze_results import calculate_event_metrics

                # 解析impact_scores_str
                impact_scores_list = []
                if isinstance(impact_scores_str, str) and impact_scores_str:
                    try:
                        impact_scores_list = json.loads(impact_scores_str)
                    except (json.JSONDecodeError, TypeError):
                        impact_scores_list = []
                elif isinstance(impact_scores_str, list):
                    impact_scores_list = impact_scores_str

                # 计算事件指标
                event_metrics = calculate_event_metrics(pred_labels_list, true_labels, impact_scores_list)
                pred_labels_f1 = event_metrics.get("f1", 0.0)
                pred_labels_auc = event_metrics.get("auc", 0.0)

            except Exception as e:
                logger.warning(f"Failed to calculate event metrics: {e}")
                pred_labels_f1 = 0.0
                pred_labels_auc = 0.0

        # 准备比较数据 - 确保Task列在Index列之后，添加F1和AUC列
        # 对于FORECASTING_EVENT任务，不写入Pred_Series相关数据
        compare_data = {
            "Index": id_val,
            "Task": task_type,
            "Dataset": dataset_name,
            "Method": method,
            "Pred_Labels_Accuracy": labels_accuracy,
            "Pred_Series_MAE": series_mae,
            "Pred_Labels_F1Score": pred_labels_f1,
            "Pred_Labels_AUC": pred_labels_auc,
            "Pred_Labels_Len": len(pred_labels_list),
            "Pred_Labels_Truth_Len": len(true_labels),
            "HistLen": int(hist_len),
            "Pred_Series_Len": pred_series_len,
            "Pred_Series_Truth_Len": len(true_series_tail),
            "Pred_Labels": str(pred_labels_list),
            "Pred_Labels_Truth": str(true_labels)
        }

        # 只有非FORECASTING_EVENT任务才写入Pred_Series数据
        if task_type != "FORECASTING_EVENT":
            compare_data["Pred_Series"] = str(pred_series_list)
            compare_data["Pred_Series_Truth"] = str(true_series_tail)
        else:
            compare_data["Pred_Series"] = ""
            compare_data["Pred_Series_Truth"] = ""

        # 如果文件不存在，创建新文件；否则追加数据 - 与update_csv函数保持一致的Index生成逻辑
        if not os.path.exists(predict_compare_path):
            compare_df = pd.DataFrame([compare_data])
            compare_df.to_csv(predict_compare_path, index=False)
        else:
            compare_df = pd.DataFrame([compare_data])
            existing_df = pd.read_csv(predict_compare_path)

            # 确保所有需要的列都存在
            required_columns = ["Index", "Task", "Dataset", "Method"]
            for col in required_columns:
                if col not in existing_df.columns:
                    existing_df[col] = None

            updated_df = pd.concat([existing_df, compare_df], ignore_index=True)
            # 重新生成连续的索引，确保不重复 - 与update_csv函数完全一致
            updated_df['Index'] = range(1, len(updated_df) + 1)
            updated_df.to_csv(predict_compare_path, index=False)
    except Exception as e:
        logger.error(f"Prediction comparison CSV error: {str(e)}")
        raise


if __name__ == "__main__":
    original_stdout = sys.stdout

    # 确保日志目录存在
    logs_dir = "../logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # 创建按日期命名的日志文件
    current_date = datetime.now().strftime("%Y%m%d")
    log_file_path = os.path.join(logs_dir, f"log_{current_date}.txt")

    # 保持文件打开状态直到程序结束
    log_file = open(log_file_path, 'a', encoding='utf-8')
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"{'=' * 120}\n")
    log_file.write(f"New session started at: {start_time}\n")
    log_file.write(f"{'=' * 120}\n")
    sys.stdout = Tee(original_stdout, log_file)
    memory_pool = MemoryPool()
    if MEMORY_STORAGE_MODE == MODE_FILE:
        load_memory_state(memory_pool)

        # 记录当前处理状态，用于判断是否需要清空记忆池
        current_task = None
        current_dataset = None
        current_method = None

        # 处理每个数据集 - 避免重复处理同一数据集
        all_datasets = set()
        if "FORECASTING_NUM" in TASK:
            all_datasets = all_datasets | set(DATASET_FORECASTING_NUM)
        if "FORECASTING_EVENT" in TASK:
            all_datasets = all_datasets | set(DATASET_FORECASTING_EVENT)
        if "UNDERSTANDING" in TASK:
            all_datasets = all_datasets | set(DATASET_UNDERSTANDING)
        if "REASONING" in TASK:
            all_datasets = all_datasets | set(DATASET_REASONING)
        for dataset_name in all_datasets:
            if dataset_name in DATASET_PATHS or dataset_name in DATASET_MERGE_PATHS:
                if dataset_name in DATASET_MERGE_PATHS:
                    dataset_path = DATASET_MERGE_PATHS[dataset_name]
                else:
                    dataset_path = DATASET_PATHS[dataset_name]

                dirname = os.path.dirname(dataset_path)
                filename = os.path.splitext(os.path.basename(dataset_path))[0]
                stream_detection_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")
                # 读取stream_summary.csv获取真实标签和Series
                stream_df = pd.read_csv(stream_detection_path)
                detection_path = os.path.join(dirname, f"detection-{filename}", 'detection_summary.csv')
                detection_df = pd.read_csv(detection_path)
                detection_dict = {}

                # 处理代表性子序列位置和分数
                for idx, row in detection_df.iterrows():
                    # 从stream_summary.csv获取Series数据
                    stream_row = stream_df.loc[idx]

                    # 使用新的安全解析函数处理Series数据
                    series_list = safe_parse_series(stream_row['Series'])
                    if not series_list:
                        print(f"Warning: Unable to parse Series data at index {idx}, skipping...")
                        continue

                    total_length = len(series_list)
                    try:
                        # 检查值是否为NaN或非字符串类型
                        pos_value = row.get('Representative_Subsequence_Positions', None)
                        if pd.isna(pos_value) or not isinstance(pos_value, str):
                            rep_positions = []
                        else:
                            rep_positions = json.loads(pos_value)
                    except (json.JSONDecodeError, TypeError):
                        # 尝试使用ast.literal_eval作为备选方案
                        try:
                            if pd.notna(row.get('Representative_Subsequence_Positions', None)):
                                rep_positions = ast.literal_eval(str(row['Representative_Subsequence_Positions']))
                            else:
                                rep_positions = []
                        except (ValueError, SyntaxError):
                            # 如果两种方法都失败，使用空列表作为默认值
                            logger.warning(
                                f"Unable to parse Representative_Subsequence_Positions, using empty list")
                            rep_positions = []
                    valid_rep_positions = []
                    for pos in rep_positions:
                        if isinstance(pos, (tuple, list)) and len(pos) == 2:
                            valid_rep_positions.append(pos)
                        elif isinstance(pos, int):
                            valid_rep_positions.append(tuple(rep_positions))
                            break
                    r_scores = [(end - start) / total_length for start, end in valid_rep_positions]
                    detection_dict[idx] = r_scores
                sorted_ids = sorted(detection_df.index)
                # logger.info(f"Found {len(sorted_ids)} rows to process")  # 减少控制台输出

                # 遍历TASK列表，根据任务类型选择不同的方法
                task_progress = tqdm(TASK, leave=True, disable=not any(valid_call(task, dataset_name, method) for task in TASK for method in (BASELINE_UNDERSTANDING + BASELINE_FORECASTING_NUM + BASELINE_FORECASTING_EVENT + BASELINE_REASONING + OUR_Method)))
                for inner_task_type in task_progress:
                    # 检查任务是否发生变化，如果变化则清空记忆池
                    if current_task != inner_task_type:
                        if current_task is not None and os.path.exists(Memory_Pool_PATH):
                            os.remove(Memory_Pool_PATH)
                            # logger.info(f"Memory pool cleared due to task change from {current_task} to {inner_task_type}: {Memory_Pool_PATH}")  # 减少控制台输出，避免中断进度条
                            memory_pool = MemoryPool()  # 重新初始化记忆池
                        current_task = inner_task_type

                    # 检查数据集是否发生变化，如果变化则清空记忆池
                    if current_dataset != dataset_name:
                        if current_dataset is not None and os.path.exists(Memory_Pool_PATH):
                            os.remove(Memory_Pool_PATH)
                            # logger.info(f"Memory pool cleared due to dataset change from {current_dataset} to {dataset_name}: {Memory_Pool_PATH}")  # 减少控制台输出，避免中断进度条
                            memory_pool = MemoryPool()  # 重新初始化记忆池
                        current_dataset = dataset_name
                    # 根据任务类型选择对应的方法列表，包含OUR_Method
                    if inner_task_type == "UNDERSTANDING":
                        methods_to_use = BASELINE_UNDERSTANDING + OUR_Method
                    elif inner_task_type == "FORECASTING_NUM":
                        methods_to_use = BASELINE_FORECASTING_NUM + OUR_Method
                    elif inner_task_type == "FORECASTING_EVENT":
                        methods_to_use = BASELINE_FORECASTING_EVENT + OUR_Method
                    elif inner_task_type == "REASONING":
                        methods_to_use = BASELINE_REASONING + OUR_Method

                    # 遍历选定的方法列表、PreLen和HistLen
                    method_progress = tqdm(methods_to_use,
                                           desc=f"Task: {inner_task_type} | Dataset: {dataset_name} | Method: {methods_to_use} | {time.strftime('%Y-%m-%d %H:%M:%S')}",
                                           leave=True, disable=not any(valid_call(inner_task_type, dataset_name, method) for method in methods_to_use))
                    for method in method_progress:
                        # 检查方法是否发生变化，如果变化则清空记忆池
                        if current_method != method:
                            if current_method is not None and os.path.exists(Memory_Pool_PATH):
                                os.remove(Memory_Pool_PATH)
                                # logger.info(f"Memory pool cleared due to method change from {current_method} to {method}: {Memory_Pool_PATH}")  # 减少控制台输出，避免中断进度条
                                memory_pool = MemoryPool()  # 重新初始化记忆池
                            current_method = method

                        # 对于 UNDERSTANDING 和 REASONING 任务，不遍历 PreLen 和 HistLen 数组
                        if inner_task_type in ["UNDERSTANDING", "REASONING"]:
                            # 检查数据集是否属于当前任务类型
                            if inner_task_type == "UNDERSTANDING" and dataset_name not in DATASET_UNDERSTANDING:
                                continue
                            elif inner_task_type == "REASONING" and dataset_name not in DATASET_REASONING:
                                continue

                            # 检查方法是否属于当前任务类型 - 只对Baseline方法进行过滤，OUR_Method不需要过滤
                            if method not in OUR_Method:  # 只对非OUR_Method的方法进行过滤
                                if inner_task_type == "UNDERSTANDING" and method not in BASELINE_UNDERSTANDING:
                                    continue
                                elif inner_task_type == "REASONING" and method not in BASELINE_REASONING:
                                    continue

                            # 对于 UNDERSTANDING 和 REASONING 任务，只执行一次，不遍历 PreLen 和 HistLen
                            # HistLen 和 PredLen 将在处理每行数据时根据实际数据长度设置

                            # 处理每一行数据
                            row_progress = tqdm(sorted_ids,
                                                desc=f"Task: {inner_task_type} | Dataset: {dataset_name} | Method: {method} | {time.strftime('%Y-%m-%d %H:%M:%S')}",
                                                leave=True, disable=not valid_call(inner_task_type, dataset_name, method))
                            for id_val in row_progress:
                                # 为UNDERSTANDING和REASONING任务添加统一的日志标题格式
                                log_entry = f"\n{'=' * 120}\nTask: {inner_task_type} | Dataset: {dataset_name} | Method: {method} | Index: {id_val + 1} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 120}"
                                log_file.write(log_entry + "\n")
                                # 从stream_summary.csv获取Series数据
                                stream_row = stream_df.loc[id_val]
                                # 检查是否存在Series列，如果不存在则使用空列表
                                # 使用新的安全解析函数处理Series数据
                                if 'Series' in stream_row:
                                    full_series = safe_parse_series(stream_row['Series'])
                                else:
                                    full_series = []

                                # 获取预测标签数据
                                pred_labels_data = []
                                if 'Labels' in stream_row and stream_row['Labels'] and stream_row['Labels'].strip():
                                    try:
                                        # 首先尝试使用json.loads解析
                                        pred_labels_data = json.loads(stream_row['Labels'])
                                    except json.JSONDecodeError:
                                        try:
                                            # 如果json.loads失败，尝试使用ast.literal_eval解析
                                            pred_labels_data = ast.literal_eval(stream_row['Labels'])
                                        except (ValueError, SyntaxError):
                                            logger.warning(
                                                f" Unable to parse Labels field: {stream_row['Labels'][:50]}...")
                                            pred_labels_data = []

                                # 为 UNDERSTANDING 和 REASONING 任务设置 HistLen 和 PredLen
                                hist_len = len(full_series)  # HistLen = 完整序列长度
                                pred_len = len(pred_labels_data)  # PredLen = 预测标签长度

                                # logger.info(f"\n{'=' * 120}")
                                # logger.info(
                                #     f"Task: {inner_task_type} | Dataset: {dataset_name} | Method: {method} | HistLen: {hist_len} | PredLen: {pred_len} | Index: {id_val + 1} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
                                # logger.info(f"{'=' * 120}")  # 减少控制台输出
                                row = detection_df.loc[id_val]

                                # 处理 Positions 字段，增加更完善的错误处理和类型检查
                                positions = []
                                if 'Positions' in stream_row:
                                    pos_value = stream_row['Positions']
                                    # 检查值是否为NaN或None
                                    if pd.isna(pos_value) or pos_value is None:
                                        logger.warning(f" Positions field is NaN or None for ID {id_val}")
                                        positions = []
                                    elif isinstance(pos_value, str) and pos_value.strip():
                                        try:
                                            # 首先尝试使用json.loads解析
                                            positions = json.loads(pos_value)
                                        except json.JSONDecodeError:
                                            try:
                                                # 如果json.loads失败，尝试使用ast.literal_eval解析
                                                positions = ast.literal_eval(pos_value)
                                            except (ValueError, SyntaxError):
                                                logger.warning(f" Unable to parse Positions field: {pos_value[:50]}...")
                                                positions = []
                                        except TypeError:
                                            logger.warning(f" Positions field has invalid type: {type(pos_value)}")
                                            positions = []
                                    elif isinstance(pos_value, (list, tuple)):
                                        # 如果已经是列表或元组类型，直接使用
                                        positions = list(pos_value)
                                    else:
                                        logger.warning(
                                            f" Positions field has unexpected type {type(pos_value)}: {pos_value}")
                                        positions = []
                                else:
                                    logger.warning(f" Positions field not found in stream_row for ID {id_val}")
                                    positions = []

                                maxlen = len(full_series)
                                # 对于Understanding任务，recent_series应该是完整的序列（除了预测部分）
                                # 如果有预测标签，则使用除预测部分外的序列；否则使用完整序列
                                if pred_len > 0 and maxlen > pred_len:
                                    # 使用序列的前 maxlen - pred_len 部分作为历史序列
                                    recent_series = ', '.join(map(str, full_series[:maxlen - pred_len]))
                                else:
                                    # 如果没有预测标签或序列长度不足，使用完整序列
                                    recent_series = ', '.join(map(str, full_series))

                                try:
                                    pos_value = row.get('Representative_Subsequence_Positions', None)
                                    if pd.isna(pos_value) or not isinstance(pos_value, str):
                                        rep_positions = []
                                    else:
                                        rep_positions = json.loads(pos_value)
                                except (json.JSONDecodeError, TypeError):
                                    # 尝试使用ast.literal_eval作为备选方案
                                    try:
                                        if pd.notna(row.get('Representative_Subsequence_Positions', None)):
                                            rep_positions = ast.literal_eval(
                                                str(row['Representative_Subsequence_Positions']))
                                        else:
                                            rep_positions = []
                                    except (ValueError, SyntaxError):
                                        # 如果两种方法都失败，使用空列表作为默认值
                                        logger.warning(
                                            f" Unable to parse Representative_Subsequence_Positions, using empty list")
                                        rep_positions = []

                                rep_subsequences = []
                                valid_rep_positions = []
                                # 检查每个元素，确保它们是元组或列表（长度为2）
                                for pos in rep_positions:
                                    if isinstance(pos, (tuple, list)) and len(pos) == 2:
                                        valid_rep_positions.append(pos)
                                    elif isinstance(pos, int):
                                        # 如果是单个整数，使用整个rep_positions作为元组
                                        valid_rep_positions.append(tuple(rep_positions))
                                        break

                                # 使用验证过的位置创建子序列
                                for start, end in valid_rep_positions:
                                    rep_subsequences.append(full_series[start:end + 1])
                                rep_series = '\n'.join([', '.join(map(str, seq)) for seq in rep_subsequences])

                                if recent_series:
                                    full_prompt, images = build_pcot_prompt(id_val, full_series, recent_series,
                                                                            rep_series,
                                                                            memory_pool, dataset_name, method,
                                                                            pred_len, hist_len, positions)
                                    log_file.write("\n[Full Prompt]\n")
                                    log_file.write(full_prompt)

                                    result, in_tokens, out_tokens, ttft, resp_time, total_time, cost = callLLM(
                                        full_prompt, images)
                                    
                                    # 只在LLM调用成功时才记录日志
                                    if not result.startswith("Error:"):
                                        log_file.write(f"\n{'-' * 120}\n")
                                        log_file.write("[Full Response]\n")
                                        log_file.write(result)

                                        response_file = save_full_response(id_val, result, dataset_name, method,
                                                                           inner_task_type, hist_len, pred_len)
                                        log_entry = {
                                            'ID': id_val,
                                            'Task': inner_task_type,
                                            'Dataset': dataset_name,
                                            'Method': method,
                                            'TTFT': ttft,
                                            'InputTokens': in_tokens,
                                            'OutputTokens': out_tokens,
                                            'Cost': cost,
                                            'TotalTime': total_time,
                                            'ResponseFile': response_file
                                        }
                                        save_log_entry(log_entry, inner_task_type, method)
                                        pred_labels, pred_series_str, impact_scores_str = parse_llm_output(
                                            result, pred_len, None, inner_task_type, method)

                                        if method == "InfTS-LLM" or method == "InfTS-LLM (+v)":
                                            try:
                                                impact_scores = json.loads(impact_scores_str)
                                            except:
                                                impact_scores = []
                                            r_scores = detection_dict.get(id_val, [])
                                            update_memory_pool(rep_series, rep_positions, r_scores, impact_scores,
                                                               memory_pool)
                                            if MEMORY_STORAGE_MODE == MODE_FILE:
                                                save_memory_state(memory_pool)

                                        update_csv(id_val, pred_labels, pred_series_str, impact_scores_str,
                                                   dataset_name, inner_task_type, method, hist_len, pred_len)
                                        generate_predict_compare_csv(id_val, pred_labels, pred_series_str,
                                                                     impact_scores_str, dataset_name, method,
                                                                     inner_task_type,
                                                                     hist_len, pred_len)

                        else:
                            # 对于 FORECASTING_NUM 任务，保持原有的双重循环逻辑
                            for pred_len in PreLen:
                                for hist_len in HistLen:
                                    # 检查数据集是否属于当前任务类型
                                    if inner_task_type == "FORECASTING_NUM" and dataset_name not in DATASET_FORECASTING_NUM:
                                        continue
                                    elif inner_task_type == "FORECASTING_EVENT" and dataset_name not in DATASET_FORECASTING_EVENT:
                                        continue

                                    # 检查方法是否属于当前任务类型 - 只对Baseline方法进行过滤，OUR_Method不需要过滤
                                    if method not in OUR_Method:  # 只对非OUR_Method的方法进行过滤
                                        if inner_task_type == "FORECASTING_NUM" and method not in BASELINE_FORECASTING_NUM:
                                            continue
                                        elif inner_task_type == "FORECASTING_EVENT" and method not in BASELINE_FORECASTING_EVENT:
                                            continue

                                    # 处理每一行数据
                                    row_progress = tqdm(sorted_ids,
                                                        desc=f"Task: {inner_task_type} | Dataset: {dataset_name} | Method: {method} | HistLen: {hist_len} | PredLen: {pred_len} | {time.strftime('%Y-%m-%d %H:%M:%S')}",
                                                        leave=True, disable=not valid_call(inner_task_type, dataset_name, method))
                                    for id_val in row_progress:
                                        # 移除控制台打印，只保留日志记录
                                        log_entry = f"\n{'=' * 120}\nTask: {inner_task_type} | Dataset: {dataset_name} | Method: {method} | HistLen: {hist_len} | PredLen: {pred_len} | Index: {id_val + 1} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 120}"
                                        log_file.write(log_entry + "\n")
                                        row = detection_df.loc[id_val]
                                        # 从stream_summary.csv获取Series数据
                                        stream_row = stream_df.loc[id_val]
                                        # 检查是否存在Series列，如果不存在则使用空列表
                                        if 'Series' in stream_row:
                                            try:
                                                series_data = stream_row['Series']
                                                if pd.isna(series_data) or series_data is None:
                                                    logger.warning(
                                                        f"Series data at index {id_val}: Series field is NaN or None")
                                                    full_series = []
                                                elif isinstance(series_data, str):
                                                    # 预处理Series字符串，处理NaN值
                                                    series_data = series_data.replace('nan', 'null').replace('NaN',
                                                                                                             'null').replace(
                                                        'NAN', 'null')
                                                    try:
                                                        full_series = json.loads(series_data)
                                                    except json.JSONDecodeError as e:
                                                        logger.warning(
                                                            f"Error parsing Series data at index {id_val}: {e}")
                                                        logger.warning(f"Series data: {series_data[:100]}...")
                                                        full_series = []
                                                elif isinstance(series_data, (list, tuple)):
                                                    full_series = list(series_data)
                                                else:
                                                    logger.warning(
                                                        f"Series data at index {id_val}: Unexpected type {type(series_data)}")
                                                    full_series = []
                                            except (json.JSONDecodeError, TypeError) as e:
                                                logger.warning(f"Error parsing Series data at index {id_val}: {e}")
                                                logger.warning(
                                                    f"Series data: {stream_row['Series'][:100] if isinstance(stream_row['Series'], str) else stream_row['Series']}")
                                                logger.warning(f" Unable to parse Series field, using empty list")
                                                full_series = []
                                        else:
                                            full_series = []
                                        # 处理 Positions 字段，增加更完善的错误处理和类型检查
                                        positions = []
                                        if 'Positions' in stream_row:
                                            pos_value = stream_row['Positions']
                                            # 检查值是否为NaN或None
                                            if pd.isna(pos_value) or pos_value is None:
                                                logger.warning(f" Positions field is NaN or None for ID {id_val}")
                                                positions = []
                                            elif isinstance(pos_value, str) and pos_value.strip():
                                                try:
                                                    # 首先尝试使用json.loads解析
                                                    positions = json.loads(pos_value)
                                                except json.JSONDecodeError:
                                                    try:
                                                        # 如果json.loads失败，尝试使用ast.literal_eval解析
                                                        positions = ast.literal_eval(pos_value)
                                                    except (ValueError, SyntaxError):
                                                        logger.warning(
                                                            f" Unable to parse Positions field: {pos_value[:50]}...")
                                                        positions = []
                                                except TypeError:
                                                    logger.warning(
                                                        f" Positions field has invalid type: {type(pos_value)}")
                                                    positions = []
                                            elif isinstance(pos_value, (list, tuple)):
                                                # 如果已经是列表或元组类型，直接使用
                                                positions = list(pos_value)
                                            else:
                                                logger.warning(
                                                    f" Positions field has unexpected type {type(pos_value)}: {pos_value}")
                                                positions = []
                                        else:
                                            logger.warning(f" Positions field not found in stream_row for ID {id_val}")
                                            positions = []
                                        maxlen = len(full_series)
                                        # 对于FORECASTING_NUM任务，recent_series应该使用配置的hist_len长度的历史数据
                                        # 从序列末尾取hist_len个数据点作为历史序列
                                        if maxlen >= hist_len:
                                            # 使用序列的最后 hist_len 个数据点作为历史序列
                                            recent_series = ', '.join(map(str, full_series[-hist_len:]))
                                        else:
                                            # 如果序列长度不足hist_len，使用完整序列
                                            recent_series = ', '.join(map(str, full_series))
                                        try:
                                            pos_value = row.get('Representative_Subsequence_Positions', None)
                                            if pd.isna(pos_value) or not isinstance(pos_value, str):
                                                rep_positions = []
                                            else:
                                                rep_positions = json.loads(pos_value)
                                        except (json.JSONDecodeError, TypeError):
                                            # 尝试使用ast.literal_eval作为备选方案
                                            try:
                                                if pd.notna(row.get('Representative_Subsequence_Positions', None)):
                                                    rep_positions = ast.literal_eval(
                                                        str(row['Representative_Subsequence_Positions']))
                                                else:
                                                    rep_positions = []
                                            except (ValueError, SyntaxError):
                                                # 如果两种方法都失败，使用空列表作为默认值
                                                logger.warning(
                                                    f"Unable to parse Representative_Subsequence_Positions, using empty list")
                                                rep_positions = []
                                        rep_subsequences = []
                                        valid_rep_positions = []
                                        # 检查每个元素，确保它们是元组或列表（长度为2）
                                        for pos in rep_positions:
                                            if isinstance(pos, (tuple, list)) and len(pos) == 2:
                                                valid_rep_positions.append(pos)
                                            elif isinstance(pos, int):
                                                # 如果是单个整数，使用整个rep_positions作为元组
                                                valid_rep_positions.append(tuple(rep_positions))
                                                break
                                        # 使用验证过的位置创建子序列
                                        for start, end in valid_rep_positions:
                                            rep_subsequences.append(full_series[start:end + 1])
                                        rep_series = '\n'.join([', '.join(map(str, seq)) for seq in rep_subsequences])
                                        if recent_series:
                                            full_prompt, images = build_pcot_prompt(id_val, full_series, recent_series,
                                                                                    rep_series,
                                                                                    memory_pool, dataset_name, method,
                                                                                    pred_len, hist_len, positions)
                                            log_file.write("\n[Full Prompt]\n")
                                            log_file.write(full_prompt)
                                            result, in_tokens, out_tokens, ttft, resp_time, total_time, cost = callLLM(
                                                full_prompt, images)
                                            
                                            # 只在LLM调用成功时才记录日志
                                            if not result.startswith("Error:"):
                                                log_file.write(f"\n{'-' * 120}\n")
                                                log_file.write("[Full Response]\n")
                                                log_file.write(result)
                                                response_file = save_full_response(id_val, result, dataset_name, method,
                                                                                   inner_task_type, hist_len, pred_len)
                                                result = process_result(result, pred_len)
                                                log_entry = {
                                                    'ID': id_val,
                                                    'Task': inner_task_type,
                                                    'Dataset': dataset_name,
                                                    'Method': method,
                                                    'TTFT': ttft,
                                                    'InputTokens': in_tokens,
                                                    'OutputTokens': out_tokens,
                                                    'Cost': cost,
                                                    'TotalTime': total_time,
                                                    'ResponseFile': response_file
                                                }
                                                save_log_entry(log_entry, inner_task_type, method)
                                                pred_labels, pred_series_str, impact_scores_str = parse_llm_output(
                                                    result, pred_len, None, inner_task_type, method)

                                                if method == "InfTS-LLM" or method == "InfTS-LLM (+v)":
                                                    try:
                                                        impact_scores = json.loads(impact_scores_str)
                                                    except:
                                                        impact_scores = []

                                                    r_scores = detection_dict.get(id_val, [])
                                                    update_memory_pool(rep_series, rep_positions, r_scores, impact_scores,
                                                                       memory_pool)

                                                true_series_tail = full_series[-pred_len:]
                                                pred_series_str = process_series(pred_series_str, true_series_tail,
                                                                                 inner_task_type)

                                                update_csv(id_val, pred_labels, pred_series_str, impact_scores_str,
                                                           dataset_name, inner_task_type, method, hist_len, pred_len)
                                                generate_predict_compare_csv(id_val, pred_labels, pred_series_str,
                                                                             impact_scores_str, dataset_name, method,
                                                                             inner_task_type,
                                                                             hist_len, pred_len)
                                                if MEMORY_STORAGE_MODE == MODE_FILE:
                                                    save_memory_state(memory_pool)

    sys.stdout = original_stdout
    log_file.close()