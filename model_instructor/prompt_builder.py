from utils.common import *
import os
import pandas as pd
import ast
import logging
import json

logger = logging.getLogger(__name__)


def load_questions(dataset_name):
    try:
        dataset_path = DATASET_MERGE_PATHS[dataset_name]
        dirname = os.path.dirname(dataset_path)
        filename = os.path.splitext(os.path.basename(dataset_path))[0]
        input_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")

        if not os.path.exists(input_path):
            logger.warning(f"Questions file not found at {input_path}")
            return [], []

        df = pd.read_csv(input_path)

        # 筛选出对应数据集的数据
        dataset_rows = df[df['Dataset'] == dataset_name]

        if dataset_rows.empty:
            logger.info(f"Available datasets in {dataset_name}: {df['Dataset'].unique().tolist()}")
            return [], []

        # 获取第一行的Question和Positions数据
        first_row = dataset_rows.iloc[0]
        questions_str = first_row['Question']
        positions_str = first_row['Positions']

        # 解析Question列（应该是一个数组字符串）
        try:
            if isinstance(questions_str, str):
                # 尝试使用ast.literal_eval解析
                questions_list = ast.literal_eval(questions_str)
            else:
                questions_list = [questions_str] if questions_str else []
        except (ValueError, SyntaxError):
            # 如果解析失败，将整个字符串作为单个问题
            questions_list = [questions_str] if questions_str else []

        # 解析Positions列
        try:
            if isinstance(positions_str, str):
                positions_list = ast.literal_eval(positions_str)
            else:
                positions_list = []
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse positions for dataset {dataset_name}")
            positions_list = []

        return questions_list, positions_list

    except Exception as e:
        logger.error(f"Error loading QATS-4 questions and positions: {e}")
        return [], []


def format_qats4_questions(questions, positions, full_series):
    if not questions:
        return ""

    formatted_questions = []
    for i, question in enumerate(questions, 1):
        # 清理问题文本，移除多余的引号和转义字符
        clean_question = question.strip().strip('"\'')

        # 获取对应的时间序列范围
        if i <= len(positions):
            position = positions[i - 1]  # positions是0索引的
            if isinstance(position, (list, tuple)) and len(position) == 2:
                clean_question = clean_question.replace("<<Time_Series>>", str(full_series[position[0]:position[1]]))
            else:
                clean_question = clean_question.replace("<<Time_Series>>", str(full_series[position[0]:]))

        formatted_questions.append(f"{i}. {clean_question}")

    return "\n\n".join(formatted_questions)


def get_forecasting_event_prompts(dataset_name, window_size=24, positions=None, id_val=None):
    domain_label = "True/False"
    dataset_instruction = ""
    if dataset_name.startswith("Weather_"):
        # Extract city name from dataset name
        domain_label = '"not rained", "rained"'
        city_name = dataset_name.split("_")[1] if "_" in dataset_name else "the location"
        dataset_instruction = (
            f"Your job is to act as a professional weather forecaster. You will be given a time-series data of the weather from the past 24 hours. Based on this information, your task is to predict whether it will rain in the next 24 hours. "
            f"Your task is to predict whether it will rain or not in {city_name} in the next {window_size} hours. Review the time-series data provided for the last {window_size} hours. Each time-series consists of hourly values separated by a '|' token for the following indicators:"
            f"- Temperature (Kelvin): {{var_1}}"
            f"- Humidity (%): {{var_2}}"
            f"- Air Pressure (hPa): {{var_3}}"
            f"- Wind Speed (m/s): {{var_4}}"
            f"- Wind Direction (degrees): {{var_5}}"
            f"Based on this information, respond with either 'rained' or 'not rained'. Do not provide any other details.\n"
            f"Dataset-specific Instructions:\n- Weather datasets: Predict weather events (e.g., rain/no rain) based on meteorological time series patterns. Analyze temperature, humidity, pressure, and other weather indicators to determine the likelihood of precipitation or other weather phenomena.")

    elif dataset_name.startswith("Finance_"):
        domain_label = '"decrease", "increase", "neutral"'
        # Extract indicator name from dataset name
        indicator_mapping = {
            "Finance_sp500": "S&P 500",
            "Finance_nikkei": "Nikkei 225"
        }
        indicator_name = indicator_mapping.get(dataset_name, "financial indicator")

        dataset_instruction = f"Your job is to act as a professional financial forecaster. You will be given a time-series data from the past 20 market days. Based on this information, your task is to predict whether the {indicator_name} price will decrease by more than 1%, increase by more than 1%, or change minimally in the next market day."

        dataset_instruction += f"""Your task is to predict whether the {indicator_name} price will: (1) Decrease: decrease by more than 1% (2) Increase: increase by more than 1% (3) Neutral: change minimally, between -1% to 1%
        in the next market day. Review the time-series data provided for the last {window_size} market days. Each time-series consists of daily values separated by a '|' token for the following indicators:
        - S&P 500: {{var_1}}
        - VIX (Volatility Index): {{var_2}}
        - Nikkei 225: {{var_3}}
        - FTSE 100: {{var_4}}
        - Gold Futures: {{var_5}}
        - Crude Oil Futures: {{var_6}}
        - Exchange rate for EUR/USD: {{var_7}}
        - Exchange rate for USD/JYP: {{var_8}}
        - Exchange rate for USD/CNY: {{var_9}}
        Based on this information, predict whether the {indicator_name} price will decrease by more than 1%, increase by more than 1%, or otherwise, in the next market day. Respond with either 'decrease', 'increase', or 'neutral'. Do not provide any other details.\n
        Dataset-specific Instructions:\n- Finance datasets: Predict financial market events (e.g., significant price movements, market volatility) based on market indicators. Analyze price trends, volume patterns, and market dynamics to identify potential market events or anomalies."""

    elif dataset_name.startswith("Healthcare_"):
        domain_label = '"did not exceed the average", "exceeded the average"'
        if "mortality" in dataset_name.lower():
            dataset_instruction = "Your job is to act as a professional healthcare forecaster. You will be given a time-series data from the past 20 weeks. Based on this information, your task is to predict whether the ratio of mortality from Influenza or Pneumonia to the total number of deaths will exceed its average in the coming week."

            dataset_instruction += f"""Your task is to predict whether the ratio of mortality from Influenza or Pneumonia to the total number of deaths will: (1) Exceed its average (2) Not exceed its average in the coming week. Review the time-series data provided for the last {window_size} weeks. Each time-series consists of weekly values separated by a '|' token for the following indicators:
            - Total deaths: {{var_1}}
            - Influenza/Pneumonia deaths: {{var_2}}/{{var_3}}
            - Mortality ratio (%): {{var_4}}
            Based on this time-series data, predict whether the mortality ratio will exceed its average or not in the coming week. Respond with either 'exceeded the average' or 'did not exceed the average'. Do not provide any other details."""

        else:  # Healthcare_positive
            dataset_instruction = "Your job is to act as a professional healthcare forecaster. You will be given a time-series data from the past 20 weeks. Based on this information, your task is to predict whether the percentage of respiratory specimens testing positive for influenza will exceed its average of 6.26% in the coming week."

            dataset_instruction += f"""Your task is to predict whether the percentage of respiratory specimens testing positive for influenza will: (1) Exceed its average of 6.26% (2) Not exceed its average of 6.26% in the coming week. Review the time-series data provided for the last {window_size} weeks. Each time-series consists of weekly values separated by a '|' token for the following indicators:
            - Number of specimens tested: {{var_1}}
            - Number of positive specimens for Influenza A: {{var_2}}
            - Number of positive specimens for Influenza B: {{var_3}}
            - Ratio of positive specimens (%): {{var_4}}
            - Ratio of positive specimens for Influenza A (%): {{var_5}}
            - Ratio of positive specimens for Influenza B (%): {{var_6}}
            Based on this time-series data, predict whether the percentage of respiratory specimens testing positive for influenza will exceed its average of 6.26% or not in the coming week. Respond with either 'exceeded the average' or 'not did not exceed the average'. Do not provide any other details."""
        dataset_instruction += "Dataset-specific Instructions:\n- Healthcare datasets: Predict healthcare outcomes (e.g., patient mortality, positive test results) based on patient monitoring data patterns. Analyze vital signs, laboratory values, and clinical indicators to assess patient risk and predict health events."

    # 去掉所有缩进
    dataset_instruction = dataset_instruction.replace("\n            ", "\n").replace("\n        ", "\n")

    # 加载对应数据集数据的历史Series填充到dataset_instruction的{{var_*}}中
    # 从DATASET_PATHS或DATASET_MERGE_PATHS中读取路径信息
    if dataset_name in DATASET_PATHS:
        dataset_path = DATASET_PATHS[dataset_name] + ".csv"
    elif dataset_name in DATASET_MERGE_PATHS:
        dataset_path = DATASET_MERGE_PATHS[dataset_name] + ".csv"
    else:
        raise KeyError(f"Dataset path not found for {dataset_name} in either DATASET_PATHS or DATASET_MERGE_PATHS")

    dirname = os.path.dirname(dataset_path)
    filename = os.path.splitext(os.path.basename(dataset_path))[0]
    stream_summary_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")

    if os.path.exists(stream_summary_path):
        df = pd.read_csv(stream_summary_path)
        dataset_rows = df[df['Dataset'] == dataset_name]

        if not dataset_rows.empty:
            processed_instruction = dataset_instruction

            # 收集所有行的变量数据，建立变量到数据的映射
            variable_data_map = {}

            # 遍历数据集的所有行来收集变量数据
            for row_idx, row in dataset_rows.iterrows():
                # 获取当前行的Positions数据
                positions_data = []
                if 'Positions' in row and row['Positions']:
                    try:
                        positions_data = ast.literal_eval(row['Positions'])
                    except (ValueError, SyntaxError):
                        logger.warning(f"Failed to parse positions for dataset {dataset_name}, row {row_idx}")

                # 获取当前行的Variable数据
                row_variables = []
                if 'Variable' in row and row['Variable']:
                    try:
                        row_variables = ast.literal_eval(row['Variable']) if isinstance(row['Variable'], str) else row[
                            'Variable']
                        if not isinstance(row_variables, list):
                            row_variables = [row_variables]
                    except (ValueError, SyntaxError):
                        row_variables = [row['Variable']]

                # 获取当前行的Series数据
                # 安全解析Series数据，处理NaN值
                if 'Series' in row and row['Series'] and str(row['Series']).strip():
                    try:
                        # 首先尝试使用json.loads解析
                        row_series_data = json.loads(row['Series'])
                    except json.JSONDecodeError:
                        try:
                            # 如果json.loads失败，尝试使用ast.literal_eval解析
                            row_series_data = ast.literal_eval(row['Series'])
                        except (ValueError, SyntaxError):
                            logger.warning(
                                f"Unable to parse Series field in prompt_builder: {str(row['Series'])[:50]}...")
                            # 如果包含NaN值，尝试手动处理
                            try:
                                import numpy as np
                                # 将NaN替换为None或0，然后解析
                                series_str = str(row['Series']).replace('nan', 'null')
                                row_series_data = json.loads(series_str)
                                # 将null转换为0或适当的数值
                                row_series_data = [0 if x is None else x for x in row_series_data]
                            except:
                                logger.warning(f"Failed to parse Series data in prompt_builder, using empty list")
                                row_series_data = []
                else:
                    row_series_data = []

                # 为每个变量建立数据映射
                for var_idx, var_name in enumerate(row_variables, 1):
                    # 修复：使用Positions中最大的范围截取有效数据
                    if positions_data and len(positions_data) > 0:
                        # 找到所有位置区间中的最大范围
                        max_start = float('inf')
                        max_end = 0

                        for pos_range in positions_data:
                            if isinstance(pos_range, (list, tuple)) and len(pos_range) >= 2:
                                start, end = pos_range[0], pos_range[1]
                                max_start = min(max_start, start)
                                max_end = max(max_end, end)

                        # 如果找到了有效的范围，使用最大范围截取数据
                        if max_start != float('inf') and max_end > max_start:
                            # 确保索引在有效范围内
                            max_start = max(0, max_start)
                            max_end = min(len(row_series_data), max_end)
                            series_str = '|'.join(map(str, row_series_data[max_start:max_end]))
                        else:
                            # 如果位置范围不完整，使用全部数据
                            series_str = '|'.join(map(str, row_series_data))
                    else:
                        # 如果没有positions_data，使用整个序列
                        series_str = '|'.join(map(str, row_series_data))

                    # 将变量数据存储到映射中，使用变量名作为key
                    if var_name not in variable_data_map:
                        variable_data_map[var_name] = series_str

            # 替换所有{{var_*}}占位符
            # 按照var_1, var_2, var_3...的顺序进行替换
            for var_idx in range(1, 10):  # 支持最多9个变量
                var_placeholder = f"{{var_{var_idx}}}"
                if var_placeholder in processed_instruction:
                    # 查找对应的变量名（通常是var_1, var_2等）
                    var_name = f"var_{var_idx}"
                    if var_name in variable_data_map:
                        processed_instruction = processed_instruction.replace(var_placeholder,
                                                                              variable_data_map[var_name])
                    else:
                        # 如果没有找到对应的变量，尝试使用第一个可用的变量数据
                        if variable_data_map:
                            first_available_data = list(variable_data_map.values())[0]
                            processed_instruction = processed_instruction.replace(var_placeholder, first_available_data)

            dataset_instruction = processed_instruction

    # 构建完整的数据集特定指令
    dataset_instruction += """TASK EXECUTION:
- Act according to the system prompt role and expertise
- Follow the user prompt template structure for analysis
- Provide predictions in the exact format specified\n"""

    return dataset_instruction, domain_label


def build_pcot_prompt(id_val, full_series, recent_series, rep_series, memory_pool, dataset_name, method, pred_len,
                      hist_len, positions):

    # 根据数据集名称确定任务类型
    task_type = None
    for task in TASK:
        if task == "UNDERSTANDING" and dataset_name in DATASET_UNDERSTANDING:
            task_type = "UNDERSTANDING"
            break
        elif task == "REASONING" and dataset_name in DATASET_REASONING:
            task_type = "REASONING"
            break
        elif task == "FORECASTING_NUM" and dataset_name in DATASET_FORECASTING_NUM:
            task_type = "FORECASTING_NUM"
            break
        elif task == "FORECASTING_EVENT" and dataset_name in DATASET_FORECASTING_EVENT:
            task_type = "FORECASTING_EVENT"
            break

    # 根据当前方法和任务类型选择对应的提示文件
    cot_file = ""
    pcot_input = ""
    memory_patches = []
    if method == "InfTS-LLM" or method == "InfTS-LLM (+v)":
        # 限制代表性序列的数量
        rep_lines = rep_series.split('\n')[:MEM_TOP_K] if rep_series else []
        formatted_reps = []

        pcot_input = f"Your role is a time series analysis expert.\n"

        # 格式化代表性序列
        for i, line in enumerate(rep_lines):
            if line.strip():
                formatted_reps.append(f"R_{i + 1}: {line}")

        pcot_input += "INPUT:"
        if task_type == "FORECASTING_NUM":
            pcot_input += "\n- Recent Subsequence (as the primary basis): " + str(recent_series)
        elif task_type == "UNDERSTANDING":
            for i, (start, end) in enumerate(positions):
                # 使用切片截取，注意Python切片是左闭右开，所以end+1
                subSequence = full_series[start:end + 1]
                pcot_input += f"\nSubSequence {i + 1}: {subSequence}"

        # 获取内存池中的记忆项
        memory_patches = memory_pool.get_memory_patches()
        # 根据任务类型选择对应的提示文件
        if task_type == "UNDERSTANDING":
            cot_file = Prompt_PATHS["InfTS-LLM-Understand"]
        elif task_type == "REASONING":
            cot_file = Prompt_PATHS["InfTS-LLM-Reason"]
        elif task_type == "FORECASTING_NUM":
            cot_file = Prompt_PATHS["InfTS-LLM-Forecast"]
        elif task_type == "FORECASTING_EVENT":
            cot_file = Prompt_PATHS["InfTS-LLM-Forecast-Event"]
        else:
            cot_file = Prompt_PATHS["InfTS-LLM-Forecast"]
            logger.warning(f"Unknown task type for dataset {dataset_name}, using default PCoT_Forecast.txt")
    elif method == "PromptCast":
        cot_file = Prompt_PATHS["PromptCast"]
    elif method == "TimeCP":
        cot_file = Prompt_PATHS["TimeCP"]
    elif method == "TimeCAP":
        cot_file = Prompt_PATHS["TimeCAP"]
    elif method == "Inf-LLM" or method == "Inf-LLM (+v)":
        if task_type == "UNDERSTANDING":
            cot_file = Prompt_PATHS["Inf-LLM-Understand"]
        elif task_type == "REASONING":
            cot_file = Prompt_PATHS["Inf-LLM-Reason"]
    elif method == "Window" or method == "Window (+v)":
        if task_type == "UNDERSTANDING":
            cot_file = Prompt_PATHS["Window-Understand"]
        elif task_type == "REASONING":
            cot_file = Prompt_PATHS["Window-Reason"]
    else:
        # 如果方法不在列表中，默认使用PCoT_Forecast.txt
        cot_file = Prompt_PATHS["InfTS-LLM-Forecast"]
        logger.warning(f"Unknown method {method}, using default PCoT_Forecast.txt")

    # 读取并添加思维链模板
    # logger.info(f"Using prompt template: {cot_file}")  # 减少控制台输出
    with open(cot_file, "r", encoding="utf-8") as f:
        content = f.read()
        if "<<Domain>>" in content:
            if dataset_name == "Gold":
                domain = "financial"
            elif dataset_name.startswith("ETTm"):
                domain = "electricity"
            elif dataset_name == "Weather":
                domain = "weather"
            else:
                domain = "data"
            content = content.replace("<<Domain>>", domain)
        if "<<Questions>>" in content:
            # 加载QATS-4问题和位置信息
            qats4_questions, qats4_positions = load_questions(dataset_name)
            formatted_questions = format_qats4_questions(qats4_questions, qats4_positions, full_series)
            content = content.replace("<<Questions>>", formatted_questions)
        if "<<PreLen>>" in content:
            content = content.replace("<<PreLen>>", str(pred_len))
        if "<<Histlen>>" in content:
            content = content.replace("<<Histlen>>", str(hist_len))
        if "<<Time_Series>>" in content:
            content = content.replace("<<Time_Series>>", str(recent_series))
        if "<<Full_Series>>" in content:
            subSequences = ""
            for i, (start, end) in enumerate(positions):
                # 使用切片截取，注意Python切片是左闭右开，所以end+1
                subSequence = full_series[start:end + 1]
                subSequences += f"\nSubSequence {i + 1}: {subSequence}"
            content = content.replace("<<Fulllen>>", str(subSequences))
        if "<<Fulllen>>" in content:
            content = content.replace("<<Fulllen>>", str(len(full_series)))
        if "<<Positions>>" in content:
            content = content.replace("<<Positions>>", str(positions))
        if "<<Poslen>>" in content:
            content = content.replace("<<Poslen>>", str(len(positions)))
        if "<<MLen>>" in content:
            content = content.replace("<<MLen>>", str(len(memory_patches)))
        if "<<RSLen>>" in content:
            content = content.replace("<<RSLen>>", str(len(rep_series)))

        # 处理事件预测的数据集特定指令
        if "<<DATASET_SPECIFIC_INSTRUCTION>>" in content:
            # 使用新的函数获取数据集特定的系统和用户提示
            if task_type == "FORECASTING_EVENT":
                dataset_instruction, domain_label = get_forecasting_event_prompts(dataset_name, hist_len, positions,
                                                                                  id_val)
                content = content.replace("<<DATASET_SPECIFIC_INSTRUCTION>>", dataset_instruction)
                # 确保 Domain_Label 被正确替换为实际的标签值
                if "<<Domain_Label>>" in content:
                    content = content.replace("<<Domain_Label>>", str(domain_label))

        if method == "PromptCast":
            # PromptCast特定替换
            content = content.replace("<<Start_Time>>", str(id_val))
            content = content.replace("<<End_Time>>", str(id_val + hist_len))
            if dataset_name == "Gold":
                dataset_variable = "gold price (USD per ounce)"
            elif dataset_name.startswith("ETTm"):
                dataset_variable = "electricity (MW)"
            elif dataset_name == "Weather":
                dataset_variable = "weather (percent)"
            else:
                dataset_variable = "value"
            content = content.replace("<<dataset_variable>>", dataset_variable)
        elif method == "TimeCP":
            if dataset_name == "Gold":
                domain_variable = "gold price"
                prediction_type = "price change"
            elif dataset_name.startswith("ETTm"):
                domain_variable = "electricity"
                prediction_type = "change in electricity"
            elif dataset_name == "Weather":
                domain_variable = "weather"
                prediction_type = "whether it will rain"
            else:
                domain_variable = "value"
                prediction_type = "value"
            content = content.replace("<<domain_variable>>", domain_variable)
            content = content.replace("<<prediction_type>>", prediction_type)
        elif method == "Inf-LLM" or method == "Inf-LLM (+v)":
            memory_patches = memory_pool.get_memory_patches()
            # 将列表转换为字符串，每个项目占一行
            memory_patches_str = "\n".join(memory_patches) if memory_patches else ""
            content = content.replace("<<Memory_Patches>>", memory_patches_str)
        pcot_input += content

    # 从DATASET_PATHS或DATASET_MERGE_PATHS中读取路径信息
    if dataset_name in DATASET_PATHS:
        dataset_path = DATASET_PATHS[dataset_name] + ".csv"
    elif dataset_name in DATASET_MERGE_PATHS:
        dataset_path = DATASET_MERGE_PATHS[dataset_name] + ".csv"
    else:
        raise KeyError(f"Dataset path not found for {dataset_name} in either DATASET_PATHS or DATASET_MERGE_PATHS")
    dirname = os.path.dirname(dataset_path)
    filename = os.path.splitext(os.path.basename(dataset_path))[0]

    # 构建图像目录路径
    image_dir = os.path.join(dirname, f"detection-{filename}/series_{id_val}")
    images = []

    # 只有当方法包含(+v)时才收集图像
    if has_vision_support(method):
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.endswith(".png"):
                    file_path = os.path.join(image_dir, filename)
                    images.append(file_path)

    return pcot_input, images