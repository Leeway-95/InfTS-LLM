import traceback
from utils.common import get_dataset_path_info
from repdetection import *
from utils.config import *

TASK_DATASET_MAPPING = {
    "UNDERSTANDING": DATASET_UNDERSTANDING,
    "FORECASTING_NUM": DATASET_FORECASTING_NUM,
    "FORECASTING_EVENT": DATASET_FORECASTING_EVENT,
    "REASONING": DATASET_REASONING
}

def process_txt_files(input_dir, output_dir):
    """处理目录中的TXT文件"""
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not txt_files:
        return
    
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=len(txt_files), desc="Processing TXT files", 
              bar_format=bar_format, dynamic_ncols=True) as pbar:
        for file in txt_files:
            file_path = os.path.join(input_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = [float(line.strip()) for line in f]
                pbar.set_description(f"Processing {file[:15]}")
                process_series(np.array(data), file, output_dir)
                pbar.update(1)
                pbar.set_postfix(completed=f"{pbar.n}/{len(txt_files)}")
            except (ValueError, IOError) as e:
                print(f"Error processing {file}: {e}")


def process_single_dataset(task_type, dataset_name):
    """处理单个数据集的检测任务"""
    dataset_path, input_path, output_dir = get_dataset_path_info(dataset_name)
    if not dataset_path:
        print(f"Warning: Dataset {dataset_name} not found in dataset paths")
        return False
    
    input_file = os.path.join(input_path, 'stream_summary.csv')
    
    print(f"Task: {task_type}, Dataset: {dataset_name}, Input path: {input_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input path '{input_file}' does not exist")
        return False
    
    try:
        # 处理目录中的TXT文件
        if os.path.isdir(input_path):
            process_txt_files(input_path, output_dir)
        # 处理CSV文件
        if input_file.endswith('.csv') and os.path.exists(input_file):
            process_csv_file(input_file, output_dir, dataset_name)

        print()  # 添加空行分隔
        return True
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    total_processed = 0
    successful_processed = 0
    for task_type in TASK:
        if task_type in TASK_DATASET_MAPPING:
            datasets = TASK_DATASET_MAPPING[task_type]
            for dataset_name in datasets:
                total_processed += 1
                if process_single_dataset(task_type, dataset_name):
                    successful_processed += 1
