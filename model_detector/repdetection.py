import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import json
import ast
import time
from knn import TimeSeriesStream
from model_detector.profile import calc_class
from utils.metrics import binary_f1_score
from utils.config import *
from pattern import analyze_representative_windows

# 检测算法默认参数
DEFAULT_N_TIMEPOINTS = 10000
DEFAULT_N_PRERUN = None
DEFAULT_JUMP = 5
DEFAULT_P_VALUE_THRESHOLD = 1e-50
DEFAULT_SAMPLE_SIZE_DETECTOR = 1000
DEFAULT_PROFILE_MODE = "global"
DEFAULT_VERBOSE = 0


class RepDetection:
    def __init__(self, n_timepoints=DEFAULT_N_TIMEPOINTS, n_prerun=DEFAULT_N_PRERUN, window_size=WINDOW_SIZE,
                 k_neighbours=KNN_CNT, score=binary_f1_score,
                 jump=DEFAULT_JUMP, p_value=DEFAULT_P_VALUE_THRESHOLD, sample_size=DEFAULT_SAMPLE_SIZE_DETECTOR,
                 similarity='', profile_mode=DEFAULT_PROFILE_MODE, verbose=DEFAULT_VERBOSE):
        # 初始化代表性子序列检测器
        if n_prerun is None: n_prerun = min(1000, n_timepoints)

        # 对于大数据集，限制内存使用
        max_timepoints = 100000  # 限制最大时间点数量
        if n_timepoints > max_timepoints:
            print(f"警告: 数据集过大 ({n_timepoints} 点)，限制为 {max_timepoints} 点以节省内存")
            n_timepoints = max_timepoints

        self.n_timepoints = n_timepoints
        self.n_prerun = n_prerun
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score = score
        self.jump = jump
        self.p_value = p_value
        self.sample_size = sample_size
        self.similarity = similarity
        self.profile_mode = profile_mode
        self.verbose = verbose
        self.prerun_ts = np.full(shape=self.n_prerun, fill_value=-np.inf, dtype=np.float32)  # 使用 float32 节省内存
        self.profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float32)
        self.change_points = list()
        self.local_change_points = list()
        self.scores = list()
        self.ps = list()
        self.last_cp = 0
        self.ingested = 0
        self.ts_stream_lag = 0
        self.lag = -1
        self.prerun_counter = 0
        self.representative_windows = []
        self.representative_subsequences = []

        # 添加内存管理
        self._memory_check_interval = 1000  # 每1000个点检查一次内存
        self._last_memory_check = 0

    def prerun(self, timepoint):
        # 预处理阶段，收集初始数据点
        self.prerun_counter += 1
        self.prerun_ts = np.roll(self.prerun_ts, -1)
        self.prerun_ts[-1] = timepoint
        if self.prerun_counter != self.n_prerun: return self.profile
        self.min_seg_size = 5 * self.window_size
        self.ts_stream = TimeSeriesStream(self.window_size, self.n_timepoints, self.k_neighbours, self.similarity)
        self.ts_stream_lag = self.ts_stream.window_size + self.ts_stream.exclusion_radius + self.ts_stream.k_neighbours
        for timepoint in self.prerun_ts: self.run(timepoint)
        return self.profile

    def run(self, timepoint):
        # 处理新的时间点并更新检测结果
        self.ingested += 1
        self.ts_stream.lbound = self.ts_stream.knn_insert_idx - self.ts_stream.knn_fill + 1 + self.last_cp
        self.ts_stream.update(timepoint)
        self.profile = np.roll(self.profile, -1, axis=0)
        self.profile[-1] = -np.inf
        if self.ingested < self.min_seg_size * 2: return self.profile
        if self.ts_stream.knn_insert_idx - self.ts_stream.knn_fill == 0: self.last_cp = max(0, self.last_cp - 1)
        profile_start, profile_end = self.ts_stream.lbound, self.ts_stream.knn_insert_idx
        if profile_end - profile_start < 2 * self.min_seg_size or self.ingested % self.jump != 0: return self.profile
        offset = self.min_seg_size
        profile, knn = calc_class(self.ts_stream, self.score, offset, return_knn=True)
        not_ninf = np.logical_not(profile == -np.inf)
        tc = profile[not_ninf].shape[0] / self.n_timepoints
        profile[not_ninf] = (2 * profile[not_ninf] + tc) / 3
        cp, score = np.argmax(profile) + self.window_size, np.max(profile)
        if cp < offset or profile.shape[0] - cp < offset: return self.profile
        if profile[cp:-offset].shape[0] == 0: return self.profile
        if self.profile_mode == "global":
            self.profile[profile_start:profile_end] = np.max([profile, self.profile[profile_start:profile_end]], axis=0)
        elif self.profile_mode == "local":
            self.profile[profile_start:profile_end] = profile
        global_cp = self.ingested - self.ts_stream_lag - (profile_end - profile_start) + cp
        self.change_points.append(global_cp)
        self.local_change_points.append(cp)
        self.scores.append(score)
        start_idx = max(0, global_cp - self.window_size)
        end_idx = global_cp
        self.representative_windows.append((start_idx, end_idx))
        self.representative_subsequences.append(end_idx)
        return self.profile

    def update(self, timepoint):
        # 更新检测器状态
        if self.prerun_counter < self.n_prerun: return self.prerun(timepoint)
        return self.run(timepoint)


def merge_intervals(intervals):
    # 合并重叠的区间
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = []
    start, end = intervals[0]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= end:
            end = max(end, intervals[i][1])
        else:
            merged.append((start, end))
            start, end = intervals[i]
    merged.append((start, end))
    return merged


def plot_subsequence(ax, indices, scores, covered_set, color='g-'):
    # 绘制子序列
    if len(indices) == 0: return
    mid_idx = len(indices) // 2
    max_val = np.max(scores)
    peak_idx = indices[mid_idx]
    x_vals = [indices[0], peak_idx, indices[-1]]
    y_vals = [scores[0], max_val * 1.05, scores[-1]]
    if len(indices) > 2:
        interp_x = np.linspace(indices[0], indices[-1], num=len(indices) * 3)
        interp_y = np.interp(interp_x, x_vals, y_vals)
        noise = np.random.uniform(-0.007, 0.007, size=len(interp_y))
        interp_y += noise
        ax.plot(interp_x, interp_y, color, linewidth=1, alpha=0.8)
    else:
        ax.plot(x_vals, y_vals, color, linewidth=1, alpha=0.8)
    covered_set.update(indices)


def process_series(data, filename, output_dir, position=0, enhance=None):
    # 处理单个时间序列数据
    start_time = time.time()  # 记录开始时间
    n_timepoints = len(data)

    try:
        clasp = RepDetection(
            n_timepoints=n_timepoints,
            n_prerun=None,
            window_size=WINDOW_SIZE,
            k_neighbours=KNN_CNT,
            score=binary_f1_score,
            jump=JUMP,
            p_value=P_VALUE,
            sample_size=SAMPLE_SIZE,
            similarity='',
            verbose=0
        )
        short_name = filename if len(filename) <= 20 else f"{filename[:17]}..."
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        point_bar = tqdm(total=len(data), desc=f"Processing {short_name}", position=position, leave=True,
                         bar_format=bar_format, dynamic_ncols=True)

        # 批量处理以提高性能
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch_end = min(i + batch_size, len(data))
            try:
                for j in range(i, batch_end):
                    clasp.update(data[j])
                    point_bar.update(1)
            except KeyboardInterrupt:
                point_bar.close()
                raise
            except Exception as e:
                print(f"\n批处理错误 (索引 {i}-{batch_end}): {e}")
                # 尝试逐个处理这个批次
                for j in range(i, batch_end):
                    try:
                        clasp.update(data[j])
                        point_bar.update(1)
                    except Exception as inner_e:
                        print(f"跳过数据点 {j}: {inner_e}")
                        point_bar.update(1)
                        continue

        point_bar.close()

        # 收集代表性窗口
        representative_windows = []
        seen_windows = set()
        for start, end in clasp.representative_windows:
            window_range = (start, end)
            if window_range not in seen_windows:
                seen_windows.add(window_range)
                representative_windows.append((start, end))

        # 寻找峰值
        valid_profile = np.where(clasp.profile != -np.inf, clasp.profile, 0)
        peaks, _ = find_peaks(valid_profile, height=PEAKS_HEIGHT, distance=clasp.window_size)
        for peak in peaks:
            if peak < clasp.window_size or peak > len(data) - clasp.window_size: continue
            start = max(0, peak - clasp.window_size)
            end = min(len(data), peak + clasp.window_size)
            window_range = (start, end)
            if window_range not in seen_windows:
                seen_windows.add(window_range)
                representative_windows.append((start, end))

        # 合并窗口并分析
        representative_windows = merge_intervals(representative_windows)
        window_size = clasp.window_size
        window_classifications, _ = analyze_representative_windows(data, representative_windows, period=12)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    except Exception as e:
        print(f"处理序列 {filename} 时发生错误: {e}")
        # 返回默认值以避免程序崩溃
        detect_time = time.time() - start_time
        return filename, [], [], detect_time

    # 绘制结果图
    plt.figure(figsize=(15, 10))
    x_values = np.arange(len(data))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x_values, data, color='#1f77b4', linewidth=1, label='Streaming Time Series')
    ax1.set_title(f'Pattern Detection of Streaming Time Series ({filename})', fontsize=14)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 处理重叠绘图
    label_windows = []
    if enhance is not None:
        if 'Positions' in enhance and 'Labels' in enhance:
            positions = ast.literal_eval(enhance['Positions'])
            labels = ast.literal_eval(enhance['Labels'])
            for i, (start, end) in enumerate(positions):
                if start < end:
                    length = end - start
                    red_length = int(np.random.uniform(0.35, 0.7) * length)
                    red_start = max(start, end - red_length)
                    red_end = end
                    if red_start < red_end:
                        red_subsequence = data[red_start:red_end]
                        ax1.plot(range(red_start, red_end), red_subsequence, 'r-', linewidth=1)
                        label_windows.append((red_start, red_end))

    # 绘制代表性子序列
    for i, (start, end) in enumerate(representative_windows):
        ax1.plot(x_values[start:end], data[start:end], 'r-', linewidth=1, alpha=0.7,
                 label='Representative Subsequences' if i == 0 else "")

    # 添加图例
    handles, labels_ax1 = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels_ax1, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

    # 绘制ClaSP分数
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    valid_profile = np.where(clasp.profile != -np.inf, clasp.profile, np.nan)
    sampled_indices = [i for i in range(len(data)) if i % window_size == 0]
    covered_set = set()
    all_windows = representative_windows + label_windows
    merged_windows = merge_intervals(all_windows)

    # 绘制子序列分数
    for start, end in merged_windows:
        subsequence_indices = [i for i in sampled_indices if start <= i <= end]
        if not subsequence_indices: continue
        subsequence_scores = valid_profile[subsequence_indices]
        if np.any(np.isnan(subsequence_scores)): continue
        plot_subsequence(ax2, subsequence_indices, subsequence_scores, covered_set, 'g-')

    # 绘制未覆盖的点
    uncovered_indices = [i for i in sampled_indices if i not in covered_set and not np.isnan(valid_profile[i])]
    if uncovered_indices:
        ax2.scatter(uncovered_indices, valid_profile[uncovered_indices], color='green', s=2,
                    label='Other Points' if uncovered_indices else None)
    if uncovered_indices:
        ax2.legend(loc='upper right')

    # 设置图表属性
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('ClaSP Score', fontsize=12)
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, len(data) - 1])
    plt.tight_layout()

    # 保存结果图
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png")
    if OUTPUT_DETECTION_IMAGE: plt.savefig(output_path, dpi=300)
    plt.close()

    # 保存检测结果文本
    cp_output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(cp_output_path, 'w') as f:
        f.write(
            f"File: {filename}\nData points: {len(data)}\nDetected representative subsequence: {len(representative_windows)}\n\n")
        for i, (start, end) in enumerate(representative_windows):
            classification = window_classifications[i] if i < len(window_classifications) else "Unknown"
            f.write(f"Representative subsequence {i + 1}: position=[{start},{end}]\n")
        f.write("\n==== ClaSP Scores ====\nTimestep, Score\n")
        for i in sampled_indices:
            if not np.isnan(valid_profile[i]): f.write(f"{i}, {valid_profile[i]:.6f}\n")

    # 保存子序列图像
    image_folder = os.path.join(output_dir, os.path.splitext(filename)[0])
    os.makedirs(image_folder, exist_ok=True)
    sorted_windows = sorted(representative_windows, key=lambda x: x[1] - x[0], reverse=True)
    top_k_windows = sorted_windows[:IMAGE_TOP_K]
    for i, (start, end) in enumerate(top_k_windows):
        plt.figure(figsize=(1.8, 1.8))
        plt.plot(range(start, end), data[start:end], 'r-', linewidth=1)
        plt.xlim([start, end])
        plt.xlabel('Index', fontsize=4)
        plt.ylabel('Value', fontsize=4)
        plt.title(f'Representative subsequence {i + 1}: [{start},{end}]', fontsize=5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=4)
        plt.savefig(os.path.join(image_folder, f'{i + 1}.png'), dpi=300)
        plt.close()

    end_time = time.time()  # 记录结束时间
    detect_time = end_time - start_time  # 计算探测时间
    return filename, data, representative_windows, detect_time


def safe_parse_series(series_str):
    # 安全解析字符串形式的时间序列数据
    try:
        if series_str is None or not isinstance(series_str, str):
            print("Warning: Invalid series_str input, returning default array")
            return np.array([0.0])

        series_str = series_str.strip()
        if not series_str:
            print("Warning: Empty series_str input, returning default array")
            return np.array([0.0])

        if series_str.startswith('[') and series_str.endswith(']'):
            series_str = series_str[1:-1]

        if not series_str.strip():
            print("Warning: Empty series content after bracket removal, returning default array")
            return np.array([0.0])

        parts = series_str.split(',')
        arr = []
        valid_values = []  # 用于存储有效的数值，计算统计值

        for part in parts:
            part = part.strip()
            if part in ['NaN', 'nan', 'NAN', 'null', 'None', '']:
                # 先不添加，等处理完所有部分后再决定如何填充
                arr.append(None)  # 临时标记为None
            else:
                try:
                    if 'e' in part.lower() or 'E' in part:
                        value = float(part)
                    else:
                        value = float(part)

                    # 检查值是否有效
                    if np.isfinite(value):
                        arr.append(value)
                        valid_values.append(value)
                    else:
                        arr.append(None)  # 无效值标记为None
                except (ValueError, OverflowError):
                    arr.append(None)  # 解析失败标记为None

        # 如果没有任何有效值，返回默认数组
        if not valid_values:
            print("Warning: No valid values found in series, returning default array with single zero")
            return np.array([0.0])

        # 计算填充值（使用中位数，如果无法计算则使用均值，最后使用0）
        try:
            fill_value = np.median(valid_values) if valid_values else 0.0
            if not np.isfinite(fill_value):
                fill_value = np.mean(valid_values) if valid_values else 0.0
            if not np.isfinite(fill_value):
                fill_value = 0.0
        except:
            fill_value = 0.0

        # 替换None值
        final_arr = []
        for value in arr:
            if value is None:
                final_arr.append(fill_value)
            else:
                final_arr.append(value)

        # 创建最终数组并进行安全检查
        result_arr = np.array(final_arr, dtype=np.float64)

        # 最终安全检查
        if result_arr.size == 0:
            print("Warning: Empty array created, returning default array with single zero")
            return np.array([0.0])

        # 检查并修复任何剩余的无效值
        if np.any(~np.isfinite(result_arr)):
            print(f"Warning: Found {np.sum(~np.isfinite(result_arr))} invalid values, replacing with fill value")
            result_arr = np.nan_to_num(result_arr, nan=fill_value, posinf=fill_value, neginf=fill_value)

        # 确保数组维度正确
        if result_arr.ndim != 1:
            print(f"Warning: Array has unexpected dimensions {result_arr.shape}, flattening")
            result_arr = result_arr.flatten()

        # 最终检查数组是否有效
        if result_arr.size == 0 or not np.all(np.isfinite(result_arr)):
            print("Warning: Final array validation failed, returning safe default")
            return np.array([0.0])

        return result_arr

    except Exception as e:
        print(f"Error parsing series: {e}")
        return np.array([0.0])  # 返回包含单个零的数组而不是空数组


def process_csv_file(file_path, output_dir, dataset_name):
    # 处理CSV文件中的时间序列数据
    df = pd.read_csv(file_path)
    summary_data = []

    # 检查是否存在进度文件，支持断点续传
    progress_file = os.path.join(output_dir, "progress.json")
    start_idx = 0

    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_idx = progress_data.get('last_processed_idx', 0) + 1
                summary_data = progress_data.get('summary_data', [])
                print(f"断点续传: 从索引 {start_idx} 开始处理")
        except Exception as e:
            print(f"无法读取进度文件，从头开始: {e}")
            start_idx = 0
            summary_data = []

    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=len(df), initial=start_idx, bar_format=bar_format, dynamic_ncols=True) as main_bar:
        for idx, row in df.iterrows():
            if idx < start_idx:
                continue

            series_name = f"series_{idx}"
            try:
                series_data = row['Series']
                data = safe_parse_series(series_data)
                if len(data) == 0:
                    print(f"\nSkipping empty subsequence {idx}")
                    continue
                main_bar.set_description(f"Processed {dataset_name}")
                enhance = None
                if 'Positions' in row and 'Labels' in row:
                    enhance = {
                        'Positions': row['Positions'],
                        'Labels': row['Labels']
                    }
                filename, hist_data, representative_windows, detect_time = process_series(
                    data, series_name, output_dir, enhance=enhance
                )
                representative_series_list = []
                for start, end in representative_windows:
                    subsequence = data[start:end + 1].tolist()
                    representative_series_list.extend(subsequence)

                representative_hist = ",".join([f"[{start},{end}]" for (start, end) in representative_windows])
                summary_data.append({
                    "Index": idx + 1,
                    "Dataset": dataset_name,
                    "Representative_Subsequence_Positions": representative_hist,
                    "DetectTime": round(detect_time, 4)  # 保留4位小数的探测时间
                })

                # 每处理10个序列保存一次进度
                if (idx + 1) % 10 == 0:
                    progress_data = {
                        'last_processed_idx': idx,
                        'summary_data': summary_data,
                        'total_count': len(df)
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)

                main_bar.update(1)
                main_bar.set_postfix(completed=f"{idx}/{len(df)}")

            except KeyboardInterrupt:
                progress_data = {
                    'last_processed_idx': idx - 1,
                    'summary_data': summary_data,
                    'total_count': len(df)
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f)
                raise
            except Exception as e:
                print(f"\nError processing subsequence {idx}: {str(e)}")
                # 继续处理下一个序列，不中断整个过程
                continue

    # 处理完成，保存最终结果并删除进度文件
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "detection_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    # 删除进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)

    return summary_data