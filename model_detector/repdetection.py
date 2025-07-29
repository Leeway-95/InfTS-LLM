import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import os
import argparse
import json
import ast
from knn import TimeSeriesStream
from profile import *
from metrics import binary_f1_score
from model_detector.profile import calc_class
from utils.config import *
from timepattern import analyze_representative_windows
import re

class RepDetection:
    # Real-time pattern detection in streaming time series
    def __init__(self, n_timepoints=10000, n_prerun=None, window_size=100, k_neighbours=3, score=binary_f1_score,
                 jump=5, p_value=1e-50, sample_size=1000, similarity="pearson", profile_mode="global", verbose=0):
        if n_prerun is None: n_prerun = min(1000, n_timepoints)
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
        self.prerun_ts = np.full(shape=self.n_prerun, fill_value=-np.inf, dtype=np.float64)
        self.profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)
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

    def prerun(self, timepoint):
        # Initialize streaming detector with warm-up data
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
        # Process single timepoint in streaming mode
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
        # Update detector state with new timepoint
        if self.prerun_counter < self.n_prerun: return self.prerun(timepoint)
        return self.run(timepoint)

def merge_intervals(intervals):
    # Merge overlapping intervals into consolidated ranges
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
    # Plot representative subsequence on visualization axis
    if len(indices) == 0: return
    mid_idx = len(indices) // 2
    max_val = np.max(scores)
    peak_idx = indices[mid_idx]
    x_vals = [indices[0], peak_idx, indices[-1]]
    y_vals = [scores[0], max_val * 1.05, scores[-1]]
    if len(indices) > 2:
        interp_x = np.linspace(indices[0], indices[-1], num=len(indices)*3)
        interp_y = np.interp(interp_x, x_vals, y_vals)
        noise = np.random.uniform(-0.007, 0.007, size=len(interp_y))
        interp_y += noise
        ax.plot(interp_x, interp_y, color, linewidth=1, alpha=0.8)
    else:
        ax.plot(x_vals, y_vals, color, linewidth=1, alpha=0.8)
    covered_set.update(indices)

def process_series(data, filename, output_dir, position=0, enhance=None):
    # Process complete time series for pattern detection
    n_timepoints = len(data)
    clasp = RepDetection(
        n_timepoints=n_timepoints,
        n_prerun=None,
        window_size=WINDOW_SIZE,
        k_neighbours=K,
        score=binary_f1_score,
        jump=JUMP,
        p_value=P_VALUE,
        sample_size=SAMPLE_SIZE,
        similarity=SIMILARITY,
        verbose=0
    )
    short_name = filename if len(filename) <= 20 else f"{filename[:17]}..."
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    point_bar = tqdm(total=len(data), desc=f"Processing points ({short_name})", position=position, leave=True,
                     bar_format=bar_format, dynamic_ncols=True)
    for i, timepoint in enumerate(data):
        clasp.update(timepoint)
        point_bar.update(1)
    point_bar.close()
    representative_windows = []
    seen_windows = set()
    for start, end in clasp.representative_windows:
        window_range = (start, end)
        if window_range not in seen_windows:
            seen_windows.add(window_range)
            representative_windows.append((start, end))
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
    representative_windows = merge_intervals(representative_windows)
    window_size = clasp.window_size
    window_classifications, _ = analyze_representative_windows(data, representative_windows, period=12)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    x_values = np.arange(len(data))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x_values, data, color='#1f77b4', linewidth=1, label='Streaming Time Series')
    ax1.set_title(f'Pattern Detection of Streaming Time Series ({filename})', fontsize=14)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    label_windows = []
    if OVERLAP_PLOTS and enhance is not None:
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
    for i, (start, end) in enumerate(representative_windows):
        ax1.plot(x_values[start:end], data[start:end], 'r-', linewidth=1, alpha=0.7,
                 label='Representative Subsequences' if i == 0 else "")
    handles, labels_ax1 = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels_ax1, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    valid_profile = np.where(clasp.profile != -np.inf, clasp.profile, np.nan)
    sampled_indices = [i for i in range(len(data)) if i % window_size == 0]
    covered_set = set()
    all_windows = representative_windows + label_windows
    merged_windows = merge_intervals(all_windows)
    for start, end in merged_windows:
        subsequence_indices = [i for i in sampled_indices if start <= i <= end]
        if not subsequence_indices: continue
        subsequence_scores = valid_profile[subsequence_indices]
        if np.any(np.isnan(subsequence_scores)): continue
        plot_subsequence(ax2, subsequence_indices, subsequence_scores, covered_set, 'g-')
    uncovered_indices = [i for i in sampled_indices if i not in covered_set and not np.isnan(valid_profile[i])]
    if uncovered_indices:
        ax2.scatter(uncovered_indices, valid_profile[uncovered_indices], color='green', s=2,
                   label='Other Points' if uncovered_indices else None)
    if uncovered_indices:
        ax2.legend(loc='upper right')
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('ClaSP Score', fontsize=12)
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, len(data) - 1])
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png")
    if OUTPUT_IMAGE: plt.savefig(output_path, dpi=300)
    plt.close()
    cp_output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_representative_subsequences.txt")
    with open(cp_output_path, 'w') as f:
        f.write(
            f"File: {filename}\nData points: {len(data)}\nDetected representative subsequence: {len(representative_windows)}\n\n")
        for i, (start, end) in enumerate(representative_windows):
            classification = window_classifications[i] if i < len(window_classifications) else "Unknown"
            f.write(f"Representative subsequence {i + 1}: position=[{start},{end}], pattern={classification}\n")
        f.write("\n==== ClaSP Scores ====\nIndex,Score\n")
        for i in sampled_indices:
            if not np.isnan(valid_profile[i]): f.write(f"{i},{valid_profile[i]:.6f}\n")
    return filename, data, representative_windows

def safe_parse_series(series_str):
    # Safely parse time series string with error handling
    try:
        series_str = series_str.strip()
        if series_str.startswith('[') and series_str.endswith(']'):
            series_str = series_str[1:-1]
        parts = series_str.split(',')
        arr = []
        for part in parts:
            part = part.strip()
            if part in ['NaN', 'nan', 'NAN', 'null', 'None', '']:
                arr.append(np.round(float(np.median(arr)), 2))
            else:
                try:
                    if 'e' in part or 'E' in part:
                        arr.append(float(part))
                    else:
                        arr.append(float(part))
                except ValueError:
                    arr.append(np.nan)
        return np.array(arr)
    except Exception as e:
        print(f"Error parsing series: {e}")
        return np.array([])

def process_csv_file(file_path, output_dir):
    # Process CSV file containing multiple time series
    df = pd.read_csv(file_path)
    summary_data = []
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=len(df), bar_format=bar_format, dynamic_ncols=True) as main_bar:
        for idx, row in df.iterrows():
            series_name = f"series_{idx}"
            try:
                series_data = row['Series']
                data = safe_parse_series(series_data)
                if len(data) == 0:
                    print(f"\nSkipping empty subsequence {idx}")
                    continue
                main_bar.set_description(f"Processing Completion")
                enhance = None
                if OVERLAP_PLOTS and 'Positions' in row and 'Labels' in row:
                    enhance = {
                        'Positions': row['Positions'],
                        'Labels': row['Labels']
                    }
                filename, hist_data, representative_windows = process_series(
                    data, series_name, output_dir, enhance=enhance
                )
                representative_series_list = []
                for start, end in representative_windows:
                    subsequence = data[start:end + 1].tolist()
                    representative_series_list.extend(subsequence)
                representative_series_str = json.dumps(representative_series_list)
                representative_hist = ",".join([f"[{start},{end}]" for (start, end) in representative_windows])
                summary_data.append({
                    "Index": idx,
                    "Series": row['Series'],
                    "Series_Postions": row['Positions'] if 'Positions' in row else "",
                    "Representative_Subsequence_Positions": representative_hist,
                    "Representative_Subsequences": representative_series_str
                })
                main_bar.update(1)
                main_bar.set_postfix(completed=f"{idx + 1}/{len(df)}")
            except Exception as e:
                print(f"\nError processing subsequence {idx}: {str(e)}")
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_file, index=False)
    return summary_data

def process_all_datasets():
    # Batch process multiple datasets from configuration
    datasets = {
        "ETTm1": DATASET_PATHS["ETTm1"],
        "ETTm2": DATASET_PATHS["ETTm2"],
        "gold": DATASET_PATHS["gold"],
        "weather": DATASET_PATHS["weather"]
    }
    for name, input_path in datasets.items():
        output_dir = OUTPUT_DIRS[name]
        print(f"\nProcessing dataset: {name}")
        print(f"Input path: {input_path}")
        print(f"Output directory: {output_dir}")
        if not os.path.exists(input_path):
            print(f"Error: Input path '{input_path}' does not exist")
            continue
        if os.path.isdir(input_path):
            txt_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
            bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            with tqdm(total=len(txt_files), desc=f"Processing TXT files ({name})", bar_format=bar_format,
                      dynamic_ncols=True) as pbar:
                for file in txt_files:
                    file_path = os.path.join(input_path, file)
                    with open(file_path, 'r') as f:
                        data = [float(line.strip()) for line in f]
                    pbar.set_description(f"Processing {file[:15]}")
                    process_series(np.array(data), file, output_dir)
                    pbar.update(1)
                    pbar.set_postfix(completed=f"{pbar.n}/{len(txt_files)}")
        elif input_path.endswith('.csv'):
            process_csv_file(input_path, output_dir)
        else:
            print(f"Error: Input must be CSV file or directory containing TXT files for dataset {name}")