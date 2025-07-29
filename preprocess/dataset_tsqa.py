from utils.common import *
def process_tsqa_dataset():
    input_path = DATASET_PATHS["TSQA"]
    output_base = OUTPUT_DIRS["TSQA"]
    if not os.path.exists(input_path):
        return "Skipped TSQA: File not found"
    os.makedirs(output_base, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if content.startswith('<Sheet1>'):
        content = content.split('\n', 1)[1].replace('</Sheet1>', '')
    reader = csv.reader(content.splitlines())
    headers = next(reader)
    if 'Label' not in headers or 'Series' not in headers:
        raise ValueError("Missing required columns in TSQA dataset")
    label_idx = headers.index('Label')
    series_idx = headers.index('Series')
    rows = list(reader)
    random.shuffle(rows)
    label_counter = {label: 0 for label in LABELS_TO_PROCESS}
    label_series = {label: [] for label in LABELS_TO_PROCESS}
    processed_count = 0
    for row in tqdm(rows, desc="Loading TSQA dataset"):
        if len(row) <= max(label_idx, series_idx):
            continue
        label = row[label_idx].strip()
        series_str = row[series_idx].strip()
        if not label or not series_str or label not in LABELS_TO_PROCESS:
            continue
        if label_counter[label] >= K:
            continue
        try:
            if series_str.startswith('"') and series_str.endswith('"'):
                series_str = series_str[1:-1]
            series_data = ast.literal_eval(series_str)
            label_dir = os.path.join(output_base, get_label_dir_name(label))
            os.makedirs(label_dir, exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.plot(series_data, color='#1f77b4', linewidth=2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(label_dir, f'tsqa_{label}_{label_counter[label]}.png'), dpi=100)
            plt.close()
            label_series[label].append(series_data)
            label_counter[label] += 1
            processed_count += 1
        except Exception:
            continue
    combined_paths = []
    for label, series_list in tqdm(label_series.items(), desc="Generating TSQA plots"):
        if not series_list:
            continue
        n = len(series_list)
        cols = min(10, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(f'{label} (TSQA)', fontsize=16)
        for i, ax in enumerate(np.array(axes).flatten()):
            if i < n:
                ax.plot(series_list[i], color='#1f77b4', linewidth=1)
                ax.grid(alpha=0.2)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(output_base, f"{get_label_dir_name(label)}_tsqa.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        combined_paths.append(output_path)
    if CREATE_GRAND_COMBINED and combined_paths:
        create_grand_combined(combined_paths, output_base, "TSQA")
    if not CREATE_COMBINED_PER_LABEL:
        for path in combined_paths:
            try:
                os.remove(path)
            except:
                pass
    TSQA_streaming_time_series(label_series)
    return f"Processed TSQA: {processed_count} series"

def TSQA_streaming_time_series(label_data):
    os.makedirs(OUTPUT_DIRS['TSQA'], exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIRS['TSQA'], 'streamingTS_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Series', 'Labels', 'Positions'])
        for i in tqdm(range(N), desc="Generating TSQA streaming dataset"):
            long_series, labels_info, positions_info = generate_stream_series(label_data)
            plot_and_save_series(long_series, i, OUTPUT_DIRS['TSQA'])
            writer.writerow([i, str(long_series), str(labels_info), str(positions_info)])