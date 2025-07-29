from utils.common import *
def process_ett_dataset(dataset_name):
    input_path = DATASET_PATHS[dataset_name]
    output_dir = OUTPUT_DIRS[dataset_name]
    output_path = os.path.join(output_dir, f'streaming_ts_{dataset_name}.csv')
    if not os.path.exists(input_path):
        return f"Skipped {dataset_name}: File not found"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    if 'date' in df.columns:
        df = df.iloc[:, 1:]
    for column in tqdm(df.columns, desc=f"Generating {dataset_name} plots and streaming dataset"):
        safe_name = safe_filename(column)
        clean_title = column.split('(')[0].strip() if '(' in column else column
        plt.figure(figsize=(20, 4))
        plt.plot(range(len(df)), df[column], color='#1f77b4', linewidth=1.2)
        plt.title(clean_title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe_name}.png'), dpi=150)
        plt.close()
    output_data = []
    for idx, col in enumerate(df.columns):
        output_data.append({
            'Index': idx,
            'Series': str(df[col].tolist()),
            'Labels': '[]',
            'Positions': '[]'
        })
    pd.DataFrame(output_data).to_csv(output_path, index=False)
    return f"Processed {dataset_name}: {len(df.columns)} series"