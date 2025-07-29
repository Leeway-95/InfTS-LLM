from utils.common import *
def process_weather_dataset():
    input_path = DATASET_PATHS["weather"]
    output_dir = OUTPUT_DIRS["weather"]
    output_csv = os.path.join(output_dir, 'streaming_ts_weather.csv')
    if not os.path.exists(input_path):
        return "Skipped weather: File not found"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    df = df.iloc[:, 1:]
    for column in tqdm(df.columns, desc="Generating weather plots"):
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
    pd.DataFrame(output_data).to_csv(output_csv, index=False)
    return f"Processed weather: {len(df.columns)} series"