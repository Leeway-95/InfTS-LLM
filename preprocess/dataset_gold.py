from utils.common import *
def process_gold_dataset():
    input_path = DATASET_PATHS["gold"]
    output_dir = OUTPUT_DIRS["gold"]
    if not os.path.exists(input_path):
        return "Skipped gold: File not found"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    if 'Time' in df.columns:
        time_series = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
    else:
        time_series = df.iloc[:, 0]
    time_series = time_series.ffill()
    plt.figure(figsize=(20, 4))
    plt.plot(range(len(time_series)), time_series, color='#1f77b4', linewidth=1.2)
    plt.title("Gold Price")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gold_price.png'), dpi=150)
    plt.close()
    output_data = [{
        'Index': 0,
        'Series': str(time_series.tolist()),
        'Labels': '[]',
        'Positions': '[]'
    }]
    pd.DataFrame(output_data).to_csv(os.path.join(output_dir, 'streaming_ts_gold.csv'), index=False)
    return f"Processed gold: 1 series"