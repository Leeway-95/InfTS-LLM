import argparse
import os
from repdetection import *
from utils.config import *

if __name__ == "__main__":
    # Main entry point for time series pattern detection
    for i in range(len(DETECTOR_INPUT_PATH)):
        print(DETECTOR_INPUT_PATH[i])
        parser = argparse.ArgumentParser(description='Time Series Change Point Detection')
        parser.add_argument('--input', type=str, default=DETECTOR_INPUT_PATH[i], help='Input file or directory path')
        parser.add_argument('--output', type=str, default=DETECTOR_OUTPUT_PATH[i], help='Output directory path')

        args = parser.parse_args()
        os.makedirs(args.output, exist_ok=True)
        if not os.path.exists(args.input):
            print(f"Error: Input path '{args.input}' does not exist")
            exit(1)

        if os.path.isdir(args.input):
            txt_files = [f for f in os.listdir(args.input) if f.endswith('.txt')]
            bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            with tqdm(total=len(txt_files),
                      desc="Processing TXT files",
                      bar_format=bar_format,
                      dynamic_ncols=True) as pbar:
                for file in txt_files:
                    file_path = os.path.join(args.input, file)
                    with open(file_path, 'r') as f:
                        data = [float(line.strip()) for line in f]
                    pbar.set_description(f"Processing {file[:15]}")
                    process_series(np.array(data), file, args.output)
                    pbar.update(1)
                    pbar.set_postfix(completed=f"{pbar.n}/{len(txt_files)}")
        elif args.input.endswith('.csv'):
            process_csv_file(args.input, args.output)
        else:
            print("Error: Input must be CSV file or directory containing TXT files")
        print()