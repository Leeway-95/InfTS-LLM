from utils.common import *
from dataset_ett import *
from dataset_gold import *
from dataset_tsqa import *
from dataset_weather import *

import os
if __name__ == '__main__':
    results = []
    results.append(process_ett_dataset("ETTm1"))
    results.append(process_ett_dataset("ETTm2"))
    results.append(process_gold_dataset())
    results.append(process_tsqa_dataset())
    results.append(process_weather_dataset())
    print("\nProcessing Summary:")
    for res in results:
        print(f"- {res}")
    print(f"\nOutput directories created at:")
    for name, path in OUTPUT_DIRS.items():
        if os.path.exists(path):
            print(f"- {name}: {os.path.abspath(path)}")