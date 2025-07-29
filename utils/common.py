import re
import matplotlib
from PIL import Image
import pandas as pd
import math
import csv
import ast
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils.config import *
matplotlib.use('Agg')
def safe_filename(name):
    return re.sub(r'[^\w]', '_', name.encode('ascii', 'ignore').decode('ascii').replace(' ', '_').replace('�', ''))
def get_label_dir_name(label):
    return {
        'level shift': "outlier_shift_level",
        'sudden spike': "outlier_spike_sudden"
    }.get(label, '_'.join(label.split()[::-1]))
def create_grand_combined(image_paths, output_base, dataset_name):
    images = [Image.open(p) for p in image_paths if os.path.exists(p)]
    if not images: return
    widths, heights = zip(*(img.size for img in images))
    max_width, max_height = max(widths), max(heights)
    cols = 2
    rows = math.ceil(len(images) / cols)
    total_width = cols * max_width
    total_height = rows * max_height
    new_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        new_img.paste(img, (col * max_width, row * max_height))
    new_img.save(os.path.join(output_base, f"Inf-LLM-pattern_{dataset_name}.png"), dpi=(300, 300))
    for img in images: img.close()
def generate_stream_series(label_data):
    long_series = []
    labels_info = []
    positions_info = []
    current_position = 0
    for label in FIXED_ORDER:
        if not label_data.get(label):
            continue
        segment = random.choice(label_data[label])
        segment_length = len(segment)
        long_series.extend(segment)
        labels_info.append(label)
        positions_info.append((current_position, current_position + segment_length - 1))
        current_position += segment_length
    return long_series, labels_info, positions_info
def plot_and_save_series(series, index, output_dir):
    plt.figure(figsize=(20, 4))
    plt.plot(series, color='#1f77b4', linewidth=1.2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'streaming_ts_{index}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path