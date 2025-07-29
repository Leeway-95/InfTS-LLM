import json
import os
from utils.config import *

# Memory item container
class MemoryItem:
    __slots__ = ('series', 'position', 'r_score', 'i_score', 'hm')
    def __init__(self, series, position, r_score, i_score):
        self.series = series
        self.position = position
        self.r_score = float(r_score)
        self.i_score = float(i_score)
        self.hm = (2 * self.r_score * self.i_score) / (self.r_score + self.i_score + 1e-10)

# Manage memory pool of patterns
class MemoryPool:
    # Initialize memory pool
    def __init__(self, max_size=20):
        self.items = []
        self.max_size = max_size
        self.threshold = 0.0
        if SAVE_MEMORY_POOL and os.path.exists(Memory_Pool_PATH):
            with open(Memory_Pool_PATH, 'r') as f:
                data = json.load(f)
                loaded_items = []
                for d in data:
                    item = MemoryItem(d['series'], d['position'], d['r_score'], d['i_score'])
                    loaded_items.append(item)
                loaded_items.sort(key=lambda x: x.hm, reverse=True)
                self.items = loaded_items[:max_size]

    # Add new item to memory
    def add_item(self, item):
        self.items.append(item)
        self.update_threshold()
        if SAVE_MEMORY_POOL:
            self.save_to_file()

    # Update memory retention threshold
    def update_threshold(self):
        if len(self.items) <= self.max_size:
            self.threshold = 0.0
            return
        sorted_items = sorted(self.items, key=lambda x: x.hm, reverse=True)
        if len(sorted_items) > self.max_size:
            self.threshold = sorted_items[self.max_size - 1].hm
            self.items = sorted_items[:self.max_size]

    # Save memory state to file
    def save_to_file(self):
        all_items = []
        if os.path.exists(Memory_Pool_PATH):
            with open(Memory_Pool_PATH, 'r') as f:
                all_items = json.load(f)
        for item in self.items:
            item_dict = {
                'series': item.series,
                'position': item.position,
                'r_score': item.r_score,
                'i_score': item.i_score
            }
            if item_dict not in all_items:
                all_items.append(item_dict)
        all_items.sort(key=lambda x: (2 * float(x['r_score']) * float(x['i_score'])) / (
                    float(x['r_score']) + float(x['i_score']) + 1e-10), reverse=True)
        all_items = all_items[:MEMORY_POOL_MAX_ITEMS]
        with open(Memory_Pool_PATH, 'w') as f:
            json.dump(all_items, f)

    # Get memory items as formatted strings
    def get_memory_patches(self):
        return [f"MEM_{idx}: {item.series} | POS: {item.position} | HM: {item.hm:.4f}" for idx, item in
                enumerate(self.items)]