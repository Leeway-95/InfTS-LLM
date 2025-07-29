from pyflink.datastream import MapFunction, KeyedProcessFunction
from pyflink.common.typeinfo import Types
from pyflink.datastream.state import ListStateDescriptor, ValueStateDescriptor

from utils.config import *


class MemoryPool(KeyedProcessFunction):
    def __init__(self, max_size: int = MEMORY_POOL_MAX_ITEMS):
        super().__init__()
        self.max_size = max_size

    def open(self, runtime_context):
        # Define state descriptors
        self.items_state = runtime_context.get_list_state(
            ListStateDescriptor(
                "memory_items",
                Types.PICKLED_BYTE_ARRAY()
            )
        )
        self.threshold_state = runtime_context.get_state(
            ValueStateDescriptor(
                "threshold",
                Types.FLOAT()
            )
        )

    def process_element(self, value, ctx):
        # Get current state
        current_items = list(self.items_state.get()) or []
        current_threshold = self.threshold_state.value() or 0.0

        # Add new item if present
        if 'memory_item' in value:
            new_item = value['memory_item']
            current_items.append(new_item)

            # Update threshold and trim if needed
            if len(current_items) > self.max_size:
                sorted_items = sorted(current_items, key=lambda x: x.hm, reverse=True)
                current_threshold = sorted_items[self.max_size - 1].hm
                current_items = sorted_items[:self.max_size]

            # Update state
            self.items_state.update(current_items)
            self.threshold_state.update(current_threshold)

        # Get memory patches for output
        memory_patches = [
            f"MEM_{idx}: {item.series} | POS: {item.position} | HM: {item.hm:.4f}"
            for idx, item in enumerate(current_items)
        ]

        # Forward the enriched data
        yield {
            **value,
            'memory_patches': memory_patches,
            'memory_threshold': current_threshold
        }