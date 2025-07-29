import logging
from memory_pool import MemoryItem
from utils.config import *
logger = logging.getLogger(__name__)

# Update memory pool with new patterns
def update_memory_pool(rep_series, position_info, r_scores, i_scores, memory_pool):
    if not rep_series or not position_info:
        return
    rep_lines = rep_series.split('\n')
    valid_items = []
    n = min(len(rep_lines), len(position_info), len(r_scores), len(i_scores))
    for i in range(n):
        rep_line = rep_lines[i].strip()
        if not rep_line:
            continue
        try:
            pos_tuple = position_info[i]
            pos_str = str(pos_tuple)
            r_val = r_scores[i]
            i_val = float(i_scores[i])
            if r_val < 0 or r_val > 1 or i_val < 0 or i_val > 1:
                continue
            score1 = 1 - r_val
            score2 = i_val
            hm = 2 * score1 * score2 / (score1 + score2) if (score1 + score2) != 0 else 0.0
            valid_items.append((hm, rep_line, pos_str, r_val, i_val))
        except Exception as e:
            logger.error(f"Processing error at index {i}: {str(e)}")
    valid_items.sort(key=lambda x: x[0], reverse=True)
    top_items = valid_items[:MAX_TOP_HM_COUNT]
    new_items = []
    for hm, rep_line, pos_str, r_val, i_val in top_items:
        try:
            new_item = MemoryItem(series=rep_line, position=pos_str, r_score=r_val, i_score=i_val)
            new_items.append(new_item)
        except Exception as e:
            logger.error(f"Memory add error: {str(e)}")
    for item in new_items:
        memory_pool.add_item(item)