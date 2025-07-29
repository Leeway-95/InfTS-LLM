import os
import json
import logging
import pandas as pd
from memory_pool import MemoryItem
from utils.config import *
logger = logging.getLogger(__name__)

# Load recent/representative series from file
def load_three_parts_from_file(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        parts = [p.strip() for p in content.split('\n\n\n') if p.strip()]
        if len(parts) < 2:
            raise ValueError("Invalid file structure")
        recent_series = parts[0]
        rep_series = parts[1] if len(parts) > 1 else ""
        return recent_series, rep_series,
    except Exception as e:
        logger.error(f"File read error: {str(e)}")
        return "", ""

# Save LLM response to file
def save_full_response(id_val, response):
    os.makedirs(RESPONSE_PATH, exist_ok=True)
    response_file = os.path.join(RESPONSE_PATH, f"response_row_{id_val}.txt")
    with open(response_file, 'w', encoding="utf-8") as f:
        f.write(response)
    print(f"Full response saved to {response_file}")
    return response_file

# Save metrics log entry to CSV
def save_log_entry(log_entry):
    try:
        from datetime import datetime
        log_entry['LogTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_df = pd.DataFrame([log_entry])
        if not os.path.exists(LOG_LLM_METRICS_PATH):
            log_df['ID'] = 0
            log_df.to_csv(LOG_LLM_METRICS_PATH, index=False)
        else:
            existing_df = pd.read_csv(LOG_LLM_METRICS_PATH)
            max_id = existing_df['ID'].max()
            log_df['ID'] = max_id + 1
            updated_df = pd.concat([existing_df, log_df], ignore_index=True)
            updated_df.to_csv(LOG_LLM_METRICS_PATH, index=False)
        print(f"Log entry saved to llm_log.csv with ID: {log_df['ID'].iloc[0]}")
    except Exception as e:
        logger.error(f"Log save error: {str(e)}")

# Save memory pool state to JSON
def save_memory_state(memory_pool):
    state = [{'series': item.series, 'position': item.position, 'r_score': item.r_score, 'i_score': item.i_score} for item in memory_pool.items]
    with open(Memory_Pool_PATH, 'w') as f:
        json.dump(state, f)

# Load memory pool state from JSON
def load_memory_state(memory_pool):
    try:
        if os.path.exists(Memory_Pool_PATH):
            with open(Memory_Pool_PATH, 'r') as f:
                state = json.load(f)
            new_items = []
            for item in state:
                try:
                    new_items.append(MemoryItem(
                        series=item['series'],
                        position=item['position'],
                        r_score=item['r_score'],
                        i_score=item['i_score']
                    ))
                except KeyError as e:
                    logger.error(f"Memory item missing key: {str(e)}")
            memory_pool.items = new_items
            memory_pool.update_threshold()
    except Exception as e:
        logger.error(f"Memory load error: {str(e)}")
        memory_pool.items = []

# Update prediction results in CSV
def update_csv(id_val, pred_labels, pred_series, impact_scores):
    try:
        os.makedirs(os.path.dirname(INSTRUCTOR_OUTPUT_PATH), exist_ok=True)
        if not os.path.exists(INSTRUCTOR_OUTPUT_PATH):
            df = pd.DataFrame(columns=["ID", "Pred_Labels", "Pred_Series", "Impact_Scores"])
            df.to_csv(INSTRUCTOR_OUTPUT_PATH, index=False)
        try:
            df = pd.read_csv(INSTRUCTOR_OUTPUT_PATH)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame(columns=["ID", "Pred_Labels", "Pred_Series", "Impact_scores"])
        for col in ["ID", "Pred_Labels", "Pred_Series", "Impact_scores"]:
            if col not in df.columns:
                df[col] = None
        if id_val in df.index:
            df.loc[id_val, "Pred_Labels"] = str(pred_labels)
            df.loc[id_val, "Pred_Series"] = pred_series
            df.loc[id_val, "Impact_scores"] = impact_scores
        else:
            new_row = {
                "ID": id_val,
                "Pred_Labels": str(pred_labels),
                "Pred_Series": pred_series,
                "Impact_scores": impact_scores
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(INSTRUCTOR_OUTPUT_PATH, index=False)
        print(f"Updated CSV row {id_val}")
    except Exception as e:
        logger.error(f"CSV Update Error: {str(e)}")
        raise