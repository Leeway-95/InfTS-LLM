import logging
import sys
import time
import json
import pandas as pd
import ast
from memory_pool import MemoryItem, MemoryPool
from llm_api import callOpenAILLM, parse_llm_output
from file_io import load_three_parts_from_file, save_full_response, save_log_entry, save_memory_state, \
    load_memory_state, update_csv
from prompt_builder import build_pcot_prompt
from memory_updater import update_memory_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from utils.config import *

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# Main processing pipeline
if __name__ == "__main__":
    original_stdout = sys.stdout
    directory = os.path.dirname(LOG_LLM_RESPONSE_PATH)
    os.makedirs(directory, exist_ok=True)
    with open(LOG_LLM_RESPONSE_PATH, 'a', encoding='utf-8') as log_file:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n\n{'=' * 80}\n")
        log_file.write(f"NEW SESSION STARTED AT: {start_time}\n")
        log_file.write(f"{'=' * 80}\n\n")
        sys.stdout = Tee(original_stdout, log_file)
        memory_pool = MemoryPool()
        if MEMORY_STORAGE_MODE == MODE_FILE:
            load_memory_state(memory_pool)
        os.makedirs(INSTRUCTOR_INPUT_PATH, exist_ok=True)
        summary_path = os.path.join(os.path.dirname(INSTRUCTOR_INPUT_PATH), 'summary.csv')
        summary_df = pd.read_csv(summary_path)
        summary_dict = {}
        for idx, row in summary_df.iterrows():
            total_length = len(ast.literal_eval(row['Series']))
            rep_positions = ast.literal_eval(row['Representative_Subsequence_Positions'])
            r_scores = [(end - start) / total_length for start, end in rep_positions]
            summary_dict[idx] = r_scores
        sorted_ids = sorted(summary_df.index)
        print(f"Found {len(sorted_ids)} rows to process")
        for id_val in sorted_ids:
            print(f"\n{'=' * 60}")
            print(f"Processing ID: {id_val} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 60}")
            row = summary_df.loc[id_val]
            full_series = ast.literal_eval(row['Series'])
            recent_series = ', '.join(map(str, full_series[:PREDICT_LENGTH]))
            rep_positions = ast.literal_eval(row['Representative_Subsequence_Positions'])
            rep_subsequences = []
            for start, end in rep_positions:
                rep_subsequences.append(full_series[start:end + 1])
            rep_series = '\n'.join([', '.join(map(str, seq)) for seq in rep_subsequences])
            if recent_series:
                full_prompt = build_pcot_prompt(recent_series, rep_series, memory_pool)
                log_file.write("\n[FULL PROMPT]")
                log_file.write(full_prompt)
                log_file.write(f"{'=' * 60}")
                result, in_tokens, out_tokens, ttft, resp_time, total_time, cost = callOpenAILLM(full_prompt)
                log_file.write("\n[FULL RESPONSE]")
                log_file.write(full_prompt)
                log_file.write(f"{'=' * 60}")
                response_file = save_full_response(id_val, result)
                log_entry = {
                    'ID': id_val,
                    'TTFT': ttft,
                    'InputTokens': in_tokens,
                    'OutputTokens': out_tokens,
                    'Cost': cost,
                    'TotalTime': total_time,
                    'ResponseFile': response_file
                }
                save_log_entry(log_entry)
                pred_labels, pred_series_str, impact_scores_str = parse_llm_output(result)
                try:
                    impact_scores = json.loads(impact_scores_str)
                except:
                    impact_scores = []
                r_scores = summary_dict.get(id_val, [])
                rep_positions = ast.literal_eval(row['Representative_Subsequence_Positions'])
                update_memory_pool(rep_series, rep_positions, r_scores, impact_scores, memory_pool)
                update_csv(id_val, pred_labels, pred_series_str, impact_scores_str)
                if MEMORY_STORAGE_MODE == MODE_FILE:
                    save_memory_state(memory_pool)
        sys.stdout = original_stdout