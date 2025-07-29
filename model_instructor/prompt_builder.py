from utils.config import *

# Build CoT prompt for LLM
def build_pcot_prompt(recent_series, rep_series, memory_pool):
    rep_lines = rep_series.split('\n')[:TOP_K] if rep_series else []
    formatted_reps = []
    for i, line in enumerate(rep_lines):
        if line.strip():
            formatted_reps.append(f"R_{i + 1}: {line}")
    memory_patches = memory_pool.get_memory_patches()
    pcot_input = "Recent Subsequence: " + recent_series + "\n\n"
    if formatted_reps:
        pcot_input += "Representative Subsequence: \n".join(formatted_reps) + "\n\n"
    if memory_patches:
        pcot_input += "\n".join(memory_patches) + "\n\n"
    with open(INPUT_PATH_COT, "r", encoding="utf-8") as f:
        pcot_input += f.read()
    return pcot_input