from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.formats.json import JsonRowDeserializationSchema, JsonRowSerializationSchema
from pyflink.common import Row, Types

from model_instructor.memory_pool import *
from model_instructor.memory_updater import update_memory_pool
from model_instructor.prompt_builder import build_pcot_prompt
from pattern_analyzer import PatternAnalyzer
from llm_processor import LLMProcessor

from utils.config import *

def create_processing_job():
    """Create and execute the Flink processing job"""
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(PARALLELISM)
    env.enable_checkpointing(CHECKPOINT_INTERVAL)

    # Build processing pipeline
    (
        env.add_source(DATASET_PATHS)
        .map(lambda row: {
            'id_val': row[0],
            'series': row[1],
            'metadata': row[2]
        })
        .name("Convert to Dict")
        .map(PatternAnalyzer())
        .name("Pattern Analysis")
        .flat_map(update_memory_pool())
        .name("Memory Update")
        .key_by(lambda value: value['id_val'])
        .process(MemoryPool())
        .name("Memory Pool")
        .map(build_pcot_prompt())
        .name("Prompt Building")
        .map(LLMProcessor())
        .name("LLM Processing")
        .map(lambda value: Row(
            value['id_val'],
            value['recent_series'],
            value.get('rep_series', ""),
            value.get('parsed_labels', []),
            value.get('pred_series', "[]"),
            value.get('impact_scores', "[]"),
            value.get('memory_patches', [])
        ))
        .name("Prepare Output")
    )

    env.execute(FLINK_JOB_NAME)