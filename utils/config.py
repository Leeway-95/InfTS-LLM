import os

LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Model parameters
TOP_K = 5
PREDICT_LENGTH = 48
MAX_TOP_HM_COUNT = 5
MEMORY_POOL_MAX_ITEMS = 7
WINDOW_SIZE = 20
JUMP = 1
SAMPLE_SIZE = 100
PEAKS_HEIGHT = 0.8
P_VALUE = 1e-70
K = 3
N = 10
EXTRACT_REP_SERIES = 1
SIMILARITY = 'cid'
MODE_FILE = 'FILE'
MODE_MEMORY = 'MEMORY'
MEMORY_STORAGE_MODE = MODE_FILE
OUTPUT_IMAGE = True
CREATE_COMBINED_PER_LABEL = False
SAVE_MEMORY_POOL = False
CREATE_GRAND_COMBINED = True

# Pattern labels
LABELS_TO_PROCESS = [
    "upward trend", "downward trend", "increased volatility",
    "decreased volatility", "fixed seasonal", "shifting seasonal",
    "sudden spike", "level shift"
]
FIXED_ORDER = [
    "sudden spike", "fixed seasonal",
    "level shift", "sudden spike",
    "downward trend", "level shift",
    "shifting seasonal", "increased volatility"
]

# Path configurations
DETECTOR_INPUT_PATH = [
    "../datasets/ettm/stream_flink-ETTm1/streaming_ts_ETTm1.csv",
    "../datasets/ettm/stream_flink-ETTm2/streaming_ts_ETTm2.csv",
    "../datasets/weather/stream_flink-weather/streaming_ts_weather.csv",
    "../datasets/tsqa/stream_flink-TSQA/streamingTS_summary.csv",
    "../datasets/gold/stream_flink-gold/streaming_ts_gold.csv"]
DETECTOR_OUTPUT_PATH = [
    "../datasets/ettm/detection_ETTm1/",
    "../datasets/ettm/detection_ETTm1/",
    "../datasets/weather/detection/",
    "../datasets/tsqa/detection/",
    "../datasets/gold/detection/"]
INSTRUCTOR_INPUT_PATH = "../datasets/tsqa/detection/"
INSTRUCTOR_OUTPUT_PATH = '../datasets/tsqa/predict/predictResult.csv'
DATASET_PATHS = {
    "ETTm1": os.path.join("../datasets/ettm/ETTm1.csv"),
    "ETTm2": os.path.join("../datasets/ettm/ETTm2.csv"),
    "gold": os.path.join("../datasets/gold/gold.csv"),
    "TSQA": os.path.join("../datasets/tsqa/TSQA.csv"),
    "weather": os.path.join("../datasets/weather/weather.csv")
}
OUTPUT_DIRS = {
    "ETTm1": os.path.join("../datasets/ettm/stream-ETTm1"),
    "ETTm2": os.path.join("../datasets/ettm/stream-ETTm2"),
    "gold": os.path.join("../datasets/gold/stream-gold"),
    "TSQA": os.path.join("../datasets/tsqa/stream-TSQA"),
    "weather": os.path.join("../datasets/weather/stream-weather")
}
INPUT_PATH_COT = 'PCoT.txt'
LOG_LLM_METRICS_PATH = '../logs/llm_log.csv'
LOG_LLM_RESPONSE_PATH = '../logs/printLog.txt'
Memory_Pool_PATH = 'memory_pool.json'
RESPONSE_PATH = '../datasets/tsqa/predict/responses/'
OVERLAP_PLOTS = True

# Stream configurations
PARALLELISM = 4
DATASETS = [
    "streamingTS/ETTm1.csv",
    "streamingTS/ETTm2.csv",
    "streamingTS/weather.csv",
    "streamingTS/gold.csv"
]
# Stream processing config
INPUT_TOPIC = "InfTS_input"
OUTPUT_TOPIC = "InfTS_output"
FLINK_JOB_NAME = "InTS-LLM"
CHECKPOINT_INTERVAL = 60000  # 1 minute in ms