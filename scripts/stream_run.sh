#!/bin/bash
#export PYTHONPATH=/path/to/project_root:$PYTHONPATH

export PYTHONPATH=/Users/leeway/PycharmProjects/InfTS-LLM:$PYTHONPATH

echo "Converting datasets to stream format..."
cd preprocess
python dataset2stream.py

echo "Starting Flink job..."
cd ../stream_flink
python flink_app.py