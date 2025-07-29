#!/bin/bash
#export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTHONPATH=/Users/leeway/PycharmProjects/InfTS-LLM:$PYTHONPATH
echo "Converting datasets to stream format..."
cd preprocess
python dataset2stream.py
echo "Starting Representative Detector..."
cd ../model_detector
python detector.py
echo "Starting Pattern-guided Instructor..."
cd ../model_instructor
python instructor.py