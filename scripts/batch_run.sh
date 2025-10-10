#!/bin/bash
#export PYTHONPATH=/path/to/project_root:$PYTHONPATH
#export PYTHONPATH=/Users/leeway/PycharmProjects/InfTS-LLM:$PYTHONPATH
export PYTHONPATH=/Users/leeway/PycharmProjects/InfTS-LLM:$PYTHONPATH

echo "0.Clearing datasets..."
cd preprocess
python dataset_init.py
cd ..

echo "1.Converting datasets to stream format..."
cd preprocess
python dataset2stream.py
cd ..

cd model_detector
echo "2.Starting Representative Detector..."
python detector.py
cd ..

echo "3.Starting Pattern-guided Instructor..."
cd model_instructor
python instructor.py
cd ..