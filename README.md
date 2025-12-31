<div align="center">
  <h2><b> <img src="https://github.com/user-attachments/assets/d275986b-27c3-4462-afd7-5c58a836a0b8" style="width:30px;height:30px;"> InfTS-LLM: Aligning Streaming Time Series with LLMs for Beyond-Context Question Answering </b></h2>
</div>

[![GitHub Stars](https://img.shields.io/github/stars/Leeway-95/InfTS-LLM?style=social)](https://github.com/Leeway-95/InfTS-LLM/stargazers)
![Topic](https://img.shields.io/badge/Streaming%20Time%20Series%20&%20LLMs%20-%20Beyond--Context%20Question%20Answering-blueviolet)

This repository provides the source code, evaluation datasets, and a demonstration for our paper, which introduces InfTS-LLM, a zero-shot framework for beyond-context Time Series Question Answering (TSQA).
>  ✨ If you find our work useful for your research, please consider giving it a <strong>star ⭐ on GitHub</strong> to stay updated with future releases.

## Example Demonstration
https://github.com/user-attachments/assets/35c9050c-edd0-400c-8e77-6366828031e0

<!--
## Key Features
InfTS-LLM can be directly applied to any LLMs without retraining:
- ✅ **Native support for multivariate time series**
-->

Here is an example of database usage monitoring, demonstrating how users can interact with LLMs for beyond-context TSQA.
<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/234d21f6-d3cc-4222-8dff-413ba0d24887" />
</p>

## Abstract
TSQA refers to answering natural language questions over numerical sequences. Recently, multimodal Large Language Models (LLMs) have been increasingly applied to analyze temporal relations and semantics. 

However, the limited in-context length of LLMs constrains existing methods for extremely long streaming time series, such as database usage monitoring, where LLMs are required to perform understanding, reasoning, and forecasting of multiple time series metrics. This limitation leads to two fundamental challenges: efficient temporal-pattern detection and beyond-context temporal coverage. 

To address these challenges, this paper proposes **InfTS-LLM**, a zero-shot method composed of two components: (a) The Representative Detector extracts representative subsequences with temporal patterns. It adopts a linear-time Cascade Detection algorithm for efficient temporal-pattern detection. (b) The Feedback Instructor constructs Pattern-guided Chains of Thought for deep thinking with LLMs. It generates feedback scores to maintain the Memory Pool with an effective eviction mechanism for beyond-context temporal coverage.

Extensive evaluations on multiple datasets demonstrate that InfTS-LLM achieves state-of-the-art performance. Further analysis reveals strengths across time series representations: visual representations enhance understanding, textual representations boost reasoning, and numerical representations improve forecasting.

<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/f17676f6-c797-4c74-bd1c-2f63ff9ecda3" />
</p>

**_“Skipping a stone on water”_**: First, each **_“hit”_** and **_“bounce”_** corresponds to a temporal-pattern boundary. Second, each **_“splash”_** corresponds to a representative subsequence. Third, the  **_“ripples”_** correspond to cached historical subsequences.

## Dependencies
* Python 3.12
* numpy==1.26.4
* numba==0.61.0
* pandas==2.3.1
* apache-flink==2.1.0

## Datasets
1. Gold datasets can be obtained from our **datasets directory**.
2. Numerical forecasting task datasets can be downloaded from [ETTm](https://drive.google.com/drive/folders/1eXR9w5eW2IMaJzbKWuMjTvdXehvYpMKA) and [Weather](https://drive.google.com/drive/folders/1cKPfcZamEWcF48ZvXyubwhkuz84tupu4).
3. Event forecasting task datasets can be downloaded from [Finance](https://github.com/geon0325/TimeCAP/tree/main/dataset/finance), [Healthcare](https://github.com/geon0325/TimeCAP/tree/main/dataset/healthcare), and [Weather](https://github.com/geon0325/TimeCAP/tree/main/dataset/weather).
4. Understanding task dataset can be downloaded from [TSQA](https://huggingface.co/datasets/ChengsenWang/TSQA).
5. Reasoning task datasets can be downloaded from [AIOps](https://github.com/netmanaiops/kpi-anomaly-detection), [WeatherQA](https://www.bgc-jena.mpg.de/wetter), and [NAB](https://github.com/numenta/NAB), [Oracle](https://zenodo.org/records/6955909), and [MCQ2](https://github.com/behavioral-data/TSandLanguage).
   
## Usages

* ### Obtain InfTS-LLM

```bash
> git clone https://github.com/Leeway-95/InfTS-LLM.git
> cd InfTS-LLM
```

* ### Install Dependencies
```bash
> conda env create -f env_linux.yaml
```
or
```bash
> pip3 install -r requirements.txt
```

* ### Batch version

```bash
> sh scripts/batch_run.sh
```

* ### Stream version
   
```bash
> sh scripts/stream_run.sh
```

## Contact Us
For inquiries or further assistance, contact us at [leeway@ruc.edu.cn](mailto:leeway@ruc.edu.cn).
