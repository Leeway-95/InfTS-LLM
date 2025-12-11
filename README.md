<div align="center">
  <h2><b> <img src="https://github.com/user-attachments/assets/d275986b-27c3-4462-afd7-5c58a836a0b8" style="width:30px;height:30px;"> InfTS-LLM: Aligning Streaming Time Series with LLMs for Beyond-Context Question Answering </b></h2>
</div>

[![GitHub Stars](https://img.shields.io/github/stars/Leeway-95/InfTS-LLM?style=social)](https://github.com/Leeway-95/InfTS-LLM/stargazers)
![Topic](https://img.shields.io/badge/Streaming%20Time%20Series%20&%20LLMs%20-%20Infinite--Alignment-blueviolet)

This repository provides the code and demonstration for our paper, which introduces InfTS-LLM, a zero-shot framework for beyond-context alignment between streaming time series with LLMs for Time Series Question Answering (TSQA).
>  ✨ If you find our work useful for your research, please consider giving it a <strong>star ⭐ on GitHub</strong> to stay updated with future releases.

## Demonstration
https://github.com/user-attachments/assets/35c9050c-edd0-400c-8e77-6366828031e0

<!--
## Key Features
InfTS-LLM can be directly applied to any LLMs without retraining:
- ✅ **Native support for multivariate time series**
-->

### Example Demonstration
Here is an example of InfTS-LLM, enabling users to interact with LLMs for temporal understanding, reasoning, and forecasting over streaming time series.
<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/ffba98a4-4179-43d5-99f8-7cfa8ab2f5c2" />
</p>

## Abstract
We address a fundamental limitation of multimodal Large Language Models (LLMs): **Beyond-Context Alignment** for TSQA that refers to answering natural language questions over numerical sequences with temporal semantics and relations. Recently, multimodal Large Language Models (LLMs) have been increasingly applied to TSQA, such as database usage monitoring, leveraging their language capabilities to understand, reason, and forecast multiple streaming time-series metrics. 

However, all existing TSQA methods are limited by the finite in-context length of LLMs. This limitation introduces fundamental challenges in extremely long sequences, including efficient temporal-pattern detection and beyond-context temporal coverage. 

To solve these challenges, this paper proposes **InfTS-LLM**, a zero-shot framework including two components: (a) a Representative Detector that extracts representative subsequences with temporal semantics using a linear-time Cascade Retraced Detection algorithm; and (b) a Feedback Instructor that construct pattern-guided chains of thought for deep reasoning, while enabling LLMs to generate feedback scores that sustain the memory pool with an effective eviction mechanism. 

Extensive evaluations on multiple datasets demonstrate that InfTS-LLM achieves state-of-the-art performance. Further analysis reveals modality-specific strengths: the visual modality enhances understanding, the textual modality boosts reasoning, and the numerical modality improves forecasting.

<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/c190cbd6-9ecf-40c2-a221-ad13371efe53" />
</p>

Inspired by the process of **_“skipping a stone on water”_**, splashes and ripples mark the trajectory, while a stone flying straight across leaves no trace. In this process, each splash corresponds to a **Representative Subsequence** identified by the Representative Detector, serving as a visible trace of streaming time series. Similarly, the Feedback Instructor maintains these traces by retaining high-scoring subsequences in the Memory Pool, analogous to ripples that persist for a period before gradually fading.

## Dependencies
* Python 3.12
* numpy==1.26.4
* numba==0.61.0
* pandas==2.3.1
* apache-flink==2.1.0

```bash
> conda env create -f env_linux.yaml
```

## Datasets
1. Gold datasets can be obtained from our **datasets directory**.
2. Numerical forecasting task datasets can be downloaded from [ETTm](https://drive.google.com/drive/folders/1eXR9w5eW2IMaJzbKWuMjTvdXehvYpMKA) and [Weather](https://drive.google.com/drive/folders/1cKPfcZamEWcF48ZvXyubwhkuz84tupu4).
3. Event forecasting task datasets can be downloaded from [Finance](https://github.com/geon0325/TimeCAP/tree/main/dataset/finance), [Healthcare](https://github.com/geon0325/TimeCAP/tree/main/dataset/healthcare), and [Weather](https://github.com/geon0325/TimeCAP/tree/main/dataset/weather).
4. Understanding task dataset can be downloaded from [TSQA](https://huggingface.co/datasets/ChengsenWang/TSQA).
5. Reasoning task datasets can be downloaded from [AIOps](https://github.com/netmanaiops/kpi-anomaly-detection), [WeatherQA](https://www.bgc-jena.mpg.de/wetter), and [NAB](https://github.com/numenta/NAB), [Oracle](https://zenodo.org/records/6955909), and [MCQ2](https://github.com/behavioral-data/TSandLanguage)
   
## Usages

* ### Obtain InfTS-LLM

```bash
git clone https://github.com/Leeway-95/InfTS-LLM.git
cd InfTS-LLM
pip3 install -r requirements.txt
```

* ### Batch version

```bash
sh scripts/batch_run.sh
```

* ### Stream version
   
```bash
sh scripts/stream_run.sh
```

## Contact Us
For inquiries or further assistance, contact us at [leeway@ruc.edu.cn](mailto:leeway@ruc.edu.cn).
