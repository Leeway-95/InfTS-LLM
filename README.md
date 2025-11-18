<div align="center">
  <h2><b> <img src="https://github.com/user-attachments/assets/d275986b-27c3-4462-afd7-5c58a836a0b8" style="width:30px;height:30px;"> InfTS-LLM: Beyond-Context Alignment of Streaming Time Series with LLMs </b></h2>
</div>

[![GitHub Stars](https://img.shields.io/github/stars/Leeway-95/InfTS-LLM?style=social)](https://github.com/Leeway-95/InfTS-LLM/stargazers)
![Topic](https://img.shields.io/badge/Streaming%20Time%20Series%20&%20LLMs%20-%20Infinite--Alignment-blueviolet)

This repository provides the code for our paper, which introduces a zero-shot framework for beyond-context alignment between streaming time series with LLMs through LLM-friendly representations, enabling temporal understanding, reasoning, and forecasting.
>  ✨ If you find our work useful for your research, please consider giving it a <strong>star ⭐ on GitHub</strong> to stay updated with future releases.

## Demonstration
https://github.com/user-attachments/assets/35c9050c-edd0-400c-8e77-6366828031e0

<!--
## Key Features
InfTS-LLM can be directly applied to any LLMs without retraining:
- ✅ **Native support for multivariate time series**
-->

### Example Demonstration
Here is an example of InfTS-LLM, enabling users to interact with the LLM for understanding, reasoning, and forecasting over streaming time series.
<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/9da39df5-8bc8-48c1-b48f-d202a4fcf1e9" />
</p>

## Abstract
We address fundamental limitations of multimodal Large Language Models (LLMs) in time-series monitoring scenarios: **Beyond-context Alignment**. Streaming time series often appear as continuous observation scenarios, such as database usage monitoring. Existing methods ignore beyond-context alignment between streaming time series and LLMs to support the above scenarios. The primary challenges are continuous temporal-pattern detection and beyond-context temporal reasoning. This paper introduces **InfTS-LLM**, including two components: (1) A **Representative Detector** that extracts temporal semantics by representative subsequences for continuous temporal-pattern detection; and (2) A **Feedback Instructor** leverages representative subsequences to generate images and construct pattern-guided chains of thought, enabling LLMs to provide feedback on global impact scores that combine with local representative scores to sustain a memory pool for beyond-context temporal reasoning. Extensive evaluations across multiple datasets show that InfTS-LLM achieves state-of-the-art results. Further analysis highlights modality-specific strengths: visual modality boosts understanding, textual modality supports reasoning, and numerical time series enhance forecasting.

<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/1d12502d-51b6-4456-bedc-d2f90733159f" />
</p>

Inspired by the process of “**skipping a stone on water**”, splashes and ripples mark the trajectory, while a stone flying straight across leaves no trace. In this process, each splash corresponds to a **Representative Subsequence** identified by the Representative Detector, serving as a visible trace of streaming time series. Similarly, the Feedback Instructor maintains these traces by retaining high-scoring subsequences in memory, analogous to ripples that persist for a period before gradually fading.

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
2. Numerical forecasting task datasets can be download from [ETTm](https://drive.google.com/drive/folders/1eXR9w5eW2IMaJzbKWuMjTvdXehvYpMKA) and [Weather](https://drive.google.com/drive/folders/1cKPfcZamEWcF48ZvXyubwhkuz84tupu4).
3. Event forecasting task datasets can be download from [Finance](https://github.com/geon0325/TimeCAP/tree/main/dataset/finance), [Healthcare](https://github.com/geon0325/TimeCAP/tree/main/dataset/healthcare), and [Weather](https://github.com/geon0325/TimeCAP/tree/main/dataset/weather).
4. Understanding task dataset can be download from [TSQA](https://huggingface.co/datasets/ChengsenWang/TSQA).
5. Reasoning task datasets can be download from [AIOps](https://github.com/netmanaiops/kpi-anomaly-detection), [WeatherQA](https://www.bgc-jena.mpg.de/wetter), and [NAB](https://github.com/numenta/NAB), [Oracle](https://zenodo.org/records/6955909), and [MCQ2](https://github.com/behavioral-data/TSandLanguage)
   
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
