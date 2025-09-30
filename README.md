<div align="center">
  <h2><b> <img src="https://github.com/user-attachments/assets/d275986b-27c3-4462-afd7-5c58a836a0b8" style="width:30px;height:30px;"> InfTS-LLM: Infinite Alignment of Streaming Time Series with LLMs </b></h2>
</div>

This repository provides the code for our paper, which introduces a framework aligning infinite-length streaming time series with LLMs through LLM-friendly representations, enabling temporal understanding, reasoning, and forecasting.
> If you find our work useful in your research. Please consider giving a star ⭐:

## Demonstration
https://github.com/user-attachments/assets/35c9050c-edd0-400c-8e77-6366828031e0

<!--
## Key Features
InfTS-LLM can be directly applied to any LLMs without retraining:
- ✅ **Native support for multivariate time series**
-->

### Example Application
Here is an example of a InfTS-LLM application, which allows users to interact with a LLM to understand and reason about **Streaming Time Series**:
<p align="left">
  <img width="2700" height="1090" alt="image" src="https://github.com/user-attachments/assets/e627d7d3-8139-45b7-9a5f-cd50bb529eef" />
</p>

## Abstract
We address fundamental limitations of existing multimodal Large Language Models (LLMs) in time-series monitoring scenarios: **Infinite Alignment**. Streaming time series often appear as continuous observation scenarios, such as database usage monitoring. Existing methods ignore aligning **Inf**inite-length streaming **T**ime **S**eries and LLMs to support the above scenarios. The primary challenges are infinite-length temporal detection and temporal reasoning beyond the context of LLMs. This paper introduces **InfTS-LLM**, including two components: (1) A **Representative Detector** that extracts temporal semantics by Representative Subsequences for infinite-length temporal detection; and (2) A **Feedback Instructor** leverages representative subsequences to generate images and construct **Pattern-guided Chains of Thought**, enabling LLMs to provide feedback on global impact scores that combine with local representative scores to sustain a memory pool for temporal reasoning beyond the context of LLMs. Extensive evaluations across multiple datasets show that InfTS-LLM achieves state-of-the-art results. Further analysis highlights modality-specific strengths: vision boosts understanding, text supports reasoning, and numerical time series enhance forecasting. These benefits arise from joint contributions of two components.

<p align="left">
  <img width="2688" height="764" alt="image" src="https://github.com/user-attachments/assets/83201329-66c9-447d-a407-d99b2481c9dc" />
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
5. Reasoning task datasets can be download from [AIOps](https://github.com/netmanaiops/kpi-anomaly-detection), [WeatherQA](https://www.bgc-jena.mpg.de/wetter), and [NAB](https://github.com/numenta/NAB), and [Oracle](https://zenodo.org/records/6955909).
   
## Usages
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
