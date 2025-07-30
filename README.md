<div align="center">
  <h2><b> <img src="https://github.com/user-attachments/assets/d275986b-27c3-4462-afd7-5c58a836a0b8" style="width:30px;height:30px;"> InfTS-LLM: Aligning Streaming Time Series with LLMs via Pattern-Guided Representative Subsequences for Beyond-Context Understanding </b></h2>
</div>

This repository contains the code for our paper, where we porpose an intuitive yet effective framework aligning streaming time series with LLMs for beyond-context understanding.

> If you find our work useful in your research. Please consider giving a star ⭐:

https://github.com/user-attachments/assets/35c9050c-edd0-400c-8e77-6366828031e0

## Abstract
Large Language Models (LLMs) have demonstrated impressive capabilities in long-context language understanding, yet their potential remains largely underexplored for handling unbounded numerical signals, such as streaming time series. Existing works either empower LLMs with time-series adaptation or infinite-length input, but seldom address both jointly. Among the primary challenges are efficient temporal pattern detection and beyond-context temporal understanding. This paper introduces **InfTS-LLM**, a unified framework which enables LLMs to adapt to both **Inf**inite-length **T**ime-**S**eries and textual streams for beyond-context understanding by incorporating two components: (1) a **Representative Detector** that identifies temporal pattern boundaries and representative subsequences to extract temporal semantics from streaming time series; (2) a **Pattern-guided Instructor** that constructs both prompt prefixes and high-impact subsequence information as pattern-guided chain-of-thought templates that are then input to LLMs, while maintaining a memory pool discarding low-scoring subsequences to support beyond-context understanding. Experiments on three real-world datasets and one synthetic dataset across alignment and forecasting tasks show that InfTS-LLM outperforms five competitive baselines, achieving state-of-the-art performance.

<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/aea3c8b0-8000-4f21-bd20-6adb2c2f63a1" />
</p>

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
2. Others can be download from [ETTm](https://drive.google.com/drive/folders/1eXR9w5eW2IMaJzbKWuMjTvdXehvYpMKA), [Weather](https://drive.google.com/drive/folders/1cKPfcZamEWcF48ZvXyubwhkuz84tupu4), and [TSQA](https://huggingface.co/datasets/ChengsenWang/TSQA).

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
