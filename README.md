<div align="center">
  <h2><b> <img src="https://github.com/user-attachments/assets/d275986b-27c3-4462-afd7-5c58a836a0b8" style="width:30px;height:30px;"> InfTS-LLM: Aligning Streaming Time Series with LLMs via Pattern-Guided Representative Subsequences for Beyond-Context Understanding </b></h2>
</div>

This repository contains the code for our paper, where we porpose an intuitive yet effective framework aligning streaming time series with LLMs for beyond-context understanding.
<!--
> If you find our work useful in your research. Please consider giving a star ⭐:
-->
## Demonstration
https://github.com/user-attachments/assets/35c9050c-edd0-400c-8e77-6366828031e0

## Abstract
Large Language Models (LLMs) have demonstrated remarkable effectiveness in long-context language understanding, yet their potential remains underexplored for handling infinite-length numerical signals, such as streaming time series. Existing methods either empower LLMs with time-series or infinite-length adaptation, but seldom integrate both jointly. Among the primary challenges are efficient temporal pattern detection and beyond-context temporal understanding. This paper introduces **InfTS-LLM**, a unified framework that enables LLMs to adapt to both **Inf**inite-length streams and **T**ime-**S**eries for beyond-context understanding by incorporating two components: (1) a **Representative Detector** that extracts representative subsequences to reduce input redundancy and token overhead while capturing the temporal semantics from streaming time series; and (2) a **Pattern-guided Instructor** that constructs the pattern-guided chain-of-thought inputs to the LLM. It identifies the temporal semantics of representative subsequences and preserves sufficient memory capacity in the memory pool to store the historical subsequences. Experiments on three real-world datasets and one synthetic dataset across alignment and forecasting tasks show that InfTS-LLM outperforms five competitive baselines, achieving state-of-the-art performance.

<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/b2e72655-3718-41e6-a51d-c926db1f122c" />
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
<!--
## Contact Us
For inquiries or further assistance, contact us at [leeway@ruc.edu.cn](mailto:leeway@ruc.edu.cn).
-->
