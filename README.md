<div align="center">
  <h2><b> InfTS-LLM: Aligning Streaming Time Series with LLMs via Pattern-Guided Representative Subsequences for Beyond-Context Understanding </b></h2>
</div>

This repository contains the code for our paper, where we porpose an intuitive yet effective framework aligning streaming time series with LLMs for beyond-context understanding.

> If you find our work useful in your research. Please consider giving a star ⭐:

## Abstract
Large Language Models (LLMs) have demonstrated impressive capabilities in long-context language understanding, yet they remain underexplored in handling unbounded numerical signals, such as streaming time series. Existing works either empower LLMs with time-series adaptation or infinite-length support, but seldom address both jointly. Among the primary challenges are efficient temporal pattern detection and beyond-context temporal understanding. This paper introduces **InfTS-LLM** to empower LLMs with **Inf**inite-length support and **T**ime-**S**eries adaptation, comprising two components: (1) a **Representative Detecto**r that extracts representative subsequences by identifying temporal pattern boundaries from streaming time series for pattern detection; (2) a **Pattern-guided Instructor** that constructs both prompt prefixes and representative subsequences into pattern-guided chain-of-thought inputs, and a memory pool manages representative subsequences for beyond-context understanding. Experiments on three real-world and one synthetic dataset across alignment and forecasting tasks show that InfTS-LLM outperforms five competitive baselines, achieving state-of-the-art performance. 

<p align="left">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/2e3664f9-b2c0-4432-b248-259c57980276" />
</p>

## Dependencies

* Python 3.12
* numpy==1.26.4
* numba==0.61.0
* pandas==2.2.3
* openai==1.60.1
* apache-flink==2.0.0

```bash
> conda env create -f env_{ubuntu,windows}.yaml
```

## Datasets
Datasets can be obtained from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or our **datasets directory**.

## Contact Us
For inquiries or further assistance, contact us at [leeway@ruc.edu.cn](mailto:leeway@ruc.edu.cn).
