---
license: other
license_name: openmdw-1.1
library_name: transformers
base_model: google/gemma-4-31b
tags:
- Ising Calibration
- quantum
- calibration
- vision-language
- nvidia
- bf16
---

# NVIDIA-Ising-Calibration-1.5-31B-BF16 Overview

<div align="center" style="line-height: 1;">
<a href="https://build.nvidia.com/nvidia/ising-calibration-1.5-31b" target="_blank" style="margin: 2px;">
    <img alt="Build API" src="https://img.shields.io/badge/API-Ising_Calibration_1.5-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://research.nvidia.com/publication/2026-04_qcaleval-benchmarking-vision-language-models-quantum-calibration-plot" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-QCalEval-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://huggingface.co/datasets/nvidia/QCalEval" target="_blank" style="margin: 2px;">
    <img alt="Benchmark Dataset" src="https://img.shields.io/badge/Benchmark_Dataset-QCalEval-76B900?logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/NVIDIA/QCalEval" target="_blank" style="margin: 2px;">
    <img alt="QCalEval GitHub" src="https://img.shields.io/badge/QCalEval-GitHub-76B900?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://developer.nvidia.com/ising" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Ising Developer Page-Learn More Here!-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://openmdw.ai/license/1-1/" target="_blank" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-OpenMDW_1.1-f5de53?color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

![QCalEval zero-shot and MM-ICL leaderboards, with Ising Calibration 1.5 highlighted](./qcaleval_leaderboard_2026_07.png)

## Description:  
NVIDIA-Ising-Calibration-1.5-31B-BF16 is a dense multimodal vision-language model built on Gemma 4 31B. It analyzes quantum computing calibration experiment plots and generates structured technical text across six analysis categories: technical description, experimental conclusion, experimental significance, fit quality assessment, parameter extraction, and experiment success classification.  
NVIDIA-Ising-Calibration-1.5-31B-BF16 was developed by NVIDIA for quantum calibration plot understanding.  
_This model is ready for commercial use._

## License/Terms of Use

GOVERNING TERMS: Use of this model is governed by the [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/). ADDITIONAL INFORMATION: Apache License, Version 2.0.

### Deployment Geography:  
Global

### Use Case:  
Quantum computing researchers, calibration engineers, and developers can use this model to analyze experiment plot images and generate technical descriptions, experimental conclusions, significance assessments, fit quality evaluations, parameter extractions, and experiment success classifications. The model assists automated or assisted calibration workflows, and outputs should be validated by domain experts before acting on experimental conclusions.  
  

### Release Date:
**NGC:** 07/23/2026 via https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/nvidia-ising-calibration-1-5-31b

  

## Reference(s):
[Gemma](https://ai.google.dev/gemma)  
[QCalEval Benchmark](https://huggingface.co/datasets/nvidia/QCalEval)  
[QCalEval: Benchmarking Vision-Language Models for Quantum Calibration Plot Understanding](https://arxiv.org/abs/2604.25884)  

## Model Architecture:   
**Architecture Type:** Dense multimodal vision-language model   
**Network Architecture:** Integrated vision processing for experiment plot images combined with a Gemma 4 31B dense language model for autoregressive text generation.  
**This model was developed based on google/gemma-4-31b.**  
**Number of model parameters:** Approximately 31B  

  
 
## Input:  
**Input Type(s):** Text, Image  
**Input Format(s):** String, Other: RGB (.png, .jpeg, .jpg)  
**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D)  
**Other Properties Related to Input:** Single-image or multi-image quantum calibration experiment plots with text prompts delivered through an OpenAI-compatible API. Suggested inference settings use temperature=0.2, zero-shot max_tokens=8192, and ICL max_tokens=32767.  
  

## Output:  
**Output Type(s):** Text  
**Output Format:** String  
**Output Parameters:** One-Dimensional (1D)  
**Other Properties Related to Output:** Natural language technical analysis, experimental conclusions, significance assessments, fit quality evaluations, parameter extractions, and experiment success classifications. Output length is controlled by max_tokens, and output is delivered through an OpenAI-compatible API served by NVIDIA NIM with a vLLM backend.  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** NVIDIA NIM with vLLM backend, BF16 serving precision    
**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Blackwell
* NVIDIA Hopper

**Supported Operating System(s):** Linux  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  

This AI model can be embedded as an Application Programming Interface (API) call into the software environment described above.

## Model Version(s): 
NVIDIA-Ising-Calibration-1.5-31B-BF16 v1.5.0  

To integrate the model, follow the NVIDIA NIM or Gemma 4 31B serving guidance, substituting the model path with nvidia/NVIDIA-Ising-Calibration-1.5-31B-BF16. The model is served through NVIDIA NIM with a vLLM backend. Suggested inference settings are temperature=0.2, zero-shot max_tokens=8192, and ICL max_tokens=32767.  
 
## Training, Testing, and Evaluation Datasets:  


## Training Dataset:

**Data Modality:**
* Image  
* Text  


**Image Training Data Size:** Less than a Million Images  
**Text Training Data Size:** Less than a Billion Tokens  
**Data Collection Method by dataset:** Synthetic
**Labeling Method by dataset:** Synthetic  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The training corpus contains 72.5K total supervised entries: 23.8K ICL-formatted entries for multi-image demonstrations and 48.7K zero-shot entries augmented using Qwen3.5-397B-A17B. The data comes from internal quantum-calibration/QCal-style synthetic data generation and focuses on calibration plot interpretation. The dataset was assembled for the 2026 Ising Calibration 1.5 release cycle.

### Testing Dataset:

**Data Collection Method by dataset:** Synthetic
**Labeling Method by dataset:** Synthetic
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** QCalEval was used as the primary external release validation benchmark for Ising Calibration 1.5. It is a quantum-calibration evaluation suite with multimodal plot-plus-text tasks covering zero-shot and ICL/few-shot settings across calibration interpretation, parameter extraction, diagnostic reasoning, and calibration-status classification. The release candidate was evaluated on 243 zero-shot examples with 1,458 response slots and 236 ICL examples with 708 response slots; raw outputs were checked for completeness and server errors, then judged with both GPT and Gemini judges to produce aggregate scores. The benchmark uses curated quantum-calibration plot tasks rather than raw sensor telemetry, and should be interpreted as domain validation rather than a broad general-purpose capability benchmark.

### Evaluation Dataset:
**Benchmark Score:** QCalEval benchmark scores. Scores are the simple average of GPT-5.4 and Gemini-3.1-Pro judges.  

Zero-shot scores:  

| Model | Mean | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA-Ising-Calibration-1.5-31B-BF16 | 74.5 | 86.2 | 66.3 | 63.5 | 86.4 | 68.4 | 76.5 |
| Gemma-4-31B-IT | 68.8 | 85.6 | 54.3 | 59.8 | 82.7 | 68.3 | 62.1 |

MM-ICL scores:  

| Model | Mean | Q3 | Q5 | Q6 |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA-Ising-Calibration-1.5-31B-BF16 | 81.2 | 71.7 | 85.0 | 86.9 |
| Gemma-4-31B-IT | 81.2 | 80.6 | 76.9 | 86.0 |

**Data Collection Method by dataset:** Synthetic  
**Labeling Method by dataset:** Synthetic  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The evaluation dataset is QCalEval, a synthetic vision-language benchmark for quantum calibration plots containing 243 entries across 87 scenario types from 22 experiment families, covering superconducting qubits and neutral atoms. It assesses six question types: technical description, experimental conclusion, experimental significance, fit quality assessment, parameter extraction, and experiment success classification. Ground-truth labels are derived from simulation parameters, providing curated quantum calibration experiments.



## Inference:
**Acceleration Engine:** NVIDIA NIM with vLLM backend, BF16 precision <br>
**Test Hardware:** NVIDIA Ada Lovelace (L40S), NVIDIA Hopper (H200 SXM), NVIDIA Grace Hopper (GH200), NVIDIA Blackwell (B200), NVIDIA Grace Blackwell (GB200), NVIDIA Grace Blackwell Ultra (GB300), and NVIDIA Grace Blackwell (DGX Spark with GB10)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.
Please make sure you have proper rights and permissions for all input image content; if image inputs include people, personal health information, or intellectual property, developers are responsible for appropriate handling.  
For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.  
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
