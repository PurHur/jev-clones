# Explainability Subcard

## Intended Task/Domain:

Scientific research and quantum computing calibration experiment analysis

## Model Type:

Dense multimodal vision-language model based on Gemma 4 31B

## Intended Users:

Quantum computing researchers, calibration engineers, and developers analyzing experiment results in automated or assisted calibration workflows.

## Output:

Types: Text. Formats: String

## Describe how the model works:

Experiment plot images are encoded into visual tokens and combined with prompt text tokens before being processed by the Gemma 4 31B dense language model. The model generates analytical text autoregressively through a NVIDIA NIM vLLM-served OpenAI-compatible API.

## Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:

Not Applicable

## Technical Limitations and Mitigation:

The model is domain-specific to quantum calibration experiments and may not generalize to broader VLM tasks. Performance varies by question type, with weaker results on experimental significance and parameter extraction than on fit quality assessment. Outputs should be validated by domain experts before being used in experimental workflows.

## Verified to have met prescribed NVIDIA quality standards:

Yes

## Performance Metrics:

QCalEval zero-shot overall 74.5; QCalEval MM-ICL overall 81.2; throughput and latency were measured for the BF16 NIM release candidate on NVIDIA GPU-accelerated systems.

## Potential Known Risks:

The model may misclassify rare or ambiguous experiment outcomes, may hallucinate details outside the quantum calibration domain, and does not have access to raw numerical traces or experiment metadata beyond what is visible in the input plots.

## License/Terms of Use

GOVERNING TERMS: Use of this model is governed by the [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/). ADDITIONAL INFORMATION: Apache License, Version 2.0.
