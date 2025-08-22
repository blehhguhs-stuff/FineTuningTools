# Unsloth Fine-Tuning and GGUF Conversion Scripts

This repository contains scripts for fine-tuning various large language models (LLMs) using Unsloth and converting the fine-tuned models to the GGUF format. All scripts are designed to be run using [Modal](https://modal.com/).

## Project Structure

The repository is organized into "Family" folders, each dedicated to a specific base model architecture.

```
.
├── GemmaFamily/
│   ├── Unsloth_gemma-3-4b.py
│   ├── convert_to_gguf_gemma-3-4b.py
│   ├── Unsloth_gemma-3n-E4B.py
│   └── convert_to_gguf_gemma-3n-E4B.py
├── LLaMAFamily/
│   ├── Unsloth_Llama3.1-8B.py
│   └── convert_to_gguf_Llama3.1-8B.py
├── MistralFamily/
│   ├── Unsloth_mistral-7b-instruct-v0.3.py
│   └── convert_to_gguf_mistral-7b-instruct-v0.3.py
├── Qwen3Family/
│   ├── Unsloth_qwen3-8b.py
│   ├── convert_to_gguf_qwen3-8b.py
│   ├── ... (other Qwen models)
└── YiFamily/
    ├── Unsloth_Yi-6B.py
    └── convert_to_gguf_Yi-6B.py
```

Each folder contains two types of scripts for each model variant:

1.  **`Unsloth_*.py`**: This script handles the fine-tuning process. It loads a base model from Hugging Face, fine-tunes it on the `WizardLMTeam/WizardLM_evol_instruct_70k` dataset, and pushes the resulting LoRA-adapted model to the Hugging Face Hub.
2.  **`convert_to_gguf_*.py`**: This script takes a fine-tuned model from the Hub, converts it to the GGUF format with `q4_k_m` quantization, and uploads the final GGUF file to a new repository on the Hub.

## Models Included

This collection includes scripts for the following models:

-   **GemmaFamily**:
    -   `google/gemma-3-4b`
    -   `google/gemma-3n-E4B`
-   **LLaMAFamily**:
    -   `meta-llama/Llama-3.1-8B`
-   **MistralFamily**:
    -   `mistralai/mistral-7b-instruct-v0.3`
-   **Qwen3Family**:
    -   `qwen/qwen3-8b`
    -   `Qwen/Qwen3-4B-Instruct`
    -   `Qwen/Qwen3-4B-Thinking-2507`
-   **YiFamily**:
    -   `01-ai/Yi-6B`

## Prerequisites

-   A [Modal](https://modal.com/) account.
-   A [Hugging Face](https://huggingface.co/) account.
-   A Hugging Face access token with `write` permissions stored as a Modal secret named `my-huggingface-secret`.

## How to Run

To run any script, use the `modal run` command from your terminal.

**1. Fine-Tune a Model:**
Navigate to the appropriate directory and run the fine-tuning script. For example, to fine-tune the Llama 3.1 8B model:
```bash
modal run LLaMAFamily/Unsloth_Llama3.1-8B.py
```

**2. Convert to GGUF:**
After the fine-tuning is complete and the model is uploaded to the Hub, run the corresponding conversion script:
```bash
modal run LLaMAFamily/convert_to_gguf_Llama3.1-8B.py
```

The scripts will print the URL of the final model repository on the Hugging Face Hub upon successful completion.

## Adapter -> GGUF (All Quantizations 2–16 bit)

Use the unified Modal script to merge a LoRA/adapter into a base model and export a full suite of GGUF quantizations.

File:
- convert_adapter_to_gguf_all_quants.py

Prerequisites:
- Modal secret named `my-huggingface-secret` containing `HUGGING_FACE_HUB_TOKEN`.

Supported quantizations:
- f16 (16-bit)
- q2_k
- q3_k_s, q3_k_m, q3_k_l
- q4_0, q4_1, q4_k_s, q4_k_m
- q5_0, q5_1, q5_k_s, q5_k_m
- q6_k
- q8_0

Example: Export every quant and do not upload
```bash
modal run convert_adapter_to_gguf_all_quants.py::main -- \
  --base_model qwen/qwen3-8b \
  --adapter realoperator42/qwen3-8b-uncensored \
  --output_dir gguf_out/Qwen3-8B \
  --quants ALL \
  --dtype fp16
```

Example: Export a subset and upload artifacts to HF
```bash
modal run convert_adapter_to_gguf_all_quants.py::main -- \
  --base_model qwen/qwen3-8b \
  --adapter realoperator42/qwen3-8b-uncensored \
  --output_dir gguf_out/Qwen3-8B \
  --quants q4_k_m,q5_k_m,f16 \
  --repo_upload your-username/qwen3-8b-all-gguf \
  --dtype bf16
```

Notes:
- The script merges the adapter into full weights using PEFT, saves a merged HF model, reloads via Unsloth, then exports GGUF per quant.
- Unsupported quantizations for a given family are skipped with warnings.
- Output files are named like `{base}--{adapter}-{QUANT}.gguf` in `output_dir`.

## Adapter -> GGUF (All Quantizations 2–16 bit) - Fixed

Use the unified Modal script to merge a LoRA/adapter into a base model and export a full suite of GGUF quantizations. This is a fixed version of the previous script.

File:
- convert_adapter_to_gguf_all_quants_fixed.py

Prerequisites:
- Modal secret named `my-huggingface-secret` containing `HUGGING_FACE_HUB_TOKEN`.

Supported quantizations:
- f16 (16-bit)
- q2_k
- q3_k_s, q3_k_m, q3_k_l
- q4_0, q4_1, q4_k_s, q4_k_m
- q5_0, q5_1, q5_k_s, q5_k_m
- q6_k
- q8_0

Example: Export every quant and do not upload
```bash
modal run convert_adapter_to_gguf_all_quants_fixed.py::main -- \
  --base_model qwen/qwen3-8b \
  --adapter realoperator42/qwen3-8b-uncensored \
  --output_dir gguf_out/Qwen3-8B \
  --quants ALL \
  --dtype fp16
```

Example: Export a subset and upload artifacts to HF
```bash
modal run convert_adapter_to_gguf_all_quants_fixed.py::main -- \
  --base_model qwen/qwen3-8b \
  --adapter realoperator42/qwen3-8b-uncensored \
  --output_dir gguf_out/Qwen3-8B \
  --quants q4_k_m,q5_k_m,f16 \
  --repo_upload your-username/qwen3-8b-all-gguf \
  --dtype bf16
```

Notes:
- The script merges the adapter into full weights using PEFT, saves a merged HF model, reloads via Unsloth, then exports GGUF per quant.
- Unsupported quantizations for a given family are skipped with warnings.
- Output files are named like `{base}--{adapter}-{QUANT}.gguf` in `output_dir`.
