import modal
import os

# Modal App Configuration
stub = modal.App("gemma-gguf-converter")

# Modal Image Configuration
image = modal.Image.debian_slim(python_version="3.10") \
    .apt_install("git", "cmake", "libcurl4-openssl-dev") \
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch",
        "transformers",
        "datasets",
        "trl",
        "accelerate",
    )

@stub.function(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=18000,
)
def convert_to_gguf():
    from huggingface_hub import login, HfApi
    from unsloth import FastLanguageModel

    # Log in to Hugging Face
    token = os.environ["HUGGING_FACE_HUB_TOKEN"]
    login(token=token)

    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="realoperator42/gemma-3n-E4B-uncensored",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Convert to GGUF and save to a local directory
    model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")

    # Upload to Hugging Face Hub
    api = HfApi(token=token)
    repo_id = "realoperator42/gemma-3n-E4B-uncensored-GGUF"
    api.create_repo(repo_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj="gguf_model/unsloth.Q4_K_M.gguf",
        path_in_repo="gemma-3n-E4B-uncensored-Q4_K_M.gguf",
        repo_id=repo_id,
        repo_type="model",
    )
    print("Model converted and pushed to Hugging Face Hub at realoperator42/gemma-3n-E4B-uncensored-GGUF")

# To run this script, execute the following command in your terminal:
# modal run convert_to_gguf.py
