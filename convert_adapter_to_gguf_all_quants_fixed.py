import os
import shutil
import json
import modal
try:
    import unsloth  # Prefer early import to enable patches if available locally
except Exception:
    unsloth = None

# Modal App Configuration
stub = modal.App("adapter-to-gguf-all-quants")

# Modal Image Configuration
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev", "ninja-build")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "huggingface_hub",
        "mistral_common",
        "sentencepiece",
        "gguf",
    )
)

# Comprehensive set spanning 2-bit to 16-bit exports
# Unsloth/llama.cpp naming is case-insensitive; we normalize to lower-case internally.
DEFAULT_QUANTS = [
    "f16",
    "q2_k",
    "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m",
    "q6_k",
    "q8_0",
]


def _basename(s: str) -> str:
    s = s.rstrip("/\\")
    return s.split("/")[-1].split("\\")[-1]


def dequantize_model_if_needed(model_path: str, output_path: str):
    """
    Check if model has quantized weights and dequantize them if needed.
    This handles models that were saved with quantization (e.g., 4-bit models).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        # Load model config to check if it's quantized
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Check if model has quantized weights (look for .absmax tensors)
        has_quantized_weights = False
        for name, param in model.named_parameters():
            if '.absmax' in name or '.quant_state' in name:
                has_quantized_weights = True
                break
        
        if has_quantized_weights:
            print("[INFO] Detected quantized model. Dequantizing weights...")
            
            # For 4-bit models, we need to dequantize
            # This is a simplified approach - in practice, you might need model-specific handling
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Save as float16 without quantization
            model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)
            
            # Clean up the saved files to remove quantization artifacts
            import glob
            safetensor_files = glob.glob(os.path.join(output_path, "*.safetensors"))
            
            from safetensors import safe_open, save_file
            for file in safetensor_files:
                # Remove quantization-related tensors from safetensor files
                with safe_open(file, framework="pt") as f:
                    keys = [k for k in f.keys() if ".absmax" not in k and ".quant_state" not in k]
                    tensors = {k: f.get_tensor(k) for k in keys}
                save_file(tensors, file)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"[WARN] Could not check/dequantize model: {e}")
        return False


@stub.function(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=18000,
)
def convert_adapter_to_gguf(
    base_model_id: str,
    adapter_id_or_path: str,
    output_dir: str = "gguf_out",
    repo_upload: str = "",
    quants: list[str] | None = None,
    max_seq_length: int = 4096,
    dtype: str = "fp16",  # "fp16" or "bf16"
):
    """
    Convert a base HF model + LoRA/adapter into multiple GGUF quantizations.

    Args:
        base_model_id: HF repo id or local path for the base model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        adapter_id_or_path: HF repo id or local path for the LoRA adapter to merge
        output_dir: Directory where GGUF files will be written
        repo_upload: Optional Hugging Face repo id to upload resulting GGUFs
        quants: List of quantization identifiers (use None to export DEFAULT_QUANTS)
        max_seq_length: Max sequence length for Unsloth loading
        dtype: Base model dtype for merge step ("fp16" or "bf16")
    """
    import unsloth
    from unsloth import FastLanguageModel
    import torch
    from huggingface_hub import login, HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    # Login to Hugging Face
    token = os.environ["HUGGING_FACE_HUB_TOKEN"]
    login(token=token)

    quants = [q.lower() for q in (quants or DEFAULT_QUANTS)]

    # Determine base model to use: auto-detect from adapter if available
    base_id = base_model_id
    try:
        peft_cfg = PeftConfig.from_pretrained(adapter_id_or_path)
        if getattr(peft_cfg, "base_model_name_or_path", None):
            print(f"[INFO] Adapter base detected: {peft_cfg.base_model_name_or_path}. Using this base for merge.")
            base_id = peft_cfg.base_model_name_or_path
    except Exception:
        pass

    # Check if base model is a quantized model (e.g., ends with -bnb-4bit)
    is_quantized_base = "-bnb-4bit" in base_id.lower() or "-4bit" in base_id.lower()
    
    if is_quantized_base:
        print(f"[INFO] Detected quantized base model. Will use Unsloth for proper handling.")
        
        # Use Unsloth to load and merge quantized models
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_id,
            max_seq_length=max_seq_length,
            dtype=torch.float16 if dtype.lower() in ("fp16", "float16") else torch.bfloat16,
            load_in_4bit=True,  # Important for 4-bit models
        )
        
        # Load adapter using PeftModel
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_id_or_path)
        
        # Merge and unload to get full model
        model = model.merge_and_unload()
        
        # Save merged model
        tmp_dir = "merged_hf_model"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Use Unsloth's save method which handles quantized models properly
        model.save_pretrained_merged(tmp_dir, tokenizer, save_method="merged_16bit")
        merged_src_dir = tmp_dir
        
    else:
        # Original logic for non-quantized models
        tokenizer = AutoTokenizer.from_pretrained(
            base_id,
            use_fast=False,
            trust_remote_code=True,
        )

        torch_dtype = torch.float16 if dtype.lower() in ("fp16", "float16") else torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )

        # Attach and merge LoRA/adapter
        merged_src_dir = None
        try:
            peft_model = PeftModel.from_pretrained(base_model, adapter_id_or_path)
            merged_model = peft_model.merge_and_unload()

            # Save merged HF model to disk
            tmp_dir = "merged_hf_model"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            merged_model.save_pretrained(tmp_dir, safe_serialization=True)
            tokenizer.save_pretrained(tmp_dir)
            merged_src_dir = tmp_dir

            # Free GPU memory
            try:
                del peft_model
                del base_model
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Could not load adapter as PEFT or merge failed; falling back to treating '{adapter_id_or_path}' as a full merged model. Reason: {e}")
            merged_src_dir = adapter_id_or_path
            try:
                del base_model
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Export across quantizations using updated llama.cpp build process
    import subprocess, sys

    os.makedirs(output_dir, exist_ok=True)
    base_name = _basename(base_id)
    adapter_name = _basename(adapter_id_or_path)
    prefix = f"{base_name}--{adapter_name}"

    results: dict[str, str] = {}

    # Prepare llama.cpp tools with CMake
    llama_dir = "/root/llama.cpp"
    try:
        if not os.path.exists(llama_dir):
            subprocess.run(["git", "clone", "--recursive", "https://github.com/ggerganov/llama.cpp", llama_dir], check=True)
        
        # Build with CMake (new build system)
        build_dir = os.path.join(llama_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        
        # Configure with CMake
        subprocess.run(
            ["cmake", "..", "-DGGML_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=build_dir,
            check=True
        )
        
        # Build
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=build_dir,
            check=True
        )
        
        # The binaries should now be in the build/bin directory
        bin_dir = os.path.join(build_dir, "bin")
        if not os.path.exists(bin_dir):
            # Some configurations put binaries directly in build dir
            bin_dir = build_dir
            
    except Exception as e:
        print(f"[ERROR] Failed to build llama.cpp with CMake: {e}")
        return results

    # Update paths to use the built binaries
    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    quantize_binary = os.path.join(bin_dir, "llama-quantize")
    if not os.path.exists(quantize_binary):
        quantize_binary = os.path.join(bin_dir, "quantize")  # Try alternative name

    # Convert merged HF model to a base F16 GGUF first
    tmp_base_dir = os.path.join(output_dir, "_base")
    os.makedirs(tmp_base_dir, exist_ok=True)
    f16_base_path = os.path.join(tmp_base_dir, f"{prefix}-F16-base.gguf")
    
    try:
        if not os.path.exists(f16_base_path):
            # For quantized models, we might need special handling
            if is_quantized_base:
                # Try using Unsloth's export if available
                try:
                    model.save_pretrained_gguf(f16_base_path, tokenizer, quantization_method="f16")
                    print(f"[OK] Exported F16 base using Unsloth -> {f16_base_path}")
                except Exception as e:
                    print(f"[WARN] Unsloth GGUF export failed, trying standard conversion: {e}")
                    subprocess.run(
                        ["python3", convert_script,
                         "--outfile", f16_base_path, "--outtype", "f16", merged_src_dir],
                        check=True
                    )
            else:
                subprocess.run(
                    ["python3", convert_script,
                     "--outfile", f16_base_path, "--outtype", "f16", merged_src_dir],
                    check=True
                )
            print(f"[OK] Created F16 base -> {f16_base_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] F16 base conversion failed: {e}")
        # Try to provide more details
        try:
            # Run again to capture stderr
            result = subprocess.run(
                ["python3", convert_script,
                 "--outfile", f16_base_path, "--outtype", "f16", merged_src_dir],
                capture_output=True, text=True
            )
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        except:
            pass
        return results

    # Copy F16 to output if requested
    if "f16" in quants:
        f16_out = os.path.join(output_dir, f"{prefix}-F16.gguf")
        shutil.copy2(f16_base_path, f16_out)
        results["f16"] = f16_out
        print(f"[OK] Exported F16 -> {f16_out}")

    # Quantize from the F16 base
    if not os.path.exists(quantize_binary):
        print(f"[ERROR] Quantize binary not found at {quantize_binary}")
        return results
        
    for q in [q for q in quants if q != "f16"]:
        try:
            q_upper = q.upper()
            out_path = os.path.join(output_dir, f"{prefix}-{q_upper}.gguf")
            subprocess.run(
                [quantize_binary, f16_base_path, out_path, q_upper],
                check=True
            )
            results[q] = out_path
            print(f"[OK] Exported {q_upper} -> {out_path}")
        except Exception as e:
            print(f"[WARN] Quantization failed for {q}: {e}")

    # Optional: upload artifacts to Hugging Face
    if repo_upload:
        api = HfApi(token=token)
        api.create_repo(repo_upload, repo_type="model", exist_ok=True)
        for q, path in results.items():
            try:
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=os.path.basename(path),
                    repo_id=repo_upload,
                    repo_type="model",
                )
                print(f"[OK] Uploaded {os.path.basename(path)} to {repo_upload}")
            except Exception as e:
                print(f"[WARN] Upload failed for {q}: {e}")

    print(json.dumps(results, indent=2))
    return results


# Local entrypoint to pass CLI args with `modal run`:
#   modal run convert_adapter_to_gguf_all_quants_fixed.py::main -- --base_model qwen/qwen3-8b --adapter realoperator42/qwen3-8b-uncensored --output_dir gguf_out/Qwen3-8B --quants ALL --dtype fp16
@stub.local_entrypoint()
def main(
    base_model: str,
    adapter: str,
    output_dir: str = "gguf_out",
    repo_upload: str = "",
    quants: str = "ALL",       # e.g. "q4_k_m,q5_k_m,f16" or "ALL"
    dtype: str = "fp16",       # "fp16" or "bf16"
    max_seq_length: int = 4096,
):
    # Parse quantization list from CLI
    if quants.upper() == "ALL":
        quants_list = None
    else:
        quants_list = [q.strip().lower() for q in quants.split(",") if q.strip()]

    convert_adapter_to_gguf.remote(
        base_model_id=base_model,
        adapter_id_or_path=adapter,
        output_dir=output_dir,
        repo_upload=repo_upload,
        quants=quants_list,
        max_seq_length=max_seq_length,
        dtype=dtype,
    )
