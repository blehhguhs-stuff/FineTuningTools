import modal
import os

# Modal App Configuration
stub = modal.App("mistral-7b-finetune-unsloth")

# Modal Image Configuration
image = modal.Image.debian_slim(python_version="3.10") \
    .apt_install("git") \
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
def finetune():
    import torch
    from huggingface_hub import login
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Log in to Hugging Face
    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])

    # Load the model and tokenizer first to apply the chat template
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mistralai/mistral-7b-instruct-v0.3",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    def format_prompt(example):
        messages = example.get("messages", [])
        if messages:
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                return {"text": text}
            except Exception as e:
                print(f"Error formatting prompt: {e}")
                return {"text": ""}
        else:
            return {"text": ""}

    # Load and format the dataset
    dataset = load_dataset("Guilherme34/uncensor", split="train")
    dataset = dataset.map(format_prompt, batched=False)  # Process one example at a time
    # Filter out empty texts
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Configure the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Start training
    trainer.train()

    # Push the model to the Hugging Face Hub
    model.push_to_hub("realoperator42/mistral-7b-instruct-v0.3-uncensored", token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    print("Model pushed to Hugging Face Hub at realoperator42/mistral-7b-instruct-v0.3-uncensored")

# To run this script, execute the following command in your terminal:
# modal run finetune.py
