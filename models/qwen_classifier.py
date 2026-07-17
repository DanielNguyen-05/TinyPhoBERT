"""
models/qwen_classifier.py
Fine-tune Qwen2.5 (QLoRA) làm standalone sequence classifier cho
Vietnamese Hate Speech Detection.
"""
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training


def _build_bnb_config(quant_cfg: dict) -> BitsAndBytesConfig:
    compute_dtype_str = quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_str)
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_qwen_base_and_tokenizer(config: dict):
    model_cfg = config["model"]
    quant_cfg = config.get("quantization", {})
    model_name = model_cfg["name"]
    num_labels = model_cfg["num_labels"]
    use_4bit = quant_cfg.get("load_in_4bit", True)
    bnb_config = _build_bnb_config(quant_cfg) if use_4bit else None

    print(f"[Qwen] Loading {model_name} ({'4-bit QLoRA' if use_4bit else 'full precision'})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto" if use_4bit else None,
        dtype=torch.bfloat16 if not use_4bit else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if use_4bit:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

        model = prepare_model_for_kbit_training(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Base params: {n_params:,} | pad_token_id={tokenizer.pad_token_id}")
    return model, tokenizer


def apply_lora(base_model, config: dict):
    lora_cfg = config.get("lora", {})
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def build_qwen_classifier(config: dict):
    base_model, tokenizer = build_qwen_base_and_tokenizer(config)
    model = apply_lora(base_model, config)
    return model, tokenizer


def load_qwen_with_adapter(config: dict, adapter_path: str):
    base_model, tokenizer = build_qwen_base_and_tokenizer(config)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer