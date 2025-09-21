import torch
from transformers import set_seed, GenerationConfig
from trl import SFTTrainer, SFTConfig

class FastLanguageModel:
    _placeholder = True
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise RuntimeError("UNSLOTH_PLACEHOLDER")
    @staticmethod
    def get_peft_model(model, **kwargs):
        raise RuntimeError("UNSLOTH_PLACEHOLDER")

def _normalize_precision(opts):
    cuda = torch.cuda.is_available()
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if not (cuda or xpu):
        return torch.float32, False, False, False
    if cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False, getattr(opts, "use_4bit", True)
    return torch.float16, False, True, getattr(opts, "use_4bit", True)

def _import_unsloth():
    from unsloth import FastLanguageModel as _RealFastLanguageModel
    return _RealFastLanguageModel

def _ensure_unsloth_loaded():
    global FastLanguageModel
    if getattr(FastLanguageModel, "_placeholder", False):
        FastLanguageModel = _import_unsloth()

def _build_sft_config(opts, use_bf16, use_fp16, include_field=True):
    base = dict(
        output_dir=opts.output_dir,
        per_device_train_batch_size=opts.per_device_batch,
        gradient_accumulation_steps=opts.grad_accum,
        learning_rate=opts.lr,
        num_train_epochs=opts.epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
        packing=False,
    )
    if include_field:
        base["dataset_text_field"] = opts.text_field or "text"
    try:
        return SFTConfig(**base), True
    except TypeError:
        if "dataset_text_field" in base:
            base.pop("dataset_text_field", None)
            return SFTConfig(**base), False
        raise

def _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, dataset_field_name):
    kwargs = dict(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=sft_cfg)
    try:
        return SFTTrainer(tokenizer=tok, **kwargs)
    except TypeError:
        pass
    if dataset_field_name:
        try:
            return SFTTrainer(processing_class=tok, dataset_text_field=dataset_field_name, **kwargs)
        except TypeError:
            pass
    return SFTTrainer(processing_class=tok, **kwargs)

def _safe_int_id(v):
    return v if isinstance(v, int) else None

def train(opts, train_ds, eval_ds=None):
    set_seed(getattr(opts, "seed", 42))
    dtype, use_bf16, use_fp16, use_4bit = _normalize_precision(opts)
    try:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=opts.model,
            max_seq_length=opts.max_seq,
            load_in_4bit=use_4bit,
            dtype=dtype,
            device_map="auto",
        )
    except RuntimeError as e:
        if "UNSLOTH_PLACEHOLDER" not in str(e):
            raise
        _ensure_unsloth_loaded()
        model, tok = FastLanguageModel.from_pretrained(
            model_name=opts.model,
            max_seq_length=opts.max_seq,
            load_in_4bit=use_4bit,
            dtype=dtype,
            device_map="auto",
        )

    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = getattr(tok, "eos_token", None) or "<|pad|>"
    tok.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"

    pad_id = _safe_int_id(getattr(tok, "pad_token_id", None)) or _safe_int_id(getattr(tok, "eos_token_id", None))
    eos_id = _safe_int_id(getattr(tok, "eos_token_id", None))
    gen_cfg = GenerationConfig(
        do_sample=getattr(opts, "do_sample", True),
        temperature=getattr(opts, "temperature", 0.7),
        top_p=getattr(opts, "top_p", 0.9),
        top_k=getattr(opts, "top_k", 50),
        repetition_penalty=getattr(opts, "repetition_penalty", 1.1),
        no_repeat_ngram_size=getattr(opts, "no_repeat_ngram_size", 2),
        max_new_tokens=getattr(opts, "max_new_tokens", 64),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    try:
        model.generation_config = gen_cfg
    except Exception:
        pass

    sft_cfg, cfg_has_field = _build_sft_config(opts, use_bf16, use_fp16, include_field=True)
    dataset_field_name = opts.text_field or "text"
    trainer = _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, None if cfg_has_field else dataset_field_name)

    trainer.train()
    trainer.save_model(opts.output_dir)
    tok.save_pretrained(opts.output_dir)
    try:
        gen_cfg.save_pretrained(opts.output_dir)
    except Exception:
        pass

    return type("Job", (), {"model_dir": opts.output_dir})
