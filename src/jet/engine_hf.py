import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig
from .decoding import make_generation_config

def _normalize_precision():
    cuda = torch.cuda.is_available()
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if not (cuda or xpu):
        return torch.float32, False, False
    if cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True

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

def train(opts, train_ds, eval_ds=None):
    set_seed(getattr(opts, "seed", 42))
    dtype, use_bf16, use_fp16 = _normalize_precision()
    tok = AutoTokenizer.from_pretrained(opts.model, use_fast=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = getattr(tok, "eos_token", None)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(opts.model, torch_dtype=dtype)
    gen_cfg = make_generation_config(tok, opts)
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
