import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .metrics import compute_rouge, compute_perplexity

def _resolve_model_source(path_or_id: str, allow_fallback: bool = True, fallback_model_id: str = "sshleifer/tiny-gpt2") -> str:
    if os.path.isdir(path_or_id):
        cfg = os.path.join(path_or_id, "config.json")
        if os.path.exists(cfg):
            return path_or_id
        if allow_fallback:
            return fallback_model_id
        raise ValueError(f"Invalid model directory '{path_or_id}': missing config.json.")
    return path_or_id

def _safe_int(v) -> Optional[int]:
    return v if isinstance(v, int) else None

def _build_gen_kwargs(tok, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {
        "do_sample": overrides.pop("do_sample", True),
        "temperature": overrides.pop("temperature", 0.7),
        "top_p": overrides.pop("top_p", 0.9),
        "top_k": overrides.pop("top_k", 50),
        "repetition_penalty": overrides.pop("repetition_penalty", 1.1),
        "no_repeat_ngram_size": overrides.pop("no_repeat_ngram_size", 2),
        "max_new_tokens": overrides.pop("max_new_tokens", 64),
    }
    pad_id = _safe_int(getattr(tok, "pad_token_id", None)) or _safe_int(getattr(tok, "eos_token_id", None))
    eos_id = _safe_int(getattr(tok, "eos_token_id", None))
    if pad_id is not None:
        cfg["pad_token_id"] = pad_id
    if eos_id is not None:
        cfg["eos_token_id"] = eos_id
    for k, v in list(overrides.items()):
        cfg[k] = v
    return {k: v for k, v in cfg.items() if v is not None}

class Evaluator:
    def __init__(self, model_dir_or_id: str, allow_fallback: bool = True, fallback_model_id: str = "sshleifer/tiny-gpt2", generation_config=None, **gen_overrides):
        src = _resolve_model_source(model_dir_or_id, allow_fallback=allow_fallback, fallback_model_id=fallback_model_id)
        self.tok = AutoTokenizer.from_pretrained(src, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(src)
        try:
            self.model.eval()
        except Exception:
            pass
        if generation_config is not None:
            try:
                self.model.generation_config = generation_config
            except Exception:
                pass
        self.gen_kwargs = _build_gen_kwargs(self.tok, gen_overrides)

    def evaluate(self, prompts: List[str], references: Optional[List[str]] = None, perplexity_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        preds: List[str] = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt")
            try:
                out = self.model.generate(**enc, **self.gen_kwargs)
            except TypeError:
                out = self.model.generate(**enc, max_new_tokens=self.gen_kwargs.get("max_new_tokens", 64))
            first = out[0] if isinstance(out, (list, tuple)) else out[0]
            txt = self.tok.decode(first, skip_special_tokens=True)
            preds.append(txt)
        metrics: Dict[str, Any] = {}
        if references is not None and len(references) == len(preds):
            metrics.update(compute_rouge(preds, references))
        if perplexity_texts:
            try:
                metrics.update(compute_perplexity(perplexity_texts, self.model, self.tok))
            except Exception:
                pass
        return {"count": len(preds), "preds": preds, "refs": references or [], "metrics": metrics}
