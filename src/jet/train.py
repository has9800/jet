import importlib
import warnings

try:
    import mlflow
except Exception:
    mlflow = None

def _log_params(params: dict):
    if mlflow is not None:
        try: mlflow.log_params(params)
        except Exception: pass

def _log_dict(obj: dict, name: str):
    if mlflow is not None:
        try: mlflow.log_dict(obj, name)
        except Exception: pass

def _gpu_env_snapshot():
    snap = {}
    try:
        import torch
        snap["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if snap["cuda_available"]:
            try:
                major, minor = torch.cuda.get_device_capability(0)
                snap["compute_capability"] = f"{major}.{minor}"
            except Exception:
                snap["compute_capability"] = "unknown"
            try:
                snap["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                snap["bf16_supported"] = None
        if hasattr(torch, "xpu"):
            try: snap["xpu_available"] = bool(torch.xpu.is_available())
            except Exception: snap["xpu_available"] = False
        try:
            import torch as _t
            snap["torch_cuda_build"] = getattr(_t.version, "cuda", None)
        except Exception:
            snap["torch_cuda_build"] = None
    except Exception:
        snap["error"] = "torch not importable"
    return snap

def _unsloth_importable() -> bool:
    try:
        importlib.import_module("unsloth")
        return True
    except Exception:
        return False

def _gpu_capability_ok() -> bool:
    try:
        import torch
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            return False
        major, minor = torch.cuda.get_device_capability(0)
        return major >= 7
    except Exception:
        return False

def choose_engine(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if _gpu_capability_ok() and _unsloth_importable():
        return "unsloth"
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available() and _unsloth_importable():
            return "unsloth"
    except Exception:
        pass
    return "hf"

def has_supported_gpu() -> bool:
    try:
        import torch
        return (hasattr(torch, "cuda") and torch.cuda.is_available()) or (hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:
        return False

def train_with_options(opts, train_ds, eval_ds=None):
    user_pref = getattr(opts, "engine", "auto")
    engine = choose_engine(user_pref)
    snap = _gpu_env_snapshot()
    _log_params({"engine_requested": user_pref, "engine_selected": engine})
    _log_dict(snap, "gpu_env.json")
    if engine == "unsloth" and not _unsloth_importable():
        warnings.warn("Unsloth not importable; falling back to HF.")
        engine = "hf"
    if engine == "unsloth" and not _gpu_capability_ok():
        warnings.warn("GPU capability insufficient; falling back to HF.")
        engine = "hf"
    if engine == "unsloth":
        from .engine_unsloth import train as engine_train
    else:
        from .engine_hf import train as engine_train
    job = engine_train(opts, train_ds, eval_ds)
    merged_dir = None
    if getattr(opts, "merge_weights", False):
        try:
            from .merge import merge_lora
            merged_dir = f"{opts.output_dir}-merged"
            merge_lora(opts.model, opts.output_dir, merged_dir)
        except Exception as e:
            _log_dict({"merge_error": str(e)}, "merge_error.json")
            return type("Job", (), {"model_dir": opts.output_dir, "merged_dir": None, "merge_error": str(e)})
    _log_dict({"model_dir": opts.output_dir, "merged_dir": merged_dir, "engine_used": engine}, "train_outputs.json")
    return type("Job", (), {"model_dir": opts.output_dir, "merged_dir": merged_dir})
