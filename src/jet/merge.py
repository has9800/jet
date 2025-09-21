from transformers import AutoModelForCausalLM
from peft import PeftModel

def merge_lora(base_model_id: str, adapter_dir: str, out_dir: str, torch_dtype=None):
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch_dtype)
    peft = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft.merge_and_unload()
    merged.save_pretrained(out_dir)
    return out_dir
