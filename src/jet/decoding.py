from transformers import GenerationConfig

def make_generation_config(tok, opts):
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
    eos_id = getattr(tok, "eos_token_id", None)
    return GenerationConfig(
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
