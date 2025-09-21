from typing import List, Dict, Any, Optional

def compute_rouge(preds: List[str], refs: List[Any]) -> Dict[str, float]:
    import evaluate
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=preds, references=refs)

def compute_perplexity(texts: List[str], model, tok, stride: int = 512, max_len: Optional[int] = None) -> Dict[str, float]:
    import math, torch
    device = next(model.parameters()).device
    enc = tok("\n\n".join(texts), return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if max_len is None:
        max_len = getattr(model.config, "max_position_embeddings", 1024)
    nlls, seq_len = [], 0
    for i in range(0, input_ids.size(1), stride):
        begin = max(i + stride - max_len, 0)
        end = min(i + stride, input_ids.size(1))
        trg_len = end - i
        if trg_len <= 0:
            continue
        ids_slice = input_ids[:, begin:end]
        target_ids = ids_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(input_ids=ids_slice, labels=target_ids).loss
        nlls.append(loss.float() * trg_len)
        seq_len += trg_len
    ppl = math.exp((sum(nlls) / seq_len).item())
    return {"perplexity": ppl}
