def test_evaluator_tiny_cpu():
    from jet.eval import Evaluator
    ev = Evaluator("sshleifer/tiny-gpt2", do_sample=False, max_new_tokens=4)
    out = ev.evaluate(["Hello"], references=None, perplexity_texts=["Hello world."])
    assert out["count"] == 1
