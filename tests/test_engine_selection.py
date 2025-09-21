def test_choose_engine_cpu():
    from jet.train import choose_engine
    assert choose_engine("auto") in {"hf","unsloth"}
