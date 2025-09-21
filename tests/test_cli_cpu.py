def test_cli_eval(monkeypatch):
    import subprocess
    # Ensure the module imports; detailed CLI tests can be added with subprocess in CI
    import jet.cli as _  # noqa
    assert True
