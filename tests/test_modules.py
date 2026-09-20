import importlib


def test_runtime_independent_modules_import():
    for module_name in ("src.filter_search", "src.release_assets"):
        assert importlib.import_module(module_name)

