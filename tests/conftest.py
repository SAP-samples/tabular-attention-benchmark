def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Run tests for a single backend only: base, cudnn, fa2, fa3, fa4",
    )


def pytest_collection_modifyitems(config, items):
    backend = config.getoption("--backend")
    if backend is None:
        return  # run everything (skip-if guards still apply)

    selected = []
    deselected = []
    for item in items:
        # Check if this test item has the requested backend marker
        if item.get_closest_marker(backend):
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_configure(config):
    for name in ("base", "cudnn", "fa2", "fa3", "fa4", "sage", "vllm"):
        config.addinivalue_line("markers", f"{name}: tests for the {name} backend")
