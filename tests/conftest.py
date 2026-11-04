from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="run tests that call Azure (needs az login and configs/endpoints.env)")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls Azure services")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
