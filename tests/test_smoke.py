import rekindle


def test_package_exposes_version():
    assert isinstance(rekindle.__version__, str)
    assert rekindle.__version__
