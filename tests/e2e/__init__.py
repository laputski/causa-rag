"""The end-to-end walkthrough of the platform over the demo realm.

This is a package, not just a directory. `conftest.py` takes its cleanup from a
neighbouring module, and a relative import with no `__init__.py` cannot find its
parent, so the whole suite failed at collection. Every other test directory was
already a package; this one lagged.
"""
