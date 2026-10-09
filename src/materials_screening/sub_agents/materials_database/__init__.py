"""Unified Materials Project database sub-agent package.

Import ``create_spec`` from ``spec_factory`` explicitly. Keeping this package
initializer dependency-free prevents the service/model layer from loading the
agent runtime and creating a circular import.
"""
