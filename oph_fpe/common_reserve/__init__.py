"""Fail-closed common-reserve audit surfaces.

Only the CR-0 capability inventory lives in this package. Reserve, cocycle,
screen, comparison, and certification producers are deliberately absent until
their prerequisites have been reviewed. Import the ``capability`` submodule
explicitly so command-line execution does not pre-import its producer.
"""
