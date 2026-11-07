# This file makes the `harness` directory a Python package.
# A package is just a folder with an __init__.py file; it lets us do
# imports like `from harness.helpdesk import HelpdeskHarness`.
#
# We re-export the main class here so users can write:
#     from harness import HelpdeskHarness
# instead of reaching into the submodule directly.

from .helpdesk import HelpdeskHarness

__all__ = ["HelpdeskHarness"]
