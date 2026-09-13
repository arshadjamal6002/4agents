"""
Minimal stub so `langfuse.langchain` can `import langchain` without installing
the full langchain meta-package (which conflicts with our pinned langgraph).

Langfuse only needs this package to exist and report a v1-style __version__;
callbacks then import from langchain_core.
"""

__version__ = "1.0.0-stub"
