"""
Filesystem MCP server — Version 3.

Exposes study notes as MCP tools/resources. In this project we import the
tool functions directly into agents (dev mode). Running `python -m` /
`mcp.run()` would serve them as a standalone MCP process.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Filesystem Server")

NOTES_BASE = Path(os.getenv("NOTES_PATH", "study_materials/sample_notes"))


@mcp.tool()
def list_study_files() -> list[str]:
    """
    List all available study note files.

    Returns a list of filenames relative to the notes directory.
    Example: ['closures.md', 'decorators.md', 'python_basics.md']

    Always call this first to discover what materials are available
    before attempting to read specific files.
    """
    if not NOTES_BASE.exists():
        return []
    return sorted(
        str(f.relative_to(NOTES_BASE)) for f in NOTES_BASE.rglob("*.md")
    )


@mcp.tool()
def read_study_file(filename: str) -> str:
    """
    Read the full content of a study note file.

    Args:
        filename: The filename to read, exactly as returned by
                  list_study_files(). Example: 'closures.md'

    Returns the full text content, or an error string if not found.
    Never raises. Errors are returned as strings so the agent
    can handle them gracefully.
    """
    file_path = NOTES_BASE / filename

    try:
        resolved = file_path.resolve()
        resolved.relative_to(NOTES_BASE.resolve())
    except ValueError:
        return (
            f"Error: path traversal attempt blocked for '{filename}'. "
            "Only files within the notes directory are accessible."
        )

    if not file_path.exists():
        available = list_study_files()
        return f"Error: '{filename}' not found. Available: {available}"

    if file_path.suffix != ".md":
        return f"Error: only .md files are accessible, got '{file_path.suffix}'"

    try:
        return file_path.read_text(encoding="utf-8")
    except (PermissionError, OSError) as e:
        return f"Error reading '{filename}': {e}"


@mcp.tool()
def search_notes(query: str) -> list[dict]:
    """
    Search across all study notes for a keyword or phrase.

    Args:
        query: The search term. Case-insensitive substring match.

    Returns a list of matches, each with keys: 'file', 'line_number', 'line'.
    Maximum 20 results to avoid overwhelming the context window.
    """
    if not NOTES_BASE.exists():
        return []

    results: list[dict] = []
    query_lower = query.lower()

    for file_path in sorted(NOTES_BASE.rglob("*.md")):
        rel_path = str(file_path.relative_to(NOTES_BASE))
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        for line_num, line in enumerate(lines, 1):
            if query_lower in line.lower():
                results.append(
                    {
                        "file": rel_path,
                        "line_number": line_num,
                        "line": line.strip(),
                    }
                )
                if len(results) >= 20:
                    return results

    return results


@mcp.resource("notes://index")
def get_notes_index() -> str:
    """
    Resource: index of all available study materials with file sizes.
    URI: notes://index
    """
    files = list_study_files()
    if not files:
        return "# Study Materials Index\n\nNo study materials found."

    lines = ["# Study Materials Index\n"]
    for filename in files:
        file_path = NOTES_BASE / filename
        try:
            size_kb = file_path.stat().st_size / 1024
            lines.append(f"- **{filename}** ({size_kb:.1f} KB)")
        except OSError:
            lines.append(f"- **{filename}** (size unknown)")
    lines.append(f"\nTotal: {len(files)} file(s)")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[Filesystem MCP] Starting server")
    print(f"[Filesystem MCP] Serving files from: {NOTES_BASE.resolve()}")
    mcp.run()
