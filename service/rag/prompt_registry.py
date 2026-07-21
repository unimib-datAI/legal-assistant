"""Versioned prompt registry.

Each prompt is stored as a :class:`PromptVersion` carrying its text, a version
label, a creation date, a changelog note, and an ``active`` flag. This keeps
prompt evolution tracked in-repo: the changelog lives next to the prompt, a
rollback is a single flag flip, and the active version can be logged so any
generated answer is traceable to the prompt that produced it.

Invariant: exactly one version per name must be marked ``active`` — violations
fail fast at import time via :meth:`PromptRegistry.active`.
"""

from dataclasses import dataclass
from datetime import date
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class PromptVersion(Generic[T]):
    """A single immutable version of a named prompt.

    ``body`` is generic: ``str`` for text prompts, ``dict[str, str]`` for prompt
    maps such as the per-query-type synthesis guidance.
    """

    name: str
    version: str
    created: date
    notes: str
    body: T
    active: bool = False


class PromptRegistry:
    """In-memory store of prompt versions keyed by prompt name."""

    def __init__(self) -> None:
        self._versions: dict[str, list[PromptVersion]] = {}

    def register(self, prompt: PromptVersion) -> None:
        """Register a version. Rejects duplicate (name, version) pairs."""
        bucket = self._versions.setdefault(prompt.name, [])
        if any(v.version == prompt.version for v in bucket):
            raise ValueError(
                f"Prompt '{prompt.name}' already has a version '{prompt.version}'"
            )
        bucket.append(prompt)

    def active(self, name: str) -> PromptVersion:
        """Return the single active version for ``name``.

        Raises ``ValueError`` if the name is unknown or the active count is not
        exactly one.
        """
        bucket = self._versions.get(name)
        if not bucket:
            raise ValueError(f"No prompt registered under name '{name}'")
        actives = [v for v in bucket if v.active]
        if len(actives) != 1:
            raise ValueError(
                f"Prompt '{name}' must have exactly one active version, "
                f"found {len(actives)}"
            )
        return actives[0]

    def get(self, name: str, version: str) -> PromptVersion:
        """Return a specific version of ``name``, active or not.

        Lets an eval pin a non-active prompt for an A/B run without flipping the
        ``active`` flag, which would change the prompt for every other caller.
        """
        bucket = self._versions.get(name)
        if not bucket:
            raise ValueError(f"No prompt registered under name '{name}'")
        for candidate in bucket:
            if candidate.version == version:
                return candidate
        known = ", ".join(v.version for v in bucket)
        raise ValueError(
            f"Prompt '{name}' has no version '{version}'; known: {known}"
        )

    def versions(self, name: str) -> list[PromptVersion]:
        """Return all registered versions for ``name`` (registration order)."""
        bucket = self._versions.get(name)
        if not bucket:
            raise ValueError(f"No prompt registered under name '{name}'")
        return list(bucket)

    def names(self) -> list[str]:
        """Return all registered prompt names."""
        return list(self._versions)

    def active_versions(self) -> dict[str, str]:
        """Return ``name -> active version`` for every registered prompt.

        Useful for logging which prompt versions a run is using.
        """
        return {name: self.active(name).version for name in self._versions}
