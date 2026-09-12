"""Shared test helpers."""


def unwrapped(text: str) -> str:
    """Console text with ALL whitespace removed, for substring assertions.

        `rich` wraps to the terminal width, and it does two different things:
        it soft-wraps at spaces, and it HARD-BREAKS a token longer than the
        width. Collapsing runs of whitespace to single spaces survives the first
        and fails the second - `--frame-ms` arrives as `--frame
    -ms` and becomes
        `--frame -ms`, which still does not match.

        Removing whitespace entirely survives both, and still requires every
        character of the needle in order. Compare BOTH sides with this.

        This has now been the cause of CI failures three times: a macOS tmp path
        broken mid-segment, a Windows `rekindle semantic
    embed`, and `--frame-ms`
        in a narrow help table. Local runs pass because the developer's terminal
        is wide.
    """
    return "".join(text.split())
