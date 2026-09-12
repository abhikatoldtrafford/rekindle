"""Test-session setup.

Pins the console width every test sees.

`rich` renders to the terminal width it detects, and at a narrow width it does
three different things to a string an assertion is looking for: soft-wraps at
spaces, hard-breaks a token longer than the width, and TRUNCATES table cells
with an ellipsis. The third one cannot be undone by any amount of normalising -
the characters are simply not there.

That made whether a test passed a property of the machine running it. It has
cost four CI failures, each of which passed locally first: a macOS tmp path
broken mid-segment, a Windows `rekindle semantic\\nembed`, `--frame-ms` in a
narrow help table, and the same option name again when the whitespace was
handled but the colour was not.

Setting it here makes the rendering deterministic everywhere. It is not a
substitute for `helpers.unwrapped` - colour still segments an option name at
any width, and a genuinely long path can still wrap - so assertions on console
output should still go through that.

Set before any import that might construct a `Console` at module scope.
"""

import os

# Assigned, not `setdefault`: a developer's COLUMNS leaking in is exactly the
# non-determinism this removes.
os.environ["COLUMNS"] = "200"
os.environ["LINES"] = "50"
