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

# And no colour. `rich` styles an option name in SEGMENTS - `--out` becomes
# `-` ESC `-out` ESC - so `"--frame-ms" in output` is false in coloured output
# however the whitespace is handled. That is what turned CI red while every
# local run was green: typer colours help unconditionally, and a local
# CliRunner is not a terminal.
#
# No test asserts anything ABOUT colour. Turning it off here removes the whole
# failure class rather than hardening one assertion at a time, and it makes a
# developer's `FORCE_COLOR` as harmless as their `COLUMNS`.
#
# All three lines below are deliberately redundant - measured: removing any
# ONE of them changes nothing, removing all three fails. They cover different
# libraries reading different variables, and the cost of keeping the belt as
# well as the braces is three lines.
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ.pop("FORCE_COLOR", None)

# And no API key. A developer's own `OPENAI_API_KEY` is visible to the test
# process, so any code path that asks "is the model available?" answers
# differently here than in CI - the LLM branch runs, tags get cached, and a
# test asserting the no-model message fails on the one machine that has a key.
#
# This is the mirror of the extras problem that kept CI red for ten pushes:
# there, the dev machine had something CI lacked and tests passed locally;
# here it fails locally and passes in CI. Both are the same defect - a result
# that depends on the machine.
#
# A test that WANTS the model path must set the key itself, explicitly.
os.environ.pop("OPENAI_API_KEY", None)
