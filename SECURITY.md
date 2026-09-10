# Security

rekindle handles private photographs, personal metadata, and optionally an API
key. This document says what it does with them.

## Reporting a vulnerability

Open a [security advisory](https://github.com/abhikatoldtrafford/rekindle/security/advisories/new).
Please don't file a public issue for anything exploitable.

## What leaves your machine

**By default, nothing.** The shipped configuration uses local models for
embedding and a template narrator that involves no model at all. rekindle makes
no network calls unless you change a provider.

If you opt into hosted providers:

| Provider setting | What is sent | To whom |
|---|---|---|
| `REKINDLE_CAPTION_PROVIDER=openai` | Image bytes for photos **that reach a memory** — never the whole library | OpenAI |
| `REKINDLE_NARRATE_PROVIDER=openai` | The `FactSheet`: dates, place names, tagged names, your descriptions | OpenAI |

`FactSheet` contents include people's names and your own captions. If that
matters to you, use `template` or a local model via `REKINDLE_LLM_BASE_URL`.

Set `REKINDLE_COST_CAP_USD` to bound spend per memory. `--dry-run` estimates
cost without sending anything.

## Secrets

Keys are read from the environment or `.env`, which is gitignored. rekindle
never writes a key to its index, logs, `MemorySpec` documents, or exports.

If you use the optional Picker API, `credentials.json` and the OAuth token cache
live in your data directory and are gitignored. Note that a Google OAuth client
in "Testing" status issues refresh tokens that **expire after 7 days**.

## Untrusted input

Photo descriptions, filenames and album names are **attacker-influenceable** —
anyone who has ever sent you an image contributed a filename, and shared albums
carry other people's text.

rekindle treats these as untrusted: they are delimited and length-capped before
reaching any model, screened for instruction-like content, and are **not
eligible to ground a factual claim**. Trusted fields are the machine-generated
ones — timestamps, GPS, dimensions, EXIF numerics.

This is defence in depth, not a proof. Prompt injection is an unsolved problem
industry-wide. Treat generated narration as untrusted output: it is shown to
you, never executed, and never used to make a decision on your behalf.

## Deleting your data

Everything rekindle creates lives under `REKINDLE_DATA_DIR` (default `./data`) —
index, thumbnails, memories, caches. Delete that directory and nothing remains.
rekindle never modifies your original photos.

## Telemetry

There is none, and there are no plans for any.
