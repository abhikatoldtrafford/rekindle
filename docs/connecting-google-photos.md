# Exporting from Google Photos

> **You may not need this page.** rekindle v1 reads a **folder**. If your photos
> are already on disk, just point it at them — no export required.
>
> This page is for getting photos *out* of Google Photos. Once extracted, the
> result is an ordinary directory that rekindle indexes like any other, reading
> dates, GPS and camera data from EXIF.
>
> A dedicated Takeout parser that also reads Google's JSON sidecars — recovering
> face tags and descriptions that EXIF doesn't carry — is the next source
> planned. Everything below about Takeout's quirks applies to it.

> **TL;DR** — Google removed the API that could read your library. Takeout is
> the only way out, and it takes hours to days. Start it early.

## Why not just use the API?

Google **removed** the `photoslibrary.readonly`, `photoslibrary.sharing` and
`photoslibrary` scopes **after 31 March 2025**. Calls relying on them now return
`403 PERMISSION_DENIED`. There is no longer *any* scope that reads a user's
existing library.

What survives:

| Route | Coverage | Face tags | GPS | Descriptions | Live? |
|---|---|---|---|---|---|
| **Takeout** | everything **you** uploaded | ✅ names only | ✅ | ✅ | incremental, every 2 months |
| **Picker API** | only what you hand-pick | ❌ | ❌ | ❌ | ✅ |
| **Library API** | app-created only | ❌ | ❌ | ❌ | ✅ |

Two clarifications people get wrong:

- **Face groupings** were never exposed by any API. Still true.
- **Content labels** *were* exposed, but only as a search *filter* — 26 fixed
  categories, never returned per-item — and they died with the scopes. That's
  why rekindle rebuilds screenshot/receipt detection locally.

---

## What Takeout does *not* include

Read this before you assume a memory is broken.

**Photos other people added to shared albums are not exported.** Takeout gives
you what *you* uploaded. If twelve people contributed to a wedding album, you
get your own photos from it and nothing else — silently, with no error.

For a memories tool this stings, because group trips and celebrations are
exactly where other people's uploads concentrate. `rekindle doctor` flags albums
that look suspiciously sparse, but there is no way to fix this from Takeout.
To include those photos you must save them to your own library in Google Photos
first, then export.

**Face tags may not exist at all.** `people[]` comes from Google Photos face
grouping, which is opt-in, and which **Google does not offer in Illinois or
Texas** for legal reasons. If you're there, or never enabled it, person-based
memories won't work and person-based *exclusions* won't either. `doctor` reports
your coverage. Use date-range and album exclusions instead — they always work.

---

## Step 1 — Start a Takeout export (do this first)

1. Go to **<https://takeout.google.com>**
2. Click **Deselect all**, then tick **Google Photos** only.
3. *(Optional)* Click **All photo albums included** to limit which albums go out.
4. Click **Next step**.
5. Configure delivery — see below.
6. Click **Create export** and wait. Google emails you when it's ready.

### Recommended settings

| Setting | Choose | Why |
|---|---|---|
| Transfer to | **Send download link via email** | See the Drive warning below. |
| Frequency | **Export every 2 months for 1 year** | Since June 2026 these are *incremental* — see below. |
| File type | **.tgz**, or **.zip** on Windows | `.tgz` handles very large archives more reliably. |
| File size | **50 GB** | Fewer splits means fewer photos separated from their metadata JSON. |

> **⚠️ Don't pick "Add to Drive" unless you have the space.** Takeout archives
> saved to Drive **count against your Google storage quota**. A 40,000-photo
> library is commonly 150–400 GB; most accounts have 15 GB. The export will
> either fail or quietly consume storage you're paying for.

### Scheduled exports are incremental

Since **June 2026**, scheduling Photos exports gives you a real incremental feed:
the **first** archive is your full library, and each **subsequent** archive
contains only items uploaded, backed up, created or edited since the last
successful export. Up to six exports, one every two months.

This is the closest thing to a live connector that still exists, and rekindle is
built around it — it merges each delta into your existing index.

Two caveats:

- **Deletions can't be expressed by an incremental export.** If you delete a
  photo in Google Photos, nothing tells rekindle. It keeps what it has and
  records when each photo was last seen. Only a fresh full export can reconcile.
- **Scheduled exports aren't available to Advanced Protection Program users.**

### How long does it take?

| Library size | Typical wait |
|---|---|
| < 5,000 photos | minutes to a few hours |
| 10,000 – 50,000 | several hours to ~1 day |
| 100,000+ | 1–3 days, split across many archives |

Google reports no progress. It just emails you when it's done.

---

## Step 2 — Unpack it

Extract **all** archives into a **single shared directory**. When an export is
split, a photo and its metadata sidecar can land in different archives, and they
only reunite if you extract them into the same tree.

```bash
mkdir -p ~/takeout

# Linux / macOS
for f in takeout-*.tgz; do tar -xzf "$f" -C ~/takeout; done
```

```powershell
# Windows — use tar (built into Windows 10+), NOT Expand-Archive.
# Expand-Archive fails on archives over ~2 GB with "Stream was too long".
New-Item -ItemType Directory -Force "$HOME\takeout" | Out-Null
Get-ChildItem takeout-*.zip | ForEach-Object {
    tar -xf $_.FullName -C "$HOME\takeout"
}
```

You should end up with `~/takeout/Takeout/Google Photos/` containing:

- `Photos from 2014/` — year folders, where most photos live.
  **Note:** these are localized — you may see `Fotos von 2014` or similar.
- `Goa Trip/` — album folders. Photos here duplicate the year folders;
  rekindle merges them.
- `IMG_1234.JPG` alongside `IMG_1234.JPG.json` — the metadata sidecars.

Point rekindle at it, quoting the path (it contains a space):

```bash
export REKINDLE_TAKEOUT_DIR="$HOME/takeout/Takeout/Google Photos"
```

---

## Step 3 — Verify before indexing

Takeout exports are quirky and sometimes incomplete. Check yours before spending
an hour on embeddings:

```bash
rekindle doctor
```

`doctor` reports photo counts, **sidecar match quality broken down by tier**,
face-tag and GPS coverage, suspected sparse albums, media types found, and any
quirks it hit.

### Why match tiers matter

rekindle matches each photo to its metadata sidecar through several strategies,
from exact filename match down to heuristics. **It records which one was used**,
because low-confidence matches are a real corruption risk — a well-known Takeout
tool shipped aggressive fuzzy matching and produced roughly **a third of its GPS
data pointing at the wrong place**.

rekindle therefore **rejects heuristic-tier matches by default**, and never lets
narration assert a place or date that came from one. If `doctor` shows a lot of
low-tier matches, investigate rather than pushing through.

### Known quirks rekindle handles

| Quirk | Handling |
|---|---|
| `IMG_1234.JPG.supplemental-metadata.json` | All known naming variants |
| Suffix itself truncated: `.supplemental-metad.json`, `.s.json` | Combinatorial matcher, ~46-char cap |
| Long filenames truncated so sidecar ≠ image name | Truncation-aware matching |
| `IMG(1).JPG.json` vs `IMG.JPG(1).json` | Both counter positions |
| `creationTime` is *upload* time | Always uses `photoTakenTime` |
| `-edited` files | Linked to the original; the edit is preferred |
| Same photo in year and album folders | Merged, albums unioned |
| `geoData` zeroed, `geoDataExif` populated | Falls back to `geoDataExif` |
| Localized folder names | Never keys on English strings |
| `.heic` files that are actually JPEG | Sniffs magic bytes, ignores extension |
| Videos, motion photos, Live Photo pairs | Parsed and indexed, not montaged in v1 |

### Timestamps

Local time is resolved in this order: EXIF `OffsetTimeOriginal` → EXIF
`DateTimeOriginal` → GPS timezone lookup → UTC. Google's re-compression
sometimes strips EXIF, so `doctor` reports which source was used.

### Windows long paths

Deep album names plus long sidecar names routinely exceed Windows' 260-character
limit. If `doctor` warns about this, enable long paths:

```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

---

## Using something other than Google Photos

Takeout is just one `Source`. rekindle also reads plain folders:

```bash
rekindle index --folder ~/Pictures
```

Adding a source (Apple Photos, Immich, Nextcloud, Synology Photos) means
implementing one protocol — see [writing-sources.md](writing-sources.md).

---

## Privacy

Everything runs locally, and the default configuration makes **no network calls
at all** — embeddings, junk filtering and narration all run on your machine.

If you opt into the OpenAI providers, only photos that reach a memory are sent
for captioning — never your whole library. See [SECURITY.md](../SECURITY.md).
