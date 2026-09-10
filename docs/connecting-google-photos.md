# Connecting your Google Photos library

> **TL;DR** — Google removed the API that could read your library. Google Takeout
> is now the only way to get your photos *with* their face tags, GPS and
> descriptions. Start the export first; it takes hours to days.

## Why not just use the API?

Google **removed** the `photoslibrary.readonly`, `photoslibrary.sharing` and
`photoslibrary` scopes on **1 April 2025**. Calls relying on them now return
`403 PERMISSION_DENIED`. There is no longer *any* scope that reads a user's
existing library.

What survives, and what it gives you:

| Route | Full library? | Face tags | GPS | Descriptions | Live? |
|---|---|---|---|---|---|
| **Takeout** | ✅ yes | ✅ names | ✅ yes | ✅ yes | ❌ manual export |
| **Picker API** | ❌ user picks | ❌ | ❌ | ❌ | ✅ live |
| **Library API** | ❌ app-created only | ❌ | ❌ | ❌ | ✅ live |

Google has **never** exposed face groupings or content labels through any API,
before or after the shutdown. Takeout is the only source for them.

So rekindle treats **Takeout as the backbone** and the **Picker API as an
optional top-up** for photos taken since your last export.

---

## Step 1 — Start a Takeout export (do this first)

1. Go to **<https://takeout.google.com>**
2. Click **Deselect all**, then scroll down and tick **Google Photos** only.
   Exporting everything else wastes hours and disk.
3. *(Optional)* Click **All photo albums included** to limit which albums are
   exported. Leave it alone to get everything.
4. Click **Next step**.
5. Configure delivery — see the recommended settings below.
6. Click **Create export** and wait. Google emails you when it's ready.

### Recommended settings

| Setting | Choose | Why |
|---|---|---|
| Transfer to | **Add to Drive** | Lets rekindle pull new exports automatically later. Email links expire after a week. |
| Frequency | **Export every 2 months for 1 year** | Turns a one-off chore into a recurring feed. This is the closest thing to a real connector that still exists. |
| File type | **.tgz** (or **.zip** on Windows) | `.tgz` handles very large archives more reliably. `.zip` is easier to open on Windows. |
| File size | **50 GB** | **Important.** Smaller sizes split your export across many archives, and Google will happily put a photo in one archive and its metadata JSON in another. Fewer splits means fewer broken pairs. |

> **Why "every 2 months" matters:** a scheduled export landing in Drive means
> rekindle can detect and ingest new exports without you visiting Takeout again.
> One-time exports mean repeating this whole process by hand.

### How long does it take?

| Library size | Typical wait |
|---|---|
| < 5,000 photos | minutes to a few hours |
| 10,000 – 50,000 | several hours to ~1 day |
| 100,000+ | 1–3 days, split across many archives |

Google does not report progress. It just emails you when it's done.

---

## Step 2 — Unpack it

Extract **all** archives into a **single shared directory**. This matters: when
an export is split, a photo and its metadata sidecar can land in different
archives, and they only reunite if you extract them into the same tree.

```bash
mkdir -p ~/takeout
# Linux / macOS
for f in takeout-*.tgz; do tar -xzf "$f" -C ~/takeout; done
# Windows PowerShell
Get-ChildItem takeout-*.zip | ForEach-Object {
    Expand-Archive $_.FullName -DestinationPath "$HOME\takeout" -Force
}
```

You should end up with `~/takeout/Takeout/Google Photos/` containing a mix of:

- `Photos from 2014/` — year folders, where most photos live
- `Goa Trip/` — album folders (photos here are **duplicates** of the year
  folders; rekindle de-duplicates by content hash)
- `IMG_1234.JPG` alongside `IMG_1234.JPG.json` — the metadata sidecars

Point rekindle at it:

```bash
REKINDLE_TAKEOUT_DIR=~/takeout/Takeout/Google Photos
```

---

## Step 3 — Verify the export before indexing

Takeout exports are quirky and sometimes incomplete. Check yours before
spending an hour on embeddings:

```bash
rekindle doctor
```

It reports how many photos were found, how many matched a metadata sidecar,
how many carry face tags and GPS, and which known quirks it hit. Investigate
before indexing if sidecar coverage is below ~95%.

### Known Takeout quirks rekindle handles for you

| Quirk | What rekindle does |
|---|---|
| Sidecar named `IMG_1234.JPG.supplemental-metadata.json` | Tries all known naming variants |
| Long filenames truncated, so sidecar name ≠ image name | Truncation-aware matching |
| Duplicate counters: `IMG(1).JPG.json` vs `IMG.JPG(1).json` | Tries both permutations |
| `creationTime` is the **upload** time, not capture time | Always uses `photoTakenTime` |
| `-edited` files have no sidecar | Inherits from the original |
| Same photo in year folder *and* album folders | De-duplicates by content hash |
| `geoData` zeroed but `geoDataExif` populated | Falls back to `geoDataExif` |
| Timestamps are UTC epoch | Resolves local time from GPS |

### Known limitation

Takeout gives you **face-tag names but not face locations**. The JSON says
Alice is in the photo; it does not say where. rekindle therefore cannot do
face-aware cropping and uses visual saliency instead.

---

## Step 4 *(optional)* — Picker API top-up

For pulling in photos taken since your last export, without waiting for another
one. Picker photos carry far less metadata, so they're a supplement, not a
replacement.

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Photos Picker API**.
3. Configure the OAuth consent screen — **External**, and add yourself as a
   test user. You do *not* need to publish or get verified for personal use.
4. Create an **OAuth client ID** of type **Desktop app**.
5. Download the JSON and save it as `credentials.json` in your data directory.
   It is gitignored.
6. Run `rekindle sync --picker`. A browser opens, you pick photos, rekindle
   ingests them.

The scope requested is `photospicker.mediaitems.readonly` — it can only see
what you explicitly pick in that session, and nothing else in your library.

---

## Using something other than Google Photos

Takeout is just one `Source`. rekindle also reads plain folders:

```bash
REKINDLE_TAKEOUT_DIR= rekindle index --folder ~/Pictures
```

Adding a new source (Apple Photos, Immich, Nextcloud, Synology Photos) means
implementing one protocol — see [writing-sources.md](writing-sources.md).
Contributions very welcome.

---

## Privacy

Everything runs locally. Your photos are never uploaded anywhere by rekindle.

The one exception: if you enable OpenAI providers, images that reach a memory
are sent to OpenAI for captioning — **not** your whole library. Set
`REKINDLE_CAPTION_PROVIDER=local` to use a local vision model instead and keep
everything on your machine.
