# M3 audit — what was verified, what was measured, what was wrong

Every number here was produced by running something on this machine against
this library. Nothing is carried over from a brief, a spec or a model card
without being re-measured — that habit cost the previous milestone four wrong
figures, and it is recorded in
[the M1 decision log](decision-log-takeout-enrichment.md).

Companion documents: the
[design and implementation plan](superpowers/plans/2026-09-11-m3-semantic.md).

---

## 1. The machine

```
GPU     NVIDIA RTX A4000 · 16,376 MiB (nvidia-smi) / 16,375 MiB (torch) · compute 8.6
Driver  551.61            (a CUDA 12.4 driver)
CUDA    Toolkit 12.4, nvcc V12.4.99 on PATH
CPU     AMD Ryzen 5 5600GT · 6 cores / 12 threads
Disk    D: 669 GB free · C: 182 GB free
Python  3.12.14 in the project venv (system python is 3.13.3) · uv 0.12.7
```

**CUDA is genuinely in use, not silently falling back.** Verified by running a
kernel and measuring it, not by reading `torch.cuda.is_available()`:

| | |
|---|---|
| `torch.cuda.is_available()` | `True` |
| device 0 | `NVIDIA RTX A4000`, 16,375 MiB, compute 8.6 |
| 4096³ fp16 matmul, 30 iterations | **63.9 TFLOP/s** |
| same shape, fp32, on the CPU | **0.359 TFLOP/s** |

A 178× ratio is a tensor core doing the work. (A second reading taken while
the GPU was already busy with a benchmark gave 24.6 TFLOP/s — recorded so the
63.9 is not mistaken for a floor.)

## 2. Does anything actually download?

The brief flagged that `cdn-lfs.huggingface.co` does not resolve here, and
warned that model weights have historically come from it. **Both halves are
true, and the conclusion does not follow.**

| host | resolves |
|---|---|
| `pypi.org` | yes — 151.101.192.223 |
| `huggingface.co` | yes — 13.35.20.8 |
| `hf.co` | yes |
| **`cdn-lfs.huggingface.co`** | **NO** |
| `cdn-lfs-us-1.hf.co` | yes |
| `cas-bridge.xethub.hf.co` | yes |
| `transfer.xethub.hf.co` | yes |
| `download.pytorch.org` | yes |

Hugging Face has moved large-file serving to **Xet**. A `HEAD` on
`huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors`
redirects to `us.aws.cdn.hf.co/xet-bridge-us/...` and returns 200;
`cdn-lfs.huggingface.co` is the retired host. Checked for all four candidate
repositories before a line of model code was written.

**Confirmed by actually downloading: 17 files, 3,937 MB, all verified.**

## 3. torch: the trap PyPI sets on Windows

| wheel | size |
|---|---|
| `torch-2.14.0-cp312-cp312-manylinux_2_28_x86_64.whl` (PyPI) | 529 MB |
| `torch-2.14.0-cp312-cp312-win_amd64.whl` (PyPI) | **118 MB** |

118 MB is not a CUDA build. On Windows, `pip install torch` from PyPI gives a
CPU-only wheel whose only symptom is that embedding runs about 150× slower —
precisely the silent fallback the brief calls a bug. The CUDA wheels are on
PyTorch's own index, so `pyproject.toml` maps `torch` there with
`[tool.uv.sources]`, restricted by marker to win32/linux and behind
`explicit = true` so it cannot perturb any other package's resolution.

What that index actually offers for cp312/win_amd64:

| channel | newest |
|---|---|
| cu124 | 2.6.0 |
| cu126 | 2.14.0 (resolved) |
| cu128 | 2.9.1+ |
| cu129 | 2.9.0 |

**cu124 is a dead end**: it stops at torch 2.6.0 for this platform. cu126 was
chosen and **verified to work on a CUDA 12.4 driver** (551.61) — CUDA minor
version compatibility holds in practice, tested with both torch 2.9.1+cu126
and the 2.14.0+cu126 the project finally resolved.

## 4. Choosing the model, with evidence

Both candidates were downloaded and run over **the same 540 real photos** —
60 each from the nine albums that have ≥60 live images with exactly one
non-year-bucket album, sampled at seed 20260911. fp16, CUDA, batch 32.

| | **CLIP ViT-L/14** | SigLIP SO400M-14-384 |
|---|---|---|
| embedding dim | 768 | 1152 |
| weights on disk | 1.71 GB | 3.51 GB |
| **encode throughput (GPU only)** | **83.1 img/s** | 38.4 img/s |
| peak VRAM | **1,153 MB** | 2,440 MB |
| end-to-end (naive loader) | 6.11 img/s | 5.57 img/s |
| 1-NN same-album accuracy | 0.9556 | **0.9796** |
| k-means purity vs albums (k=9) | **0.7407** | 0.5852 |
| k-means NMI vs albums | **0.6859** | 0.6208 |
| public pre-built ONNX export | **yes** (`Xenova/…`) | **no** |
| public aesthetic head for its space | **yes** (LAION v2, 768-d) | **no** |
| licence | MIT (upstream openai/CLIP) | Apache-2.0 |

**Read the two quality numbers carefully — they disagree, and the
disagreement is the finding.** SigLIP wins 1-NN: its nearest neighbour is more
often from the same album, so its *local* neighbourhoods are better. CLIP wins
k-means purity and NMI by a wide margin: its *global* structure lines up
better with event boundaries. Scene clustering is a global-structure job, so
the metric that matters here favours CLIP.

Album labels are weak ground truth — they are *event* labels, not scene
labels, and absolute values mean little. The comparison on identical inputs
is what the table is for.

Qualitative text→image retrieval on the same 540 photos, top-5 albums per
query (both models produce sensible results; this is what a text path into
the library looks like):

| query | CLIP ViT-L/14 top-5 albums | SigLIP top-5 albums |
|---|---|---|
| "a photo of snow-covered mountains" | Kashmir ×3, Leh Ladakh ×2 | Kashmir ×4, Leh Ladakh |
| "a wedding ceremony" | Wedding_arnab_pics ×5 | Wedding_arnab_pics ×5 |
| "a plate of food on a table" | Diwali Kali Puja 22 ×5 | Diwali ×3, Avyan |
| "a beach" | paramita desktop ×2, Gopalpur ×2, Leh Ladakh | Gopalpur ×5 |
| "a baby" | Avyan ×2, Wedding, paramita desktop ×2 | Avyan ×2, paramita desktop ×3 |

*(Gopalpur is a beach town in Odisha; Kashmir and Leh Ladakh are the two
mountain trips. The retrieval is finding real content, not album names — the
model never sees an album name.)*

### Verdict

**CLIP ViT-L/14 is the default.** It is 2.2× faster to encode at half the
VRAM, it wins the metric that matches the milestone's main job, and it is the
**only** candidate for which both a pre-built ONNX export and a trained
aesthetic head already exist in public — which is what makes the CPU fallback
and the aesthetic feature possible without this milestone exporting and
training its own artifacts.

SigLIP SO400M stays in the registry as a torch-only alternative
(`--model siglip-so400m`), because its 1-NN advantage is real and a
"find more like this one" feature would prefer it.

## 5. The library, re-measured

Read read-only from `data/rekindle.sqlite`, which is **already at schema v3**
— M2 bumped it concurrently, and `PhotoStore` on this branch (v2) cannot open
it at all. That is why the semantic pipeline reads the index through its own
version-agnostic reader.

| | measured |
|---|---|
| rows | 19,480 |
| live (not archived, not trashed) | 19,318 |
| archived | 162 |
| trashed | **0** |
| images (all) / live images | 18,363 / **18,201** |
| videos | 1,117 |
| with a capture date | 19,480 (100%) |
| with face tags | 10,887 (55.9%) |
| **live, with no face tag at all** | **8,433** |
| with GPS | 2,330 |
| distinct albums | 66 — 23 `Photos from YYYY` buckets + **43 named** |
| distinct people | 40 |
| span | 2000 – 2026 |

**Every library figure in the brief re-measured as correct**, including the
ones that are easy to get wrong: Kashmir 510, Leh Ladakh 277, ladakh 237,
Avyan 1134, Wedding_arnab_pics 810, Singapore Malyasia 338, Gopalpur 249,
43 named albums, 40 people, 8,433 untagged. That is a change from M1, where
four carried-over numbers were all wrong.

Two numbers the brief did **not** contain, and which sharpen the milestone's
justification:

* **19,480 of 19,480 photos have at least one album** — Takeout's year folders
  become albums, so "has an album" is not a signal at all. **15,388 live
  photos have no album that is not a year bucket.**
* **7,037 live photos have no face tag *and* no real album** (6,483 of them
  images). That, not 8,433, is the population that is invisible to every
  metadata recipe at once.

File types among live images: 18,122 `.jpg`, 52 `.jpeg`, 16 `.gif`, 10 `.png`,
1 `.mpo`.

## 6. Throughput: the GPU was never the bottleneck

The benchmark's end-to-end figure — 6.11 img/s with the GPU encoding at 83 —
says the pipeline was 93% JPEG decode on one core. 81.9 of 88.4 seconds for
540 photos went on decoding and resizing. A faster GPU would have changed
nothing.

Two changes in `embed.py`, both measured:

* **`Image.draft()`** — asks libjpeg to decode at 1/2, 1/4 or 1/8 scale
  straight from the DCT coefficients. A 4000×3000 photo headed for a 224×224
  crop is decoded at ~500×375; the pixels the resize would have discarded are
  never materialised.
* **A six-thread decode pool** — Pillow releases the GIL inside the C decoder,
  so threads give real parallelism with no image pickling across a process
  boundary.

| | img/s, end to end |
|---|---|
| naive loop (the benchmark) | 6.11 |
| `draft()` + 6 decode threads, batch 64 | **54.1** |

**8.9× faster, with no change to the model, the GPU or the vectors.**

`tests/test_semantic_embed.py::test_decode_draft_does_not_change_the_embedding_much`
pins the speed hack to not being a correctness change.

## 7. The verification run

Executed with **`HF_HUB_OFFLINE=1`** set for the whole run, against
`data/rekindle.sqlite` (a byte copy of the reference index; the original was
never opened for writing) and the real 61 GB library on `D:`.

### Embedding

```
device: cuda (requested: auto)
torch:  2.14.0+cu126 (CUDA build: 12.6)
gpu:    NVIDIA RTX A4000, 16375 MiB VRAM, driver 551.61, compute 8.6
runtime: torch

considered=18201  embedded=18201  already=0  missing=0  unreadable=0  accounted=True
336.1 s  ->  54.1 img/s
matrix on disk 55.9 MB · manifest 2.7 MB
```

**Every one of the 18,201 live images embedded. Nothing was missing, nothing
failed to decode, and the accounting identity held.** The whole library takes
**5 minutes 36 seconds**. A second run embeds nothing and reports 18,201
already present.

55.9 MB is 18,201 × 768 × 4 bytes exactly. Storing the same vectors as BLOBs
on `photos` would have added 56 MB to a table that `iter_photos()` streams in
full on every enrich run.

### Search

| | |
|---|---|
| matrix load from disk | **0.05 s** |
| mean query latency (10 queries, k=10, includes text encoding) | **31 ms** |

Real queries against the whole library:

| query | best score | what came back |
|---|---|---|
| snow covered mountains | 0.2525 | `Jammu and Kashmir 21st May 2015 213.JPG` (album: Kashmir) |
| a sunset | 0.2548 | `Photo0585.jpg` (paramita desktop) |
| a temple | 0.2919 | `PH7DD~52.JPG` (paramita desktop) |
| fireworks | 0.2836 | `DSC_0805.JPG` |
| a birthday cake | 0.2680 | `IMG_20181008_203801.jpg` |
| a beach and the sea | 0.2702 | `Image129.jpg` |
| a screenshot of a phone screen | 0.2657 | `IMG-20151102-WA0002.jpg` |

Scores of 0.25–0.29 are normal magnitudes for CLIP text↔image cosine; they are
not weak matches. 31 ms per query over 18,201 vectors is why there is no ANN
index: an approximate one would turn 31 ms into perhaps 5 ms and cost a C++
dependency and a recall knob.

Run through the shipped CLI, offline, as a user would:

```
$ HF_HUB_OFFLINE=1 rekindle semantic find "a dog" -k 5
 0.2440  2024-12-25  .../Photos from 2024/IMG_20241225_205833.jpg
 0.2416  2011-10-10  .../Photos from 2011/Photo0213.jpg
 ...
```

The second hit is a street dog photographed in 2011. It has no face tag and no
album, so no recipe in the design could ever have reached it. It is now one
word away.

### Scene clustering — the point of the milestone

Run over the **6,483 embedded photos that have no face tag and no album that
is not a year bucket** — the population invisible to every metadata recipe at
once.

```
k = 57 (sqrt(n/2))  ->  57 non-empty clusters, 29 iterations, converged, 0.4 s
all 6,483 photos assigned; sizes 1 / 97 (median) / 462
re-running with the same seed gives identical membership: True
```

**What the clusters actually turned out to be.** Sixteen random members of
each were rendered as a contact sheet and looked at — a cohesion score is not
evidence that a cluster is a scene:

| cluster | n | auto-label | what it actually is |
|---|---|---|---|
| #31 | 462 | *a forest or green countryside* | **Correct.** Himalayan foothill landscapes — hill stations, forested valleys, a river, a jungle-camp sign. Coherent and usable as one memory. |
| #50 | 243 | *a plate of food* | **Correct, 16/16.** Home cooking and restaurant plates. |
| #56 | 151 | *a dog or a cat* | **Correct, 16/16.** Street dogs around one neighbourhood. 151 photos no metadata recipe could ever have found. |
| #25 | 149 | *a sunset or sunrise* | **Coherent but not a memory.** Every member is a downloaded stock desktop wallpaper. The cluster is a ready-made junk filter, which may be worth more than a memory. |
| #16 | 270 | *a hospital or a clinic* | **Coherent, badly labelled.** It is institutional buildings with gardens — a campus, balcony views, an old colonial building. See below. |

The clustering works. **The labelling is the weak part, and one of its failure
modes is dangerous:**

* **Label scores are 0.127–0.282**, and only 23 of the 32 vocabulary entries
  were used. `a religious ceremony or temple` was chosen for **12 of 57
  clusters** — it absorbs anything with ornate colour.
* **Cluster #16 was labelled "a hospital or a clinic" and contains no
  hospital.** In this project a hospital is a *sensitive context* (main spec
  §7.2). A cosmetic label that lands on a sensitive-context word stops being
  cosmetic. Recorded in `known-limitations.md`: these labels must not be wired
  to sensitive-context detection as they stand.
* **48 of 57 clusters have a neighbouring centroid closer to them than their
  own members are** (median cohesion 0.834, median nearest-centroid 0.894).
  CLIP embeddings live in a narrow cone, so absolute cosines compress and
  k-means is slicing one continuous region rather than finding islands. The
  partition is useful and reproducible; it is not evidence that 57 natural
  scene types exist. This is the strongest argument for the deferred
  density-based clustering.

### Aesthetic ranking

The LAION head loads in 0.40 s and scores **all 18,201 vectors in 0.16 s
(110,529/s)** — no GPU, no torch.

```
min 2.120 · p10 4.285 · median 4.906 · p90 5.404 · max 6.734 · mean 4.867
```

The 508→40 problem the capability exists for, with and without the spread rule:

| album | candidates | plain top-40: near-duplicate pairs (cos > 0.92) | with spread |
|---|---|---|---|
| Kashmir | 508 | **16** | **0** |
| Leh Ladakh | 277 | **47** | **0** |
| Wedding_arnab_pics | 810 | **17** | **0** |
| Gopalpur | 184 | **18** | **0** |

Plain top-k on an aesthetic score really is a near-duplicate machine: Leh
Ladakh's top 40 contained 47 pairs above 0.92 cosine. The spread rule removes
all of them for a score cost of 0.09–0.22 rating points at the tail
(Kashmir's 40th pick falls 5.139 → 5.083).

### The ONNX CPU fallback, measured against the GPU path

The fallback exists so a contributor without 2.5 GB of torch can use the
features. If its vectors did not land in the same place as the torch path's,
a store filled by one and queried by the other would be quietly wrong — so
the same 32 real photos were run through both:

| | ONNX, fp32, CPU | torch, fp16, CUDA |
|---|---|---|
| throughput | **1.04 img/s** | **51.1 img/s** |
| session / model load | 2.29 s | ~7 s |
| whole library would take | ~4.9 hours | 5 min 36 s |

| agreement between the two runtimes | |
|---|---|
| image-vector cosine | min 0.9746 · **mean 0.9966** |
| text-vector cosine | **1.00000** |
| queries returning a byte-identical top-5 | 3 of 4 (the fourth differs only in the order of ranks 3–5) |

The residual disagreement is fp16 against fp32, not a defect. **The fallback
is correct and 49× slower**, which is the honest trade and is why the torch
path is preferred automatically when it is available.

`Xenova/clip-vit-large-patch14` also ships `vision_model_uint8.onnx` and
`_fp16` variants that would be several times faster. They are **not** pinned,
because their agreement with the fp32 path has not been measured here and an
unmeasured quantisation is exactly the silent quality loss the rest of this
design works to prevent.

**Two defects found by running this comparison, neither visible to a green
suite:**

1. **The ONNX pin named the wrong graph.** `Xenova/clip-vit-large-patch14`
   ships `onnx/model.onnx`, `onnx/vision_model.onnx` and
   `onnx/text_model.onnx`. The first is the *combined* CLIP model and requires
   `input_ids`, `pixel_values` **and** `attention_mask` in one call; giving it
   images alone fails with *"Required inputs (['pixel_values',
   'attention_mask']) are missing"*. It looked entirely correct in the
   repository file listing and passed every test, because no test loaded a
   real session. Fixed to the two single-tower graphs, each of which takes one
   input and returns the 768-d projected embedding, and pinned by
   `test_the_onnx_pin_names_the_single_tower_graphs_not_the_combined_one`.

2. **`TorchEncoder` had no `local_files_only`.** The first encoder load would
   download 1.7 GB — straight through this milestone's central promise that
   only `rekindle semantic setup` touches the network. It surfaced the instant
   torch was installed on this machine: a CLI test that expected an immediate
   "install the extra" message instead sat pulling CLIP into a pytest
   temporary directory. Offline is now the default, with an explicit
   `allow_download` flag, and
   `test_the_torch_encoder_never_downloads` asserts that an empty cache
   produces a message rather than a download.

## 8. The face gate, measured honestly

**Ground truth was produced by a human looking at the photos**, not by Google's
face tags — those are recall-poor, and grading against them would measure
agreement with the problem. 64 live, image, *face-tag-free* photos were
sampled at seed 20260911, rendered as four 16-up contact sheets, and labelled
by eye. 19 of the 64 contain a visible human face.

Detector: **YOLOv11n-face**, ONNX, CPU only — **11.3 img/s** on this CPU
(all 18,201 images would take about 27 minutes).

### Results at the shipped thresholds (detect 0.45, gate 0.15)

| | |
|---|---|
| **recall (faces caught)** | **1.000 — 19 of 19, zero misses** |
| precision | 0.633 |
| true positives / false positives | 19 / 11 |
| **false negatives** | **0** |
| true negatives | 34 |
| proposed publishable | 34 of 64 |

**No face was missed.** That is the number that matters: a false negative here
is a stranger published.

### What the 11 false alarms actually are

Nine of the eleven are the detector doing its job on something the labelling
convention excluded:

| # | top score | what it is |
|---|---|---|
| 06 | 0.840 | The Diskit Buddha statue, Ladakh — a large sculpted face |
| 17 | 0.617 | A newspaper page whose printed photo shows a crowd |
| 23, 40, 44, 47 | 0.57–0.85 | Durga and Kali idols, and printed religious artwork |
| 62 | 0.794 | A quotation card with an illustrated portrait |
| 55 | 0.473 | A Kashmir lake photo with a distant boat |
| 35, 60 | 0.52–0.53 | **Genuine errors** — a plate of food and a cake |
| 24 | 0.328 | Night lights; the only `uncertain`, correctly not eligible |

Idols and statues are the dominant error mode in this library, which is a
Bengali family library full of Durga Puja. Under the opposite convention —
"anything face-shaped blocks publication" — precision would be 0.95 or better.
Whether a statue should block is a policy question for the user, so the gate
reports counts and boxes and lets a human decide; it is not a defect of the
detector.

### Threshold sweep — and a finding

| detect | gate | precision | recall | proposed publishable |
|---|---|---|---|---|
| 0.45 | 0.45 | 0.655 | 1.000 | 35 |
| 0.45 | 0.30 | 0.633 | 1.000 | 34 |
| **0.45** | **0.15** | **0.633** | **1.000** | **34** |
| 0.45 | 0.10 | 0.633 | 1.000 | 34 |
| 0.45 | 0.05 | 0.633 | 1.000 | 34 |
| 0.30 | 0.10 | 0.633 | 1.000 | 34 |
| 0.25 | 0.05 | 0.633 | 1.000 | 34 |

**The two-threshold design buys exactly one held-back photo out of 64 on this
detector.** YOLOv11n-face is confident or silent: it almost never produces a
detection between 0.05 and 0.45. The asymmetry costs almost nothing and buys
almost nothing *here* — it is kept because it costs almost nothing, and
because a different detector, a different library or a blurrier photo is
exactly when it would start earning its place.

### Over 400 untagged photos, through the shipped entry point

```
eligible=219  has_face=148  uncertain=33  errors=0  accounted=True   11.4 img/s
```

**54.8% of face-tag-free photos are proposed as face-free** — against a
current publishing rule that default-denies 100% of them.

### What this does NOT establish

**19 positives is a small sample.** Nineteen consecutive successes bound the
miss rate below roughly **15% at 95% confidence** — no tighter. A gate that
has never been seen to miss is not a gate that cannot miss, and the cases it
would miss (a face at 20 px, deep shade, a profile behind a shoulder) are
under-represented in 64 random photos.

**So the gate must not auto-publish, and it does not.** There is no code path
in rekindle that turns `ELIGIBLE` into a published file. `ELIGIBLE` means
"propose this to a human", `review_queue()` includes the eligible ones for
exactly that reason, and the CLI prints *"This is a PROPOSAL, not a decision"*
above the table. Anyone wanting a number to rely on should re-run this
measurement on several hundred hand-checked photos from their own library
first.

## 9. Mutation testing

**49 plausible defects were injected one at a time** — a guard deleted, a
comparison flipped, a normalisation dropped, an allow-list removed — and the
full suite run against each.

**First pass: 41 killed, 8 survived.** Eight lines could be deleted or
inverted with the suite green. Each survivor is now covered by a test in
`tests/test_semantic_mutants.py`, and a re-run kills all eight. The survivors
were real gaps, not noise:

| surviving mutation | why the old tests could not see it |
|---|---|
| `without_people` checking only `people` | Nothing in the real index has one person field set and the other empty, so both clauses agree on today's data. The failure would land on the **face gate**. |
| `load_matrix`'s row-contiguity check | Needs a manifest with a gap — a shape no happy-path test produces. Without it, every hash after the gap names the wrong vector. |
| `_normalise` dropped | Only the torch and ONNX encoders call it, and neither runs in CI. Nothing exercised it at all. |
| the `require()` guard before the ONNX path | Deleting it still exits 3, just with a message about an internal import instead of the extra. |
| k-means seeding ignoring the seed | Solid-colour fixtures have one correct partition that every seeding finds. |
| `min_cosine` not masking membership | `outliers` is computed separately, so the list stayed right while the photo stayed in a cluster. |
| preprocessing ignoring its config | CLIP's configured size (224) is also the hardcoded fallback. |
| `top_k` returning an unsorted partition | See below. |

**One mutant could not be killed by data, and that is recorded rather than
papered over.** `numpy.argpartition` documents the order within its partition
as undefined, but numpy 2.5.3's introselect returns an already-descending
prefix for every n and k tried (n = 6…5000, k = 2…20, random and adversarial).
On this numpy the argsort in `top_k` is a no-op and the mutation is invisible.
A test built on real scores would have pinned numpy's current internals rather
than `top_k`'s contract, so the test instead supplies the unsorted partition
numpy is free to return, and asserts that the sort fixes it.

**Two real defects were found by writing the tests**, neither by reading code:

1. `load_encoder` reached the ONNX branch on a machine with no extras and
   reported whatever failed first (a missing directory) instead of naming the
   extra to install.
2. **transformers 5.x changed the return type.** `get_image_features` returns
   `BaseModelOutputWithPooling`, not a tensor; the 4.x code path dies with
   `'BaseModelOutputWithPooling' object has no attribute 'float'`. Found by
   running the real model during the benchmark — exactly the "found by
   running, not by reading" pattern the M1 log describes.

Baseline on the branch point was 263 passed / 9 skipped. Now: **473 passed /
9 skipped**, `ruff check` and `ruff format --check` clean, and every test runs
with no network, no GPU, no model weights and no photos.

## 10. Things in the brief that turned out to be wrong or incomplete

1. **"`cdn-lfs.huggingface.co` does NOT resolve … verify early that your
   chosen model actually downloads."** The fact is correct and the warning was
   worth heeding, but the host is **retired**: Hugging Face serves large files
   through Xet (`us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`) now, and all
   3,937 MB downloaded without trouble. The real Windows download trap turned
   out to be somewhere else entirely — PyPI's CPU-only torch wheel (§3).

2. **"CUDA Toolkit 12.4 installed" implies a cu124 build.** For
   cp312/win_amd64, PyTorch's cu124 channel stops at torch **2.6.0**. cu126 is
   required to get a current torch, and it works fine on the 12.4 driver.

3. **"8,433 with NO face tags at all ← the population that motivates this
   milestone."** The count is right, but it is not the right population.
   **19,480 of 19,480 photos have an album**, because Takeout's year folders
   become albums — so "and no real album" has to be spelled out. The number
   that motivates the milestone is **7,037** (6,483 of them images).

4. **Everything else in the brief's library table re-measured as correct** —
   19,480 / 19,318 / 10,887 / 2,330 / 1,117 / 162, 43 named albums, 40 people,
   Kashmir 510, Leh Ladakh 277, ladakh 237, Avyan 1134, Wedding_arnab_pics 810,
   Singapore Malyasia 338, Gopalpur 249, 2000–2026. No repeat of M1's
   four-wrong-numbers problem.

5. **The brief said to assume `SCHEMA_VERSION = 3` is taken.** It is already
   taken: the shared index is at v3 *now*, so `PhotoStore` on this branch
   cannot even open it. That turned "keep your footprint small" from advice
   into a hard requirement, and is why the semantic pipeline reads the index
   through its own version-agnostic reader.

---

# Part II — the resumed session

Written after the session that produced everything above ended unexpectedly.
Its last commit was preserved unreviewed by the orchestrator as `612f498`,
with a note that it needed reviewing before being trusted. Everything below
was measured on the same machine against the same library.

## 11. The inherited commit: reverted, and why

`612f498` carried one line of source:

```diff
-  "local_files_only": not allow_download,
+  "local_files_only": False,
```

**It is a debugging leftover, not a decision, and it was reverted.** Three
pieces of evidence, in increasing order of how much they settle it.

1. **It contradicts the docstring three lines above it**, which states the
   offline contract in full and explains that the default was chosen *because*
   the alternative was found by running.

2. **It broke the suite.** The baseline on the inherited tree was **471 passed,
   3 failed** — not the 473/9 the audit above reports:

   ```
   FAILED test_semantic_cli.py::test_embed_without_the_extra_names_the_extra
   FAILED test_semantic_cli.py::test_embed_with_device_cuda_refuses_rather_than_falling_back
   FAILED test_semantic_mutants.py::test_the_torch_encoder_never_downloads
   ```

   The third failed with `DID NOT RAISE SemanticUnavailable` while pytest's
   captured stderr showed a weight-loading progress bar — the exact failure
   that test was written to prevent, reproduced by the change itself. **The
   change is its own mutation test, and the suite killed it.**

3. **There was no load problem to debug around.** With the line restored and
   `HF_HUB_OFFLINE=1` set: the cache at `data/models` loads in **7.02 s**,
   `device auto` resolves to `cuda`, parameters land on `cuda:0` in
   `torch.float16`, and an empty cache directory refuses with *"run `rekindle
   semantic setup`"* without attempting a download.

### What it would actually have cost

Measured after the fact, because "it contradicts the offline promise" understates
it. Same load, same populated cache, with the hub endpoint refusing connections:

| | time to load |
|---|---|
| `local_files_only = not allow_download` | **15.0 s** |
| `local_files_only = False` | **385.8 s** |

**25×.** `huggingface_hub` retries a `HEAD` five times per file before falling
back to the cache. On a genuinely offline machine the inherited change costs
six and a half minutes on **every** encoder load, not just the first.

### The offline guarantee is now tested, not assumed

The first replacement test written for it **did not fail when the defect was
put back**, because the fixture sets `HF_HUB_OFFLINE=1` — the environment was
doing the work and the code could have been anything. That is the decision
log's "test that cannot fail", caught only by re-injecting the defect.

`test_loading_from_the_cache_makes_no_network_request_at_all` instead **unsets**
`HF_HUB_OFFLINE`, points `HF_ENDPOINT` at a local socket that counts what
arrives, loads from the populated cache, and requires the count to be **zero**.
It is about the code rather than the environment, and it fails in 12 s.

## 12. Does the batch size exploit 16 GB of VRAM?

**No, and it should not try.** The obvious criticism of a hardcoded `32` is that
it was tuned for a laptop card and wastes an A4000. Measured on 512 real photos,
pre-decoded so the GPU is the only variable:

| batch | img/s | peak VRAM |
|---|---|---|
| 8 | 75.1 | 899 MiB |
| 16 | 71.2 | 966 MiB |
| 32 | **79.4** | 1,099 MiB |
| 64 | 79.3 | 1,365 MiB |
| 128 | 76.5 | 1,900 MiB |
| 256 | 75.4 | 2,963 MiB |

**A 32× range of batch sizes spans 71–79 img/s — no trend, all noise.** Even
batch 256 uses 18% of the card. Sizing the batch from available VRAM would have
been cargo cult: it would have produced a large number, changed nothing, and
looked like engineering.

### Why the batch is flat

`TorchEncoder.encode_images`, stage by stage, 256 images at batch 32:

| stage | share | throughput alone |
|---|---|---|
| **Hugging Face image processor (CPU)** | **65.9%** | **117 img/s** |
| host → device copy | 0.7% | 11,686 img/s |
| **ViT-L/14 forward (GPU)** | **32.7%** | **236 img/s** |
| device → host + `tolist` | 0.2% | 36,348 img/s |
| `_normalise` (python) | 0.5% | 15,022 img/s |

The encoder is **CPU-preprocessing-bound**. The GPU is idle two thirds of the
time even during the part of the pipeline that is supposed to be GPU work, and
a bigger batch grows the serial CPU half in exact proportion to the GPU half it
is trying to fill.

**The cause is a silent fallback.** `torchvision` is not installed, so
transformers falls back from `CLIPImageProcessor` to `CLIPImageProcessorPil`
and says so only in a log line nobody reads. This is the same shape as PyPI's
CPU-only torch wheel in §3 — a dependency that is quietly absent and whose only
symptom is being slower. `use_fast=True` does not help; it resolves to the same
PIL class, and produces byte-identical pixel values (max abs difference
0.000000), which is how that was confirmed rather than assumed.

### The lever that does exist

The old loop called `pool.map` and blocked on the result, so the decode pool was
idle for every second of the encode and the encode was idle for every second of
the decode — while the module docstring claimed the pool ran "one batch ahead of
the GPU". **It did not, and nothing tested the claim.**

`prepare_images` / `encode_prepared` now split both encoders at the CPU/device
line, `embed_photos` runs the CPU half in the pool it already owns, and batches
are submitted `PREFETCH_BATCHES` ahead and consumed in order.

| prefetch | median img/s | observed range |
|---|---|---|
| 1 | 25.4 | 18 – 29 |
| 2 | 52.3 | 30 – 53 |
| 4 | 78.7 | 52 – 83 |
| 6 | 60.1 | 58 – 93 |
| **12** | **93.4** | 67 – 103 |
| 24 | 70.8 | 68 – 98 |

*(Median of five interleaved rounds over 512 real photos, the order rotated each
round.)*

**Read the ranges, not only the medians.** A prefetch of 1 — the old shape — is
3–4× slower than anything from 4 up, and that gap is far outside the noise.
Between 4 and 24 **this machine cannot tell them apart**: another agent was
building in the main worktree throughout, and the same setting measured 17.9 and
106.8 img/s twenty minutes apart. An earlier descending sweep that looked
monotonic was a warming page cache, not a result; it is recorded here because
believing it would have been the easy mistake. 12 is chosen as 2 × workers and
the comment in the source says the measurement could not pick a winner.

### The whole library, again

```
considered=18201  embedded=18201  already=0  missing=0  unreadable=0  accounted=True
188.4 s  ->  96.6 img/s   (3m 08s)
peak VRAM allocated by torch: 1,099 MiB of 16,375  (6.7% of the card)
```

| | before | after |
|---|---|---|
| full library | 336.1 s · 54.1 img/s | **188.4 s · 96.6 img/s** |
| wall clock | 5 min 36 s | **3 min 08 s** |

**1.79×, with the vectors bit-identical** — verified over 1,024 real photos at a
fixed batch, and again on the toy encoder at every prefetch depth from 1 to 40.

One honest wrinkle. Only **25 of 18,201** vectors are bit-identical to the store
the previous session left, worst cosine **0.99902**. That is not the pipeline
change. It is that the old store was built at batch 64 and the rebuild ran at
the default 32, and **fp16 reduction order depends on batch shape**:

| batch, against batch 32 | bit-identical | min cosine |
|---|---|---|
| 16 | yes | 0.99999996 |
| 64 | **no** | 0.99997994 |
| 128 | **no** | 0.99997996 |

Irrelevant at the cosines search and clustering work with (0.25–0.29), but it
means **a store should be filled with one batch size throughout**, and it is a
further reason not to treat the batch as a free knob. The existing store was
left untouched rather than rebuilt.

## 13. Where each of the three models actually runs

The question was whether the face detector and the aesthetic predictor quietly
sit on the CPU while CLIP uses the card. **One of them does, it is the slowest
stage in the milestone, and it cannot do otherwise.**

| | device | evidence | throughput | whole library |
|---|---|---|---|---|
| CLIP ViT-L/14 | **GPU** | params on `cuda:0`, `torch.float16`, 1,099 MiB allocated during a real run | 96.6 img/s end to end | **3m 08s** |
| aesthetic head | **CPU** | never imports torch; five numpy matmuls | ~110,000 vec/s | **0.16 s** |
| YOLOv11n-face | **CPU only** | see below | 11.7 → 26.4 img/s | 26m → **11m** |

**The aesthetic head is on the CPU and that is correct.** It is a linear chain
of five matrix multiplies over vectors that are already in memory; the whole
library scores in 0.16 s. Moving it to the GPU would optimise 0.05% of the
milestone and add a torch dependency to a feature that currently has none.

**The face detector is on the CPU because it has no choice here.** Two reasons,
both measured rather than inferred:

```
onnxruntime 1.30.0
ort.get_device()             -> 'CPU'
ort.get_available_providers()-> ['AzureExecutionProvider', 'CPUExecutionProvider']
model input shape            -> [1, 3, 640, 640]
```

* **The installed onnxruntime is the CPU build.** There is no
  `CUDAExecutionProvider` to select, so `FaceDetector(prefer_gpu=True)` resolves
  to `('CPUExecutionProvider',)` and **silently does nothing**. Fixing it means
  `onnxruntime-gpu`, which *replaces* `onnxruntime` in the same import
  namespace — putting the verified ONNX CPU fallback (§7) at risk for the sake
  of a stage that is not CLIP. **Not done, deliberately.**
* **The pinned graph is fixed batch 1.** Even with a CUDA provider, batching
  would need the model re-exported.

### What was done instead

The scan was entirely serial and 60% of its cost was not inference at all:

| stage | share | alone |
|---|---|---|
| open + `draft` + convert | 30.9% | 38.9 img/s |
| letterbox (PIL resize + numpy) | 29.4% | 40.9 img/s |
| onnxruntime `session.run` (CPU) | 39.6% | 30.3 img/s |
| YOLO decode (python) | 0.1% | 8,520 img/s |

Pillow, numpy and onnxruntime all release the GIL, so threads give real
parallelism. Two runs over 240 real face-tag-free photos:

| workers | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| img/s, `detect()` directly | 9.6 | 18.8 | 25.3 | 23.7 | 16.2 |
| img/s, through `gate_photos` | 11.7 | 19.0 | 26.4 | **28.5** | 27.3 |

**2.44×. The library scan goes from 26.0 minutes to 10.6.**

The two runs put the peak in different places, so they do **not** agree that 4
beats 6 — only that anything from 4 up is roughly 2.4× serial and the curve is
flat there. **4 is the shipped default**, as the conservative end of that
plateau: it leaves a core for onnxruntime's own intra-op pool
(`intra_op_num_threads` defaults to 0, meaning all of them) and does not fall
off a 4-core machine. `--workers` is exposed for anyone whose machine disagrees.

### A publishing gate does not get "probably the same"

Over the same 240 real photos, **at every width from 1 to 8, the verdicts, the
boxes and their order were identical to the serial run**, with
eligible/has_face/uncertain/errors at 129/76/35/0 throughout.

And through the shipped CLI over the same 400 untagged photos §8 used:

```
219 proposed as face-free, 148 contain a face, 33 uncertain, 0 unreadable  (20.9 img/s)
This is a PROPOSAL, not a decision. Nothing is published.
```

**Exactly the counts §8 recorded at 11.4 img/s.** The threading changed the
speed and nothing else.

## 14. The gate's driver had no test at all

Mutation testing on the new pipeline turned up a survivor that had nothing to do
with it. Replacing `if tagged - allowed:` with `if False:` — **deleting the
allow-list, so a photo Google has tagged with a stranger is sent to the detector
to be judged on its pixels** — left the entire suite green.

`test_semantic_faces.py` tests `classify`, `_nms`, `_iou`, `Box` and
`precision_recall` thoroughly, and every one of those is worth testing. **None
of them is the driver.** The allow-list, the missing / unreadable /
detector-raised paths, the accounting identity and the threshold pass-through
were all untested, in the one component of this milestone whose failure mode is
publishing a stranger's face.

Two new files close it, both built on detectors that score a real feature of a
real image rather than mocks:

* `tests/test_semantic_gate_driver.py` — 38 tests, no model, no weights, runs in
  CI.
* `tests/test_semantic_face_detector.py` — 11 tests against the **real ONNX
  detector on real photos**, skipped unless both the weights and an index are
  present.

### Why a real-photo test, when the helpers were already tested exhaustively

Because deleting the `_nms` **call** from `FaceDetector.detect` also left the
suite green — `_nms` was tested; its being used was not.

That gap is not academic. Over 300 random real photos at the shipped gate
threshold, **NMS removed at least one box on 220 of them, and on one photo it
cut 206 raw boxes down to 33.** Without it the gate would report a landscape as
containing two hundred faces.

A synthetic face cannot stand in, and this was tried rather than assumed: a
hand-drawn face scores **0.0033** on YOLOv11n-face and produces no raw boxes at
all to suppress. The detector is confident or silent — the same property §8's
threshold sweep found — so **only real photographs reach this code**.

## 15. Mutation testing, this session

**55 defects injected one at a time, 41 killed.** Eight rounds; each round's
survivors became the next round's tests, which is why the same mutation appears
twice with different outcomes.

| # | target | injected | killed | survived |
|---|---|---|---|---|
| 1 | the pipelined embed loop | 12 | 9 | 3 |
| 2 | `gate_photos`, after round 1 exposed the allow-list | 13 | 12 | 1 |
| 3 | `TorchEncoder`, against the new weights-gated file | 5 | 4 | 1 |
| 4 | round 3's survivor, after the request-counting test | 1 | **1** | 0 |
| 5 | the threaded gate | 9 | 8 | 1 |
| 6 | round 5's survivor, after the fixture grew a second box | 2 | **1** | 1 |
| 7 | `FaceDetector` geometry | 7 | 3 | 4 |
| 8 | round 7's survivors, after the invariants were tightened | 6 | 3 | 3 |
| | **total** | **55** | **41** | |

Fourteen raw survivors across the rounds collapse to **four distinct ones**: the
rest were killed by a later round once the test that should have caught them
existed. Those four are below.

**Three tests that could not fail were caught by injecting the defect and
watching them pass** — which is the only way any of them would have been found:

1. **The first offline test.** The fixture set `HF_HUB_OFFLINE=1`, so the
   environment enforced the guarantee and the code could have been anything.
   Replaced with the request-counting server in §11.
2. **The first bounds test for detector geometry.** `detect` *clamps* its boxes
   into the image, so a completely wrong pad or scale still produces boxes that
   are "inside the photo". Replaced with translation and scale invariance: move
   the photo by (dx, dy) and the box must move by exactly (dx, dy).
3. **A 2:1 canvas for the letterbox test.** Not wide enough to lose the face to
   a centre crop when the scale is picked with `max` instead of `min`. 4:1 is.

And one **fixture certifying its own fiction**, in miniature: `faces_found +=
len(boxes)` survived because `ScriptedDetector` only ever returned one box, so
"boxes" and "boxes above `detect_threshold`" could not differ. It now also
produces a fainter second detection — which real photos have, as a face in the
background or a reflection — and the mutation dies.

### The survivors, recorded rather than papered over

| survivor | why it is not a defect |
|---|---|
| consuming the newest future rather than the oldest in `embed_photos` | Rows are assigned in `add_many` order and the manifest maps hash → row, so a different consumption order yields a different but entirely self-consistent matrix. A **throughput** defect (it blocks on the least-complete work), not a correctness one. Killing it would mean pinning row order, which is not a contract. |
| `pad = (0, 0)` instead of centring the letterbox | The paste and the undo use the same pad, so top-left padding is a self-consistent alternative convention. An equivalent mutant. |
| zero-area boxes not filtered | No real photo in the sample produces one. The guard is defence against an export that might. |
| `NEAREST` instead of `BILINEAR` when resizing | Breaks no invariant. It would show up as **recall**, which is measured on a hand-checked sample in §8, not in a unit test. |

## 16. Where the suite stands

| | |
|---|---|
| inherited baseline | 471 passed, **3 failed**, 10 skipped |
| now | **550 passed, 10 skipped** |
| `ruff check`, `ruff format --check` | clean |
| tests needing a network | 0 |
| tests needing a GPU | 0 |
| tests needing model weights | 0 in CI (23 skip without them) |

The 23 that skip are the two weights-gated files, `test_semantic_torch_encoder.py`
and `test_semantic_face_detector.py`. They exist because mutation testing showed
that **no CI test loads a real model**, so the bodies of both the torch encoder
and the ONNX detector could be deleted with the suite green. On a machine that
has run `rekindle semantic setup` they run and kill those mutations; in CI they
report why they skipped.

## 17. What is still open

* **`onnxruntime-gpu` for the face detector.** It would need a provider swap in
  a shared import namespace, and the win over four threads is unmeasured. The
  honest position is that 11 minutes for the whole library is now acceptable and
  the risk to the ONNX fallback was not.
* **`torchvision`, to get transformers' fast image processor.** It is the
  remaining two thirds of the encode. Not installed here, and installing it
  would change the pixel values the model sees, which means re-measuring
  agreement before trusting any vector it produced — a re-embed of the whole
  library, not a dependency line.
* **The gate's sample is still 19 positives.** Nothing in this session enlarged
  it. §8's caveat stands in full: nineteen consecutive successes bound the miss
  rate below roughly 15% at 95% confidence and no tighter, and the gate still
  must not auto-publish, and still does not.
