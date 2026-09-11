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
| `draft()` + 6 decode threads, batch 64 | **see §7** |

`tests/test_semantic_embed.py::test_decode_draft_does_not_change_the_embedding_much`
pins the speed hack to not being a correctness change.

---

*(Sections 7-9 — the verification run, the face gate and the mutation
results — follow.)*
