/* rekindle's memory workshop.
 *
 * Vanilla ES2020, no build step, no framework, no CDN. The page is small
 * enough that a virtual DOM would be more code than the thing it manages, and
 * every dependency here would be a file fetched from somewhere - which is the
 * one thing this page must never do.
 *
 * Every state-changing call posts to /api/edit and REPLACES the local state
 * with the server's answer. The browser never computes what the memory
 * contains; it asks. That is what makes "a dropped photo stays dropped"
 * testable without a browser.
 */

"use strict";

const TOKEN = new URLSearchParams(location.search).get("t") || "";
const $ = (id) => document.getElementById(id);

let state = null;      // the last /api/... session payload
let cutFilter = "";    // which rejection reason the cut grid is showing
let results = [];      // the last search / similar / same-day hits

/* ---------------------------------------------------------------- plumbing */

async function api(path, options = {}) {
  const headers = Object.assign({ "X-Rekindle-Token": TOKEN }, options.headers || {});
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, Object.assign({}, options, { headers }));
  const payload = await response.json().catch(() => ({ error: "the server sent no JSON" }));
  if (!response.ok) throw new Error(payload.hint ? `${payload.error} ${payload.hint}` : payload.error);
  return payload;
}

const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body) });
const thumb = (hash, width) => `/api/thumb/${hash}?w=${width}&t=${encodeURIComponent(TOKEN)}`;

function text(node, value) { node.textContent = value === undefined || value === null ? "" : String(value); }

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function say(node, message, bad) {
  text(node, message);
  node.style.color = bad ? "var(--bad)" : "";
}

/* ------------------------------------------------------------------ status */

async function loadStatus() {
  let status;
  try {
    status = await api("/api/status");
  } catch (error) {
    say($("library"), error.message, true);
    return;
  }
  if (status.state === "loading") {
    text($("library"), "reading your library…");
    setTimeout(loadStatus, 400);
    return;
  }
  if (status.state === "failed") {
    say($("library"), status.error, true);
    return;
  }
  const withheld = Object.entries(status.withheld_by_reason || {})
    .map(([reason, n]) => `${n} ${reason.replace(/_/g, " ")}`).join(", ");
  text($("library"),
    `${status.photos.toLocaleString()} photos available` +
    (status.withheld ? ` · ${status.withheld} withheld by the guardrails (${withheld})` : "") +
    (status.unfingerprinted ? ` · ${status.unfingerprinted} not fingerprinted` : ""));

  const music = $("music");
  music.replaceChildren(new Option("silence", ""));
  (status.music || []).forEach((name) => music.add(new Option(name, name)));

  if (!status.semantic_installed) {
    $("mode-semantic").disabled = true;
    say($("prompt-note"),
      "Prompt search needs the semantic extra: uv sync --extra semantic, then rekindle semantic embed. " +
      "Everything else on this page works without it.");
  } else {
    say($("prompt-note"), status.semantic_note || "");
  }
  loadOffers();
}

async function loadOffers() {
  let payload;
  try {
    payload = await api("/api/offers");
  } catch (error) {
    say($("prompt-note"), error.message, true);
    return;
  }
  const box = $("offers");
  const draw = () => {
    const needle = $("offers-search").value.trim().toLowerCase();
    box.replaceChildren();
    payload.offers
      .filter((o) => !needle || `${o.title} ${o.key} ${o.recipe}`.toLowerCase().includes(needle))
      .slice(0, 400)
      .forEach((offer) => {
        const button = element("button", "offer" + (offer.dismissed ? " dismissed" : ""));
        button.type = "button";
        button.appendChild(element("b", null, offer.title));
        const meta = element("span", "meta",
          `${offer.recipe} · ${offer.subtitle || offer.key}` +
          (offer.dismissed ? " · dismissed" : offer.cooling ? " · shown recently" : ""));
        button.appendChild(meta);
        button.addEventListener("click", () => build({ recipe: offer.recipe, key: offer.key }));
        box.appendChild(button);
      });
  };
  $("offers-search").addEventListener("input", draw);
  draw();
}

/* ---------------------------------------------------------------- building */

function build(params) {
  $("workshop").hidden = true;
  $("progress").hidden = false;
  $("log").replaceChildren();
  text($("stage"), "starting…");
  $("prompt-go").disabled = true;

  const query = new URLSearchParams(params);
  query.set("t", TOKEN);
  const stream = new EventSource(`/api/build?${query.toString()}`);
  const note = (line) => $("log").appendChild(element("li", null, line));

  stream.addEventListener("stage", (event) => text($("stage"), JSON.parse(event.data).message));
  stream.addEventListener("tags", (event) => {
    const data = JSON.parse(event.data);
    note(`${data.tags.length} visual descriptions, from ${data.source}: ${data.tags.join(" · ")}`);
    if (data.weak) {
      note("! Nothing describes this prompt visually, so your words were searched directly. " +
           "That is the measured-bad path.");
    }
    if (data.months && data.months.length) note(`narrowed to months ${data.months.join(", ")}`);
  });
  stream.addEventListener("searched", (event) => {
    const data = JSON.parse(event.data);
    text($("stage"), `searched ${data.done} of ${data.of}: ${data.tag}`);
    note(`${data.tag} → ${data.hits} hits`);
  });
  stream.addEventListener("found", (event) => {
    const data = JSON.parse(event.data);
    note(`${data.seed_days.length} capture days agreed, expanded to ${data.pool} candidates`);
    data.seed_days.forEach((day) => note(`  ${day.day} — ${day.hits} photos, ${day.tags} descriptions`));
    if (data.albums.length) note(`your own albums added whole: ${data.albums.join(", ")}`);
    if (data.unmatched.length) {
      note(`! ${data.unmatched.join(", ")} narrowed nothing — searched as a picture only. ` +
           "rekindle has no gazetteer, so a place name never filtered anything.");
    }
    note(`tag agreement ${(data.agreement * 100).toFixed(0)}% — measured NOT to tell you whether ` +
         "the concept is in your library, so nothing is refused on it.");
  });
  stream.addEventListener("shot", () => text($("stage"), "composing the memory…"));
  stream.addEventListener("ready", (event) => {
    stream.close();
    $("prompt-go").disabled = false;
    adopt(JSON.parse(event.data));
  });
  stream.addEventListener("error", (event) => {
    stream.close();
    $("prompt-go").disabled = false;
    if (event.data) {
      const data = JSON.parse(event.data);
      say($("stage"), data.hint ? `${data.message} — ${data.hint}` : data.message, true);
    } else {
      say($("stage"), "the connection to rekindle dropped", true);
    }
  });
}

/* -------------------------------------------------------------- the memory */

function adopt(payload) {
  state = payload;
  $("progress").hidden = true;
  $("workshop").hidden = false;
  draw();
}

function candidate(hash) {
  return (state.candidates || []).find((c) => c.file_hash === hash);
}

function draw() {
  text($("chosen-count"), `${state.order.length} of ${state.max_shots}`);
  text($("memory-sub"), `${state.title} — ${state.subtitle}` + (state.public_safe ? " · public-safe" : ""));

  const banner = $("banner");
  banner.hidden = !state.note;
  text(banner, state.note || "");

  drawChosen();
  drawCut();
  if (results.length) drawResults();
  drawGuardrails();
  drawPace();
  drawReproduce();
}

function drawChosen() {
  const grid = $("chosen");
  grid.replaceChildren();
  state.order.forEach((hash, position) => {
    const info = candidate(hash);
    if (!info) return;
    const card = photoCard(info, { draggable: true, position });
    if (info.reason === "added_by_you") card.classList.add("added");
    const actions = element("div", "actions");
    actions.appendChild(action("Remove", () => edit({ op: "remove", file_hash: hash })));
    if (info.burst && info.burst.length > 1) {
      actions.appendChild(action(`Other frames (${info.burst.length - 1})`, () => showBurst(info)));
    }
    actions.appendChild(action("Same day", () => loadInto(`/api/day?hash=${hash}`, "the same day")));
    actions.appendChild(action("More like this", () => loadInto(`/api/similar?hash=${hash}`, "similar photos")));
    card.appendChild(actions);
    grid.appendChild(card);
  });
  wireDragging(grid);
}

function drawCut() {
  const cut = (state.candidates || []).filter((c) => c.state !== "chosen");
  text($("cut-count"), String(cut.length));

  const counts = new Map();
  cut.forEach((c) => counts.set(c.reason, (counts.get(c.reason) || 0) + 1));
  const chips = $("cut-filter");
  chips.replaceChildren();
  $("cut").parentElement.querySelectorAll("p.muted.truncated").forEach((n) => n.remove());
  const all = element("button", "chip" + (cutFilter ? "" : " on"), `all (${cut.length})`);
  all.type = "button";
  all.addEventListener("click", () => { cutFilter = ""; drawCut(); });
  chips.appendChild(all);
  [...counts.entries()].sort((a, b) => b[1] - a[1]).forEach(([reason, n]) => {
    const chip = element("button", "chip" + (cutFilter === reason ? " on" : ""),
      `${reason.replace(/_/g, " ")} (${n})`);
    chip.type = "button";
    chip.addEventListener("click", () => { cutFilter = cutFilter === reason ? "" : reason; drawCut(); });
    chips.appendChild(chip);
  });

  const grid = $("cut");
  grid.replaceChildren();
  const shown = cut.filter((c) => !cutFilter || c.reason === cutFilter);
  const LIMIT = 400;
  if (shown.length > LIMIT) {
    const note = element("p", "muted truncated",
      `Showing the first ${LIMIT} of ${shown.length}. Filter by a reason above, ` +
      "or search for what you are looking for.");
    grid.parentElement.insertBefore(note, grid);
  }
  shown.slice(0, LIMIT).forEach((info) => {
       const card = photoCard(info, {});
       const actions = element("div", "actions");
       actions.appendChild(action("Put it in", () => edit({ op: "add", file_hash: info.file_hash })));
       if (info.instead_of) {
         actions.appendChild(action("Use this frame instead",
           () => edit({ op: "swap", out: info.instead_of, in: info.file_hash })));
       }
       card.appendChild(actions);
       grid.appendChild(card);
     });
}

function photoCard(info, { draggable = false, position = null } = {}) {
  const card = element("div", "card");
  card.dataset.hash = info.file_hash;
  if (draggable) card.draggable = true;

  const image = document.createElement("img");
  image.loading = "lazy";
  image.decoding = "async";
  image.src = thumb(info.file_hash, 320);
  image.alt = info.caption || info.name;
  image.addEventListener("click", () => openSheet(info));
  card.appendChild(image);

  if (position !== null) card.appendChild(element("div", "seq", String(position + 1)));
  if (info.favorite) card.appendChild(element("div", "flag", "★"));

  const body = element("div", "body");
  const when = info.taken_at_local ? info.taken_at_local.slice(0, 10) : "no date";
  body.appendChild(element("div", "why", info.reason.replace(/_/g, " ")));
  const detail = [when];
  if (info.people.length) detail.push(info.people.slice(0, 2).join(", "));
  if (info.has_gps) detail.push("GPS");
  body.appendChild(element("div", null, detail.join(" · ")));
  card.appendChild(body);
  return card;
}

function action(label, handler) {
  const button = element("button", null, label);
  button.type = "button";
  button.addEventListener("click", handler);
  return button;
}

/* ------------------------------------------------------------------- edits */

async function edit(payload) {
  try {
    adopt(await post("/api/edit", Object.assign({ session_id: state.session_id }, payload)));
  } catch (error) {
    say($("banner"), error.message, true);
    $("banner").hidden = false;
  }
}

/* Native HTML5 drag and drop. A sortable library would be the fifth
   dependency of a project that has four; the drop target is computed from the
   card under the pointer and the whole order is sent to the server, which
   refuses anything that is not a permutation. */
function wireDragging(grid) {
  let dragged = null;
  grid.addEventListener("dragstart", (event) => {
    const card = event.target.closest(".card");
    if (!card) return;
    dragged = card;
    card.classList.add("dragging");
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", card.dataset.hash);
  });
  grid.addEventListener("dragend", () => {
    if (dragged) dragged.classList.remove("dragging");
    grid.querySelectorAll(".drop-target").forEach((n) => n.classList.remove("drop-target"));
    dragged = null;
  });
  grid.addEventListener("dragover", (event) => {
    const card = event.target.closest(".card");
    if (!card || !dragged || card === dragged) return;
    event.preventDefault();
    grid.querySelectorAll(".drop-target").forEach((n) => n.classList.remove("drop-target"));
    card.classList.add("drop-target");
  });
  grid.addEventListener("drop", (event) => {
    const card = event.target.closest(".card");
    if (!card || !dragged || card === dragged) return;
    event.preventDefault();
    const order = state.order.slice();
    const from = order.indexOf(dragged.dataset.hash);
    const to = order.indexOf(card.dataset.hash);
    if (from < 0 || to < 0) return;
    order.splice(to, 0, order.splice(from, 1)[0]);
    edit({ op: "reorder", order });
  });
}

/* ------------------------------------------------------ finding more photos */

$("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $("search").value.trim();
  if (!query) return;
  const mode = document.querySelector('input[name="mode"]:checked').value;
  loadInto(`/api/search?q=${encodeURIComponent(query)}&mode=${mode}`, `“${query}”`);
});

async function loadInto(path, what) {
  say($("search-note"), `looking for ${what}…`);
  try {
    const payload = await api(`${path}${path.includes("?") ? "&" : "?"}t=${encodeURIComponent(TOKEN)}`);
    results = payload.hits || [];
    say($("search-note"),
      `${results.length} photos for ${what}` +
      (payload.day ? ` (${payload.day})` : "") +
      (payload.mode ? ` · ${payload.mode}` : "") +
      ". Anything the guardrails withhold is simply not here.");
  } catch (error) {
    results = [];
    say($("search-note"), error.message, true);
  }
  drawResults();
}

function drawResults() {
  const grid = $("results");
  grid.replaceChildren();
  const chosen = new Set(state ? state.order : []);
  results.forEach((hit) => {
    const info = candidate(hit.file_hash) || {
      file_hash: hit.file_hash, caption: "", taken_at_local: null,
      state: "available", reason: hit.why || "found", name: hit.file_hash.slice(0, 12),
      people: [], albums: [], has_gps: false, favorite: false, burst: [], instead_of: "",
    };
    const card = photoCard(info, {});
    const actions = element("div", "actions");
    if (chosen.has(hit.file_hash)) {
      actions.appendChild(element("span", "muted", "already in the memory"));
    } else {
      actions.appendChild(action("Put it in", () => edit({ op: "add", file_hash: hit.file_hash })));
    }
    card.appendChild(actions);
    grid.appendChild(card);
  });
}

function showBurst(info) {
  results = info.burst.map((hash) => ({ file_hash: hash, why: hash === info.file_hash ? "kept" : "collapsed as a near-duplicate" }));
  say($("search-note"), `${info.burst.length} frames were taken as one burst; rekindle kept the sharpest. Pick another and it swaps in place.`);
  const grid = $("results");
  grid.replaceChildren();
  info.burst.forEach((hash) => {
    const detail = candidate(hash) || { file_hash: hash, caption: "", taken_at_local: null, state: "available", reason: hash === info.file_hash ? "kept" : "near duplicate", name: hash.slice(0, 12), people: [], albums: [], has_gps: false, favorite: false, burst: [], instead_of: "" };
    const card = photoCard(detail, {});
    const actions = element("div", "actions");
    if (hash === info.file_hash) {
      actions.appendChild(element("span", "muted", "this is the one in the memory"));
    } else {
      actions.appendChild(action("Use this frame instead", () => edit({ op: "swap", out: info.file_hash, in: hash })));
    }
    card.appendChild(actions);
    grid.appendChild(card);
  });
  $("results").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ------------------------------------------------------- pace and rendering */

function drawPace() {
  $("frame-ms").value = state.pace.frame_ms;
  $("title-ms").value = state.pace.title_ms;
  $("music").value = state.pace.music || "";
}

["frame-ms", "title-ms"].forEach((id) => {
  $(id).addEventListener("change", () => edit({
    op: "pace",
    frame_ms: Number($("frame-ms").value),
    title_ms: Number($("title-ms").value),
  }));
});
$("music").addEventListener("change", () => edit({ op: "pace", music: $("music").value }));

$("render").addEventListener("click", async () => {
  $("render").disabled = true;
  say($("render-note"), $("no-mp4").checked
    ? "rendering the WebP and GIF…"
    : "rendering… the MP4 pass decodes every shot at full size and can take a minute. " +
      "Tick “skip the MP4” for a quick look.");
  try {
    adopt(await post("/api/render", { session_id: state.session_id, no_mp4: $("no-mp4").checked }));
    const result = state.last_render;
    say($("render-note"),
      `${result.shots} shots (${result.preview_rendered} in the preview) · WebP ${Math.round(result.webp_bytes / 1024)} KB` +
      (result.mp4 ? ` · MP4 ${Math.round(result.mp4_bytes / 1024)} KB` : result.mp4_skipped ? " · no MP4 (ffmpeg not on PATH)" : "") +
      (result.withheld ? ` · ${result.withheld} shots withheld by the guardrails` : "") +
      ` → ${result.folder}`);
    showPreview(result);
  } catch (error) {
    say($("render-note"), error.message, true);
  }
  $("render").disabled = false;
});

$("dismiss").addEventListener("click", async () => {
  if (!confirm("Dismissing is permanent: this memory is never offered again. Undo it with `rekindle undismiss`.")) return;
  try { adopt(await post("/api/dismiss", { session_id: state.session_id })); }
  catch (error) { say($("render-note"), error.message, true); }
});

function showPreview(result) {
  const box = $("preview");
  box.replaceChildren();
  const base = `/api/output/${state.session_id}`;
  const stamp = Date.now(); // a re-render writes the same names; defeat the cache
  if (result.webp) {
    const image = document.createElement("img");
    image.src = `${base}/${result.webp}?t=${encodeURIComponent(TOKEN)}&v=${stamp}`;
    image.alt = "the memory, as an animated WebP";
    box.appendChild(image);
  }
  if (result.mp4) {
    const video = document.createElement("video");
    video.controls = true;
    video.src = `${base}/${result.mp4}?t=${encodeURIComponent(TOKEN)}&v=${stamp}`;
    box.appendChild(video);
  }
}

function drawReproduce() {
  const box = $("reproduce");
  text($("origin-cmd"), shell(state.origin.command));
  if (!state.reproduce) { box.hidden = true; return; }
  box.hidden = false;
  text($("reproduce-cmd"), shell(state.reproduce));
}

/* Quote for a shell. Double quotes rather than single: cmd.exe, PowerShell and
   every POSIX shell agree on them, and single quotes are literal in cmd.exe. */
function shell(argv) {
  return (argv || []).map((token) =>
    /[\s"'`$&|<>^();]/.test(token) ? `"${token.replace(/(["\\$`])/g, "\\$1")}"` : token
  ).join(" ");
}

/* ------------------------------------------------------------- the reports */

function drawGuardrails() {
  const box = $("guardrails");
  box.replaceChildren();
  const g = state.guardrails;
  box.appendChild(element("div", null,
    `${g.considered} candidates · ${g.kept} survived composition · ` +
    `${g.collapsed} collapsed as near-duplicates · orientation ${g.orientation}`));

  if (Object.keys(g.dropped).length) {
    const table = document.createElement("table");
    Object.entries(g.dropped).forEach(([reason, n]) => {
      const row = table.insertRow();
      row.insertCell().outerHTML = `<td class="n">${n}</td>`;
      text(row.insertCell(), reason.replace(/_/g, " ") + (g.examples[reason] ? ` (e.g. ${g.examples[reason]})` : ""));
    });
    box.appendChild(table);
  }
  if (state.strata && state.strata.dimension) {
    const s = state.strata;
    box.appendChild(element("div", "muted",
      `spread across ${s.used} of ${s.offered} ${s.dimension}s` +
      (s.lost_to_gates.length ? ` · no usable photos from ${s.lost_to_gates.slice(0, 8).join(", ")}` : "") +
      (s.without_slots.length ? ` · ${s.without_slots.length} more had photos but no room` : "")));
  }
  const facts = state.facts;
  box.appendChild(element("div", "muted",
    `${facts.photo_count} shots · years ${(facts.years || []).join(", ") || "none"}` +
    (Object.keys(facts.people || {}).length ? ` · people ${Object.keys(facts.people).slice(0, 5).join(", ")}` : "")));
}

/* ------------------------------------------------------------- detail sheet */

function openSheet(info) {
  $("sheet-img").src = thumb(info.file_hash, 900);
  $("sheet-img").alt = info.caption || info.name;
  const body = $("sheet-body");
  body.replaceChildren();
  const rows = [
    ["file", info.name],
    ["taken", info.taken_at_local || "unknown"],
    ["caption", info.caption || "(none)"],
    ["why", info.reason.replace(/_/g, " ")],
    ["size", info.width && info.height ? `${info.width} × ${info.height}` : "unknown"],
    ["people", info.people.length ? info.people.join(", ") : "no face tags"],
    ["albums", info.albums.length ? info.albums.join(", ") : "none"],
    ["public-safe", info.public_safe ? "yes" : "no"],
  ];
  rows.forEach(([label, value]) => body.appendChild(element("div", null, `${label}: ${value}`)));
  $("sheet").hidden = false;
}

$("sheet-close").addEventListener("click", () => { $("sheet").hidden = true; });
$("sheet").addEventListener("click", (event) => { if (event.target === $("sheet")) $("sheet").hidden = true; });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") $("sheet").hidden = true; });

$("prompt-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = $("prompt").value.trim();
  if (prompt) build({ prompt });
});

loadStatus();

/* --------------------------------------------------------------- calibrate
 *
 * A guided sequence, not a settings screen. One question at a time, "4 of 9"
 * in the header, and an end.
 *
 * The page holds NO calibration logic. It does not know which photograph
 * comes next, what the answers derive to, or whether the publishing gate may
 * be widened - it asks and it renders. That is what lets the whole flow be
 * tested by calling functions, and it is what stops the safeguard existing on
 * the terminal side only.
 *
 * The measured value behind each photograph is never sent to the browser. A
 * blind judgement stops being blind the moment the page can render the
 * number, and one debugging console.log is all that would take.
 */

let cal = null;        // the last /api/calibrate payload
let calStep = null;    // the step being worked on
let calMode = "";      // blind | slider | default
let calValue = null;   // the number the slider or a derivation is proposing

const calFail = (error) => say($("cal-area"), error.message, true);

function calStepFor(setting) {
  return (cal.steps || []).find((s) => s.setting === setting) || null;
}

async function calStatus() {
  cal = await api("/api/calibrate");
  const offer = cal.offer || {};
  $("calibrate-offer").hidden = !offer.show;
  text($("calibrate-headline"), offer.headline);
  return cal;
}

function calNextTodo() {
  return (cal.steps || []).find((s) => s.state === "todo") || null;
}

async function calBegin() {
  cal = await post("/api/calibrate/begin", {});
  $("calibrate-offer").hidden = true;
  $("calibrate").hidden = false;
  await calGo(calNextTodo());
}

async function calGo(step) {
  if (!step) return calSummarise();
  calStep = step;
  calValue = null;
  $("cal-finish").hidden = true;
  $("cal-refusal").hidden = true;
  text($("cal-title"), step.title);
  text($("cal-progress"), step.position + " of " + cal.total_steps);
  text($("cal-area"), step.area + ". " + step.what);
  calRenderModes(step);
  await calSetMode(step.modes[0]);
}

function calRenderModes(step) {
  const labels = {
    blind: "show me photographs",
    slider: "let me pick a number",
    default: "what does rekindle's value do?",
  };
  const box = $("cal-modes");
  box.replaceChildren();
  step.modes.forEach((mode) => {
    const chip = element("button", "chip", labels[mode] || mode);
    chip.type = "button";
    chip.addEventListener("click", () => calSetMode(mode).catch(calFail));
    box.appendChild(chip);
  });
}

async function calSetMode(mode) {
  calMode = mode;
  Array.from($("cal-modes").children).forEach((chip, i) =>
    chip.classList.toggle("on", calStep.modes[i] === mode)
  );
  $("cal-blind").hidden = mode !== "blind";
  $("cal-slider").hidden = mode !== "slider";
  $("cal-default").hidden = mode !== "default";
  $("cal-apply").hidden = true;
  if (mode === "blind") return calAsk();
  if (mode === "slider") return calSlider();
  return calDefault();
}

/* -- blind judgement */

function calShow(payload) {
  text($("cal-confidence"), payload.confidence || "");
  if (payload.derived !== null && payload.derived !== undefined) calPropose(payload.derived);
  if (payload.exhausted) {
    text($("cal-question"), "That is every example this library can offer.");
    $("cal-photos").replaceChildren();
    return;
  }
  text($("cal-question"), payload.question);
  text($("cal-note"), calStep.note || "");
  text($("cal-caption"), payload.caption || "");
  text($("cal-yes"), payload.yes);
  text($("cal-no"), payload.no);
  const photos = $("cal-photos");
  photos.replaceChildren();
  payload.hashes.forEach((hash) => {
    const img = document.createElement("img");
    img.src = thumb(hash, 900);
    img.alt = "";
    photos.appendChild(img);
  });
}

async function calAsk() {
  calShow(await api("/api/calibrate/example?setting=" + encodeURIComponent(calStep.setting)));
}

async function calAnswer(saidYes) {
  calShow(await post("/api/calibrate/answer", { setting: calStep.setting, said_yes: saidYes }));
}

/* -- slider, with the effect recomputed as it moves */

async function calSlider() {
  const step = calStep;
  const range = $("cal-range");
  const lo = step.minimum === null ? step.default / 4 : step.minimum;
  const hi = step.maximum === null ? step.default * 4 : step.maximum;
  // 200 stops across whatever range the setting declares, so a float
  // threshold gets real resolution and an int one still lands on integers
  // once the server floors it.
  range.min = lo;
  range.max = hi;
  range.step = (hi - lo) / 200;
  range.value = step.value;
  text($("cal-tradeoff"), "Raising it: " + step.raising + "  ·  Lowering it: " + step.lowering);
  await calShowConsequence(Number(range.value));
}

let calConsequenceTimer = null;
function calSliderMoved() {
  const value = Number($("cal-range").value);
  text($("cal-value"), value.toPrecision(4));
  // Debounced: the consequence runs the REAL composition gate over the whole
  // library, and firing it on every pixel of a drag would make the slider
  // feel broken.
  clearTimeout(calConsequenceTimer);
  calConsequenceTimer = setTimeout(() => calShowConsequence(value).catch(calFail), 180);
}

async function calShowConsequence(value) {
  calPropose(value);
  text($("cal-value"), Number(value).toPrecision(4));
  const payload = await api(
    "/api/calibrate/consequence?setting=" + encodeURIComponent(calStep.setting) + "&value=" + value
  );
  if (!payload.countable) {
    text($("cal-consequence"), payload.why);
    $("cal-effect").replaceChildren();
    return;
  }
  text($("cal-consequence"), payload.sentence);
  const box = $("cal-effect");
  box.replaceChildren();
  const show = (hashes, className, label) =>
    hashes.forEach((hash) => {
      const figure = element("figure");
      const img = document.createElement("img");
      img.src = thumb(hash, 320);
      img.alt = "";
      img.className = className;
      figure.appendChild(img);
      figure.appendChild(element("figcaption", null, label));
      box.appendChild(figure);
    });
  show(payload.newly_dropped, "dropped", "would be dropped");
  show(payload.newly_kept, "kept", "would be kept");
}

/* -- show the default's effect, accept or nudge */

async function calDefault() {
  const step = calStep;
  text($("cal-default-value"), "rekindle's value is " + step.default + " (" + step.unit + ").");
  text($("cal-measured"), step.measured);
  const payload = await api(
    "/api/calibrate/consequence?setting=" +
      encodeURIComponent(step.setting) +
      "&value=" +
      step.default
  );
  text(
    $("cal-default-effect"),
    payload.countable
      ? "On your library that keeps " + payload.kept_now.toLocaleString() + " photographs."
      : payload.why
  );
}

/* -- committing */

function calPropose(value) {
  calValue = value;
  const button = $("cal-apply");
  button.hidden = false;
  text(button, "Use " + Number(value).toPrecision(4));
}

async function calApply() {
  if (calValue === null) return;
  const payload = await post("/api/calibrate/choose", {
    setting: calStep.setting,
    value: calValue,
  });
  if (payload.refused) return calRefuse(payload.refused);
  cal = payload;
  await calGo(calNextTodo());
}

/* -- THE SAFEGUARD.
 *
 * The server refuses; the page shows the refusal and asks for the sentence
 * back. There is no boolean anywhere on this path, which is the point: a
 * drag-and-release sends a number, and a number can never widen this gate.
 */

function calRefuse(refusal) {
  $("cal-refusal").hidden = false;
  text($("cal-refusal-meaning"), refusal.meaning);
  text($("cal-refusal-phrase"), refusal.phrase);
  $("cal-refusal-input").value = "";
  $("cal-refusal-input").focus();
}

async function calConfirm() {
  const payload = await post("/api/calibrate/confirm", {
    setting: calStep.setting,
    phrase: $("cal-refusal-input").value,
  });
  if (!payload.confirmed) {
    text($("cal-refusal-meaning"), "That is not the sentence. Nothing has been changed.");
    return;
  }
  $("cal-refusal").hidden = true;
  await calApply();
}

/* -- the way out */

function calSummarise() {
  $("cal-blind").hidden = true;
  $("cal-slider").hidden = true;
  $("cal-default").hidden = true;
  $("cal-apply").hidden = true;
  $("cal-modes").replaceChildren();
  $("cal-finish").hidden = false;
  text($("cal-title"), "Done");
  text($("cal-progress"), "");
  const overrides = cal.overrides || {};
  const names = Object.keys(overrides);
  const box = $("cal-summary");
  box.replaceChildren();
  if (!names.length) {
    box.appendChild(
      element(
        "p",
        null,
        "You kept every one of rekindle's defaults. That is a real result: they " +
          "were derived from a different library and they fit yours."
      )
    );
    return;
  }
  const table = document.createElement("table");
  names.forEach((name) => {
    const step = calStepFor(name);
    const row = table.insertRow();
    row.insertCell().textContent = name;
    row.insertCell().textContent = step ? step.default : "";
    row.insertCell().textContent = overrides[name];
    const why = row.insertCell();
    why.textContent = step ? step.confidence : "";
    why.className = "muted";
  });
  box.appendChild(table);
}

async function calWrite() {
  const written = await post("/api/calibrate/finish", {});
  const box = $("cal-affected");
  box.replaceChildren();
  box.appendChild(
    element("p", null, "Written to " + written.written + ". The previous file is at " + written.backup + ".")
  );
  const waiting = element(
    "p",
    "muted",
    "Re-running selection for every memory you have built — no frames, no encoding…"
  );
  box.appendChild(waiting);
  const report = await api("/api/calibrate/affected");
  waiting.remove();
  box.appendChild(element("p", null, report.sentence));
  if (report.affected.length) {
    const list = document.createElement("ul");
    report.affected.forEach((change) => {
      const bits = [];
      if (change.gained) bits.push("+" + change.gained);
      if (change.lost) bits.push("−" + change.lost);
      list.appendChild(
        element(
          "li",
          null,
          change.memory_id + ": " + change.before + " → " + change.after + "  " + bits.join(" ")
        )
      );
    });
    box.appendChild(list);
    box.appendChild(
      element(
        "p",
        "muted",
        "Rebuilding is the expensive half. Only these are worth redoing; everything " +
          "else would come out of the renderer byte-identical."
      )
    );
  }
  report.uncheckable.forEach((change) =>
    box.appendChild(element("p", "muted", change.memory_id + ": " + change.note))
  );
}

$("calibrate-start").addEventListener("click", () => calBegin().catch(calFail));
$("calibrate-later").addEventListener("click", () => {
  $("calibrate-offer").hidden = true;
});
$("calibrate-never").addEventListener("click", () =>
  post("/api/calibrate/not-now", {})
    .then(() => {
      $("calibrate-offer").hidden = true;
    })
    .catch(calFail)
);
$("cal-quit").addEventListener("click", () => {
  $("calibrate").hidden = true;
});
$("cal-yes").addEventListener("click", () => calAnswer(true).catch(calFail));
$("cal-no").addEventListener("click", () => calAnswer(false).catch(calFail));
$("cal-range").addEventListener("input", calSliderMoved);
$("cal-apply").addEventListener("click", () => calApply().catch(calFail));
$("cal-accept").addEventListener("click", () =>
  post("/api/calibrate/choose", { setting: calStep.setting, accept: true })
    .then((payload) => {
      cal = payload;
      return calGo(calNextTodo());
    })
    .catch(calFail)
);
$("cal-skip").addEventListener("click", () =>
  post("/api/calibrate/skip", { setting: calStep.setting })
    .then((payload) => {
      cal = payload;
      return calGo(calNextTodo());
    })
    .catch(calFail)
);
$("cal-redo").addEventListener("click", () =>
  post("/api/calibrate/reset", { setting: calStep.setting })
    .then((payload) => {
      cal = payload;
      return calGo(calStepFor(calStep.setting));
    })
    .catch(calFail)
);
$("cal-refusal-go").addEventListener("click", () => calConfirm().catch(calFail));
$("cal-refusal-cancel").addEventListener("click", () => {
  $("cal-refusal").hidden = true;
});
$("cal-write").addEventListener("click", () => calWrite().catch(calFail));

calStatus().catch(() => {
  /* The library is still loading, or this build has no calibration state.
     Either way the offer simply does not appear - it must never be the thing
     that stops someone opening their photographs. */
});
