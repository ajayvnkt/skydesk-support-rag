// SkyDesk Airways demo: mock site interactions + docked support chat over POST /api/ask.
const $ = (s, el = document) => el.querySelector(s);
const msgs = $("#msgs"), input = $("#input"), send = $("#send"), chat = $("#chat"), launcher = $("#launcher");
const CHIPS = [
  "getting error SD-ERR-112 when i try to pay, what does that mean??",
  "How many kg can i check in on SkyClassic?",
  "is there wifi on the plane and how much does it cost?",
  "My dad passed away last night and i need to fly home for the funeral tmrw. do you have a bereavment discount?",
];
const CHIP_LABELS = ["Error SD-ERR-112 at payment", "Checked bag allowance on SkyClassic", "Is there wifi on board?", "Bereavement fare"];
let busy = false, lastFocus = null;

function sessionId() {
  let id = null;
  try { id = sessionStorage.getItem("skydesk-sid"); } catch {}
  if (!id) {
    id = "w" + Math.random().toString(36).slice(2, 12);
    try { sessionStorage.setItem("skydesk-sid", id); } catch {}
  }
  return id;
}
const SID = sessionId();

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Light markdown: paragraphs, "- " / "1. " lists, **bold**, and [doc-id] markers -> numbered refs.
function md(text, refs) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\[([a-z0-9][\w-]*(?:\s*[,;]\s*[a-z0-9][\w-]*)*)\]/gi, (m, ids) => {
      const nums = ids.split(/\s*[,;]\s*/).filter((d) => refs[d]).map((d) => `<span class="ref" title="Source ${refs[d]}">${refs[d]}</span>`);
      return nums.length ? nums.join("") : "";
    });
  return String(text || "").trim().split(/\n{2,}/).map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l)))
      return "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""))}</li>`).join("") + "</ul>";
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  }).join("");
}

function scrollDown() { msgs.scrollTop = msgs.scrollHeight; }

function addUser(text) {
  msgs.append(el("div", "m user", esc(text)));
  scrollDown();
}

function addBot(node) {
  const m = el("div", "m bot");
  m.append(node);
  msgs.append(m);
  scrollDown();
  return m;
}

function greet() {
  msgs.innerHTML = "";
  const b = el("div", "bubble", "<p>Hi, I'm the <b>SkyDesk Assistant</b>. I answer from the SkyDesk Help Center and show my sources. What can I help you with?</p>");
  const chips = el("div", "chips");
  CHIPS.forEach((q, i) => {
    const c = el("button", "chip", esc(CHIP_LABELS[i]));
    c.type = "button";
    c.dataset.q = q;
    c.onclick = () => ask(q);
    chips.append(c);
  });
  const m = addBot(b);
  m.append(chips);
}

function citeChips(cites, wrap) {
  const meta = $(".meta", wrap);
  cites.forEach((c, i) => {
    const chip = el("button", "cite-chip", `<b>${i + 1}</b>${esc(c.title)}`);
    chip.type = "button";
    chip.setAttribute("aria-expanded", "false");
    chip.title = "Show the source passage";
    let detail = null;
    chip.onclick = () => {
      if (detail) { detail.remove(); detail = null; chip.setAttribute("aria-expanded", "false"); return; }
      const rr = c.rerank == null ? "n/a" : `${c.rerank}/3`;
      const sec = (c.section || "").split(" > ").filter((x) => x !== c.title).pop();
      detail = el("div", "cite-detail", `<h4>${esc(c.title)}${sec ? " &rsaquo; " + esc(sec) : ""}</h4>
        <div class="src-meta">Help Center &middot; updated ${esc(c.updated || "n/a")} &middot; <code>${esc(c.doc_id)}</code></div>
        <blockquote>${md(c.snippet.length >= 600 ? c.snippet + "..." : c.snippet, {})}</blockquote>
        <details class="why"><summary>Why this source</summary>Keyword rank ${esc(c.bm25_rank ?? "-")} &middot; meaning rank ${esc(c.dense_rank ?? "-")} &middot; relevance check ${esc(rr)}</details>`);
      meta.after(detail);
      chip.setAttribute("aria-expanded", "true");
      detail.scrollIntoView({ block: "nearest", behavior: "smooth" });
    };
    meta.append(chip);
  });
}

function confPill(c) {
  if (c == null) return "";
  const cls = c >= 0.75 ? "" : " mid";
  return `<span class="conf-pill${cls}" title="Answer confidence ${c.toFixed(2)}"><i></i>${c >= 0.75 ? "High" : "Moderate"} confidence</span>`;
}

function actionBtn(label, icon, msg, cls = "btn-soft") {
  const b = el("button", "btn " + cls, `<svg><use href="#${icon}"/></svg>${esc(label)}`);
  b.type = "button";
  b.onclick = () => { b.disabled = true; addBot(el("div", "bubble", `<p>${esc(msg)}</p>`)); };
  return b;
}

function render(r) {
  const refs = {};
  (r.citations || []).forEach((c, i) => (refs[c.doc_id] = i + 1));
  const b = el("div", "bubble", md(r.answer, refs));
  const wrap = addBot(b);
  if (r.handoff) {
    const h = r.handoff;
    const card = el("div", "handoff", `<h4><span class="pulse"></span>Connecting you to the ${esc(h.team)}</h4>
      <dl><dt>Team</dt><dd>${esc(h.team)}</dd><dt>Reason</dt><dd>${esc(h.reason)}</dd>
      <dt>Priority</dt><dd><span class="prio">${esc(h.priority)}</span></dd></dl>
      <div class="small">Your message has been passed on, so you won't need to repeat yourself.</div>`);
    const act = el("div", "actions");
    act.append(actionBtn("Talk to an agent", "i-chat", `Demo: a ${h.team} agent would join this chat now. Typical wait: under 2 minutes.`, "btn-primary"),
      actionBtn("Request a callback", "i-phone", "Demo: a callback has been requested. We'd call you within 30 minutes."));
    card.append(act);
    wrap.append(card);
  } else if (r.route === "escalate") {
    const act = el("div", "actions");
    act.append(actionBtn("Ask a human", "i-user", "Demo: you're in the queue for a SkyDesk agent. They can see this conversation."));
    wrap.append(act);
  }
  const meta = el("div", "meta", r.route === "answer" ? confPill(r.confidence) : "");
  if (r.route === "clarify") meta.innerHTML = '<span class="conf-pill mid"><i></i>Needs one more detail</span>';
  wrap.append(meta);
  citeChips(r.citations || [], wrap);
  if (!meta.children.length) meta.remove();
  scrollDown();
}

async function ask(q) {
  q = (q || "").trim();
  if (!q || busy) return;
  openChat();
  $(".chips", msgs)?.remove();
  busy = true; send.disabled = true; input.value = "";
  addUser(q);
  const typing = addBot(el("div", "bubble typing", "<i></i><i></i><i></i>"));
  typing.setAttribute("aria-label", "Assistant is typing");
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q, session_id: SID }) });
    const data = await res.json().catch(() => ({ error: "Sorry, something went wrong. Please try again." }));
    typing.remove();
    if (data.error) {
      const w = addBot(el("div", "bubble", `<p>${esc(data.error)}</p>`));
      const act = el("div", "actions");
      act.append(actionBtn("Ask a human", "i-user", "Demo: you're in the queue for a SkyDesk agent."));
      w.append(act);
    } else render(data);
    loadCost();
  } catch {
    typing.remove();
    addBot(el("div", "bubble", "<p>I can't reach the help center right now. Please check your connection and try again.</p>"));
  } finally {
    busy = false; send.disabled = false; input.focus();
  }
}

function openChat(prefill) {
  if (chat.hidden) { lastFocus = document.activeElement; chat.hidden = false; launcher.classList.add("hide"); launcher.setAttribute("aria-expanded", "true"); }
  if (prefill) { input.value = prefill; }
  input.focus();
}
function closeChat(reset) {
  chat.hidden = true; launcher.classList.remove("hide"); launcher.setAttribute("aria-expanded", "false");
  if (reset) greet();
  (lastFocus && lastFocus !== document.body ? lastFocus : launcher).focus();
}

let toastT;
function toast(text) {
  const t = $("#toast");
  t.textContent = text; t.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2600);
}

async function loadCost() {
  try {
    const s = await (await fetch("/api/cost")).json();
    $("#cost").textContent = `AI cost so far: $${s.total_usd.toFixed(4)} of $${s.budget_usd.toFixed(0)} cap (${s.calls} model calls).`;
  } catch { $("#cost").textContent = "AI cost so far: unavailable."; }
}

function openModal() { $("#modal").hidden = false; loadCost(); $("#modal [data-close]").focus(); }
function closeModal() { $("#modal").hidden = true; }

// Wire up: mock buttons toast, help-ish buttons open the chat with a question pre-filled.
document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-toast],[data-ask],[data-open-chat],[data-modal],[data-close]");
  if (!t) return;
  if (t.matches("[data-ask]")) openChat(t.dataset.ask);
  else if (t.matches("[data-open-chat]")) openChat();
  else if (t.matches("[data-modal]")) openModal();
  else if (t.matches("[data-close]")) closeModal();
  else toast("Demo site: nothing here is bookable. Try the assistant, bottom right.");
});
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#modal").hidden) closeModal(); else if (!chat.hidden) closeChat(false);
});
launcher.onclick = () => openChat();
$("#minBtn").onclick = () => closeChat(false);
$("#closeBtn").onclick = () => closeChat(true);
$("#composer").onsubmit = (e) => { e.preventDefault(); ask(input.value); };
$("#logLink").href = `/api/log?session_id=${SID}`;
document.querySelectorAll(".trip button").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".trip button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-checked", x === b); });
});

greet();
fetch("/api/health").then((r) => r.ok && $("#dot").classList.add("on")).catch(() => {});
loadCost();
