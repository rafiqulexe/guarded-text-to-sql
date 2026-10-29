// sqlguard UI. Everything from the API (question, SQL, row values, error text, model names) is
// untrusted: it is only ever written with textContent / createTextNode, never parsed as HTML.
"use strict";

(() => {
  const config = JSON.parse(document.getElementById("config").textContent);
  const $ = (id) => document.getElementById(id);

  const RUNNING = "RUNNING";
  const STATUS_CLASS = {
    SUCCEEDED: "chip-ok",
    BLOCKED: "chip-danger",
    FAILED: "chip-danger",
    AWAITING_CONFIRMATION: "chip-warn",
    UNANSWERABLE: "chip-muted",
    CANCELLED: "chip-muted",
  };
  const SIGNALS = [
    ["judge", "Judge verdict"],
    ["consistency", "Self-consistency"],
    ["repairs", "Repairs used"],
    ["sanity", "Result sanity"],
  ];

  let current = null; // the run on screen

  // --- DOM helper: text only ----------------------------------------------------------------------

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function show(id, visible) {
    $(id).hidden = !visible;
  }

  // --- API ----------------------------------------------------------------------------------------

  async function call(method, path, body) {
    const response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (!response.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`;
      throw new Error(detail);
    }
    return data;
  }

  function setError(message) {
    $("error").textContent = message || "";
    show("error", Boolean(message));
  }

  function setBusy(busy) {
    for (const id of ["ask-btn", "confirm-btn", "cancel-btn"]) $(id).disabled = busy;
  }

  async function ask() {
    const question = $("question").value;
    if (!question.trim() || $("ask-btn").disabled) return;
    setError("");
    setBusy(true);
    renderRunning(question);
    try {
      render(await call("POST", "/api/ask", { question }));
    } catch (err) {
      show("run", false);
      setError(`Request failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function act(action) {
    if (!current) return;
    setError("");
    setBusy(true);
    try {
      render(await call("POST", `/api/runs/${encodeURIComponent(current.run_id)}/${action}`));
    } catch (err) {
      setError(`${action === "confirm" ? "Show results" : "Cancel"} failed: ${err.message}`);
      try {
        render(await call("GET", `/api/runs/${encodeURIComponent(current.run_id)}`));
      } catch {
        /* keep what is on screen */
      }
    } finally {
      setBusy(false);
    }
  }

  // --- rendering ----------------------------------------------------------------------------------

  function renderRunning(question) {
    current = null;
    show("run", true);
    setStatus(RUNNING);
    clear($("badge-wrap"));
    $("run-meta").textContent = "";
    $("run-question").textContent = question;
    for (const id of ["confirm-bar", "blocked-panel", "failed-panel", "unanswerable-panel",
      "cancelled-panel", "sql-card", "results"]) show(id, false);
    clear($("transitions"));
    clear($("timeline"));
    $("models").textContent = "";
  }

  function render(run) {
    current = run;
    show("run", true);
    setStatus(run.status);
    renderBadge(run);
    renderMeta(run);
    $("run-question").textContent = run.question;

    const status = run.status;
    const injection = (run.input_flags || []).includes("possible_injection");

    show("confirm-bar", status === "AWAITING_CONFIRMATION");
    $("confirm-text").textContent = injection
      ? "Possible prompt injection — review the SQL"
      : "Low confidence — review the SQL";

    show("blocked-panel", status === "BLOCKED");
    if (status === "BLOCKED") {
      const event = blockingEvent(run);
      $("blocked-rule").textContent = event ? event.rule : "guard";
      $("blocked-detail").textContent = stripCandidate(event ? event.detail : run.detail);
    }

    show("failed-panel", status === "FAILED");
    if (status === "FAILED") $("failed-detail").textContent = run.error || run.detail || "";

    show("unanswerable-panel", status === "UNANSWERABLE");
    if (status === "UNANSWERABLE") $("unanswerable-reason").textContent = stripCandidate(run.detail);

    show("cancelled-panel", status === "CANCELLED");

    // Always show the SQL when there is any; UNANSWERABLE has none.
    const hasSql = Boolean(run.sql) && status !== "UNANSWERABLE";
    show("sql-card", hasSql);
    if (hasSql) renderSql(run.sql);

    // Rows only for SUCCEEDED: never for BLOCKED or while awaiting confirmation.
    const rowsVisible = status === "SUCCEEDED" && Array.isArray(run.rows) && Array.isArray(run.columns);
    show("results", rowsVisible);
    if (rowsVisible) renderTable(run.columns, run.rows, run.limit);
    else clearTable();

    renderTimeline(run);
  }

  function setStatus(status) {
    const chip = $("status");
    chip.className = `chip ${STATUS_CLASS[status] || "chip-info"}`;
    chip.textContent = status === RUNNING ? "Running" : status.replace(/_/g, " ");
  }

  function badgeLevel(score) {
    if (score >= config.confidenceHigh) return ["HIGH", "badge-high"];
    if (score >= config.confidenceThreshold) return ["MEDIUM", "badge-medium"];
    return ["LOW – CONFIRM", "badge-low"];
  }

  function renderBadge(run) {
    const wrap = $("badge-wrap");
    clear(wrap);
    if (typeof run.confidence !== "number") return;
    const [label, cls] = badgeLevel(run.confidence);
    const badge = el("button", `badge ${cls}`);
    badge.type = "button";
    badge.setAttribute("aria-expanded", "false");
    badge.setAttribute("aria-label", `Confidence ${label} ${run.confidence.toFixed(2)}, show breakdown`);
    badge.append(el("span", null, label), el("span", "score", run.confidence.toFixed(2)));
    wrap.append(badge);

    const pop = el("div", "popover");
    pop.hidden = true;
    pop.setAttribute("role", "dialog");
    pop.append(el("p", "popover-title", "Confidence breakdown"));
    const list = el("ul", "signals");
    const breakdown = run.breakdown || {};
    for (const [key, name] of SIGNALS) {
      const value = Number(breakdown[key] ?? 0);
      const item = el("li", "signal");
      item.append(
        el("span", null, `${name} ×${Number(config.weights[key]).toFixed(2)}`),
        el("span", "signal-value", value.toFixed(2)),
      );
      const bar = el("span", "signal-bar");
      const fill = el("span");
      fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
      bar.append(fill);
      item.append(bar);
      list.append(item);
    }
    pop.append(list);
    if (run.judge_verdict) {
      pop.append(el("p", "panel-body small muted", `Judge: ${run.judge_verdict} — ${run.judge_reason || ""}`));
    }
    wrap.append(pop);
    pop.addEventListener("click", (event) => event.stopPropagation());

    badge.addEventListener("click", (event) => {
      event.stopPropagation();
      pop.hidden = !pop.hidden;
      badge.setAttribute("aria-expanded", String(!pop.hidden));
    });
  }

  function renderMeta(run) {
    const parts = [`run ${String(run.run_id).slice(0, 8)}`];
    if (typeof run.latency_ms === "number") parts.push(`${(run.latency_ms / 1000).toFixed(1)} s`);
    const tokens = (run.input_tokens || 0) + (run.output_tokens || 0);
    if (tokens) parts.push(`${tokens.toLocaleString()} tokens`);
    if (run.cost_usd_list) parts.push(`$${Number(run.cost_usd).toFixed(4)} (list $${Number(run.cost_usd_list).toFixed(4)})`);
    $("run-meta").textContent = parts.join(" · ");
  }

  function blockingEvent(run) {
    const events = run.guard_events || [];
    return (
      events.find((e) => !e.passed && (e.severity === "block" || e.severity === "critical")) ||
      events.find((e) => !e.passed) ||
      null
    );
  }

  function stripCandidate(text) {
    return String(text || "").replace(/^\[#\d+\]\s*/, "").replace(/^candidate #\d+:\s*/, "");
  }

  // --- SQL card (tokenised into text nodes; no HTML) ----------------------------------------------

  const KEYWORDS = new Set((
    "select from where and or not in is null as join inner left right full outer cross on group by " +
    "order having limit offset distinct union all intersect except with case when then else end asc " +
    "desc between like ilike exists any some true false interval over partition fetch first next rows " +
    "only cast filter within lateral using natural current_date current_timestamp extract date"
  ).split(" "));
  const SQL_TOKEN = /(--[^\n]*|\/\*[\s\S]*?\*\/)|('(?:[^']|'')*')|("(?:[^"]|"")*")|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_$]*)/g;

  function renderSql(sql) {
    const code = $("sql");
    clear(code);
    let last = 0;
    for (const match of sql.matchAll(SQL_TOKEN)) {
      if (match.index > last) code.append(document.createTextNode(sql.slice(last, match.index)));
      const [text, comment, string, quoted, number, word] = match;
      let cls = null;
      if (comment) cls = "tok-comment";
      else if (string) cls = "tok-string";
      else if (quoted) cls = "tok-ident";
      else if (number) cls = "tok-number";
      else if (word) cls = KEYWORDS.has(word.toLowerCase()) ? "tok-keyword" : "tok-ident";
      code.append(el("span", cls, text));
      last = match.index + text.length;
    }
    if (last < sql.length) code.append(document.createTextNode(sql.slice(last)));
  }

  async function copySql() {
    if (!current || !current.sql) return;
    const button = $("copy-btn");
    try {
      await navigator.clipboard.writeText(current.sql);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Copy failed";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 1500);
  }

  // --- results table ------------------------------------------------------------------------------

  const NUMERIC = /^-?\d+(\.\d+)?$/;

  function clearTable() {
    clear($("results-head"));
    clear($("results-body"));
    $("results-note").textContent = "";
  }

  function cellText(value) {
    if (value === null || value === undefined) return "NULL";
    if (Array.isArray(value)) return `{${value.map(cellText).join(",")}}`;
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function renderTable(columns, rows, limit) {
    clearTable();
    const head = el("tr");
    for (const name of columns) head.append(el("th", null, name));
    $("results-head").append(head);

    const body = document.createDocumentFragment();
    for (const row of rows) {
      const tr = el("tr");
      for (const value of row) {
        const numeric = typeof value === "number" || (typeof value === "string" && NUMERIC.test(value));
        const cls = value === null ? "null" : numeric ? "num" : null;
        tr.append(el("td", cls, cellText(value)));
      }
      body.append(tr);
    }
    $("results-body").append(body);

    const n = rows.length;
    const rowsText = `${n.toLocaleString()} row${n === 1 ? "" : "s"}`;
    $("results-note").textContent =
      typeof limit === "number" ? `showing ${rowsText} of LIMIT ${limit.toLocaleString()}` : `showing ${rowsText}`;
  }

  // --- guard timeline + transitions + models ------------------------------------------------------

  function renderTimeline(run) {
    const transitions = $("transitions");
    clear(transitions);
    const history = run.transitions || [];
    const start = history.length ? Date.parse(history[0].at) : 0;
    history.forEach((t, i) => {
      const item = el("li", null, t.state);
      if (i === history.length - 1) {
        const cls = STATUS_CLASS[t.state];
        if (cls === "chip-ok") item.classList.add("final-ok");
        else if (cls === "chip-danger") item.classList.add("final-danger");
        else if (cls === "chip-warn") item.classList.add("final-warn");
      }
      if (i > 0 && start) item.append(el("span", "muted", `+${Date.parse(t.at) - start} ms`));
      if (t.detail) item.title = t.detail;
      transitions.append(item);
    });

    const timeline = $("timeline");
    clear(timeline);
    for (const event of run.guard_events || []) {
      const blocked = !event.passed && (event.severity === "block" || event.severity === "critical");
      const flagged = event.passed && event.severity === "warn";
      const kind = event.passed ? (flagged ? "event-flag" : "event-pass") : blocked ? "event-blocked" : "event-repair";
      const item = el("li", `event ${kind}`);
      item.append(el("span", "event-icon", event.passed ? (flagged ? "!" : "✓") : "✗"));
      const head = el("div", "event-head");
      head.append(el("span", "event-stage", event.stage), el("span", "event-rule", event.rule));
      if (blocked) head.append(el("span", "event-tag", event.severity === "critical" ? "blocked · critical" : "blocked"));
      else if (!event.passed) head.append(el("span", "event-tag muted", "repair / dropped"));
      else if (flagged) head.append(el("span", "event-tag muted", "flag"));
      item.append(head);
      if (event.detail) item.append(el("p", "event-detail", event.detail));
      timeline.append(item);
    }

    $("models").textContent = modelsText(run);
  }

  function modelsText(run) {
    const describe = (calls) => {
      const seen = new Map();
      for (const c of calls) {
        if (!c.model) continue;
        const label = c.fallback_from ? `${c.model} (fallback from ${c.fallback_from})` : c.model;
        seen.set(label, true);
      }
      return [...seen.keys()].join(", ");
    };
    const calls = run.llm_calls || [];
    const sqlBy = describe(calls.filter((c) => c.purpose === "generate" || c.purpose === "repair"));
    const judgedBy = describe(calls.filter((c) => c.purpose === "judge"));
    const parts = [];
    if (sqlBy) parts.push(`SQL by ${sqlBy}`);
    if (judgedBy) parts.push(`judged by ${judgedBy}`);
    return parts.join(" · ");
  }

  // --- events -------------------------------------------------------------------------------------

  $("ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    ask();
  });
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      ask();
    }
  });
  $("question").addEventListener("input", () => {
    $("count").textContent = String($("question").value.length);
  });
  $("confirm-btn").addEventListener("click", () => act("confirm"));
  $("cancel-btn").addEventListener("click", () => act("cancel"));
  $("copy-btn").addEventListener("click", copySql);
  document.addEventListener("click", () => {
    for (const pop of document.querySelectorAll(".popover")) pop.hidden = true;
  });
})();
