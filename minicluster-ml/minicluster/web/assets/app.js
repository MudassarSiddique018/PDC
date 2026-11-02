/* MiniCluster ML dashboard - vanilla JS, no build step. */
(function () {
  "use strict";

  const state = {
    info: null,
    cluster: { workers: [], online: 0, total: 0, cores: 0, running_tasks: 0, tasks_dispatched: 0, jobs: {} },
    jobs: new Map(),      // id -> summary or detail
    logs: [],
    datasets: [],
    route: "overview",
    jobId: null,
    token: safeGet("mc_token") || "",
  };

  // ------------------------------------------------------------ utils
  function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
  const esc = (v) => String(v == null ? "" : v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => `${Math.max(0, Math.min(100, v || 0)).toFixed(0)}%`;
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString());
  const secs = (v) => (v == null ? "-" : v < 1 ? `${(v * 1000).toFixed(0)} ms` : v < 120 ? `${v.toFixed(2)} s` : `${Math.floor(v / 60)}m ${(v % 60).toFixed(0)}s`);
  const bytes = (b) => (!b ? "-" : b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`);
  const time = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString() : "-");
  const dateTime = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : "-");
  const bar = (v, cls = "") => `<div class="bar ${cls}"><span style="width:${pct(v)}"></span></div>`;
  const pill = (s) => `<span class="pill ${esc(s)}">${esc(s)}</span>`;
  const TYPE_LABEL = { compute: "Level 1 · Compute", training: "Level 2 · ML Training", benchmark: "Level 3 · Benchmark" };

  function toast(msg, isError) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast" + (isError ? " error" : "");
    t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (t.hidden = true), 4500);
  }

  async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (state.token) headers.Authorization = "Bearer " + state.token;
    if (opts.json !== undefined) {
      headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.json);
    }
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    if (res.status === 401) { askToken(); throw new Error("API token required"); }
    const ct = res.headers.get("content-type") || "";
    const body = ct.includes("json") ? await res.json() : await res.text();
    if (!res.ok) {
      let msg = body && body.detail ? body.detail : body;
      if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return body;
  }

  function askToken() {
    const d = $("#token-dialog");
    if (!d.open) d.showModal();
  }
  $("#token-form").addEventListener("submit", () => {
    state.token = $("#token-input").value.trim();
    safeSet("mc_token", state.token);
    connect();
    loadInitial();
  });

  // --------------------------------------------------------- websocket
  let ws, wsRetry = 0;
  function connect() {
    if (ws) { try { ws.close(); } catch (e) { /* ignore */ } }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws${state.token ? "?token=" + encodeURIComponent(state.token) : ""}`);
    ws.onopen = () => { wsRetry = 0; setConn(true); };
    ws.onclose = (e) => {
      setConn(false);
      if (e.code === 4401) { askToken(); return; }
      setTimeout(connect, Math.min(10000, 500 * 2 ** wsRetry++));
    };
    ws.onmessage = (m) => handleEvent(JSON.parse(m.data));
  }
  function setConn(on) {
    $("#conn").className = "conn " + (on ? "conn-on" : "conn-off");
    $("#conn-text").textContent = on ? "Live" : "Reconnecting...";
  }

  function handleEvent(ev) {
    if (ev.type === "hello") {
      state.cluster = ev.data.cluster;
      ev.data.jobs.forEach((j) => state.jobs.set(j.id, Object.assign(state.jobs.get(j.id) || {}, j)));
      state.logs = ev.data.logs;
      renderAll();
    } else if (ev.type === "cluster") {
      state.cluster = ev.data;
      if (state.route === "overview") { renderKPIs(); renderWorkers(); renderActiveJob(); }
      renderTopbar();
    } else if (ev.type === "job") {
      const prev = state.jobs.get(ev.data.id);
      state.jobs.set(ev.data.id, ev.data);
      if (state.route === "overview") renderActiveJob();
      if (state.route === "jobs") renderJobsTable();
      if (state.route === "job" && state.jobId === ev.data.id) scheduleDetail();
      if (prev && prev.status !== ev.data.status && ["completed", "failed"].includes(ev.data.status)) {
        toast(`${ev.data.name}: ${ev.data.status}${ev.data.headline ? " - " + ev.data.headline : ev.data.error ? " - " + ev.data.error : ""}`, ev.data.status === "failed");
      }
    } else if (ev.type === "job_deleted") {
      state.jobs.delete(ev.data.id);
      if (state.route === "jobs") renderJobsTable();
    } else if (ev.type === "log") {
      state.logs.push(ev.data);
      if (state.logs.length > 400) state.logs.splice(0, state.logs.length - 400);
      renderLogs();
    }
  }

  // ------------------------------------------------------------ router
  const TITLES = { overview: "Cluster Overview", new: "New Job", jobs: "Jobs & Results", job: "Job Detail", predict: "Predict", logs: "Event Log" };
  function route() {
    const parts = (location.hash || "#/overview").slice(2).split("/");
    state.route = parts[0] === "jobs" && parts[1] ? "job" : parts[0] || "overview";
    if (!TITLES[state.route]) state.route = "overview";
    state.jobId = state.route === "job" ? parts[1] : null;
    $$(".page").forEach((p) => (p.hidden = p.id !== "page-" + state.route));
    $$("nav a[data-route]").forEach((a) => a.classList.toggle("active", a.dataset.route === (state.route === "job" ? "jobs" : state.route)));
    $("#page-title").textContent = TITLES[state.route];
    $(".navbar").classList.remove("open");
    renderAll();
  }
  window.addEventListener("hashchange", route);
  $("#menu-btn").addEventListener("click", () => $(".navbar").classList.toggle("open"));

  function renderAll() {
    renderTopbar();
    renderLogs();
    if (state.route === "overview") { renderKPIs(); renderWorkers(); renderActiveJob(); }
    if (state.route === "jobs") renderJobsTable();
    if (state.route === "job") loadDetail(state.jobId);
    if (state.route === "predict") renderPredict();
    if (state.route === "new") renderExplainer();
  }

  // ---------------------------------------------------------- overview
  function renderTopbar() {
    const c = state.cluster, j = c.jobs || {};
    $("#topbar-stats").innerHTML =
      `<span class="chip">Workers <b>${c.online}/${c.total}</b></span>` +
      `<span class="chip">Cores <b>${c.cores}</b></span>` +
      `<span class="chip">Running <b>${j.running || 0}</b></span>` +
      `<span class="chip">Queued <b>${j.queued || 0}</b></span>`;
  }

  function renderKPIs() {
    const c = state.cluster, j = c.jobs || {};
    const online = c.workers.filter((w) => w.status === "online");
    const avgCpu = online.length ? online.reduce((a, w) => a + w.cpu, 0) / online.length : 0;
    const tiles = [
      ["Workers", `${c.online}/${c.total}`, c.total - c.online ? `${c.total - c.online} offline` : "all online"],
      ["CPU cores", c.cores, "used by multiprocessing"],
      ["Avg CPU", `${avgCpu.toFixed(0)}%`, "across online PCs"],
      ["Running", j.running || 0, `${c.running_tasks} task(s) in flight`],
      ["Queued", j.queued || 0, "jobs waiting"],
      ["Tasks", num(c.tasks_dispatched), `${j.completed || 0} job(s) completed`],
    ];
    $("#kpis").innerHTML = tiles.map(([l, v, s]) => `<div class="cell"><div class="value">${esc(v)}</div><div class="label">${l}</div><div class="sub">${esc(s)}</div></div>`).join("");
  }

  const PC_ICON = '<svg class="pc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/></svg>';

  function renderWorkers() {
    const ws_ = state.cluster.workers;
    if (!ws_.length) {
      $("#workers").innerHTML = `<div class="empty">No workers yet. On each lab PC run:<br><br><code>python -m minicluster.worker --coordinator ${esc(location.hostname)} --name PC-01</code></div>`;
      return;
    }
    const rows = ws_.map((w) => {
      const busy = w.status === "online" && w.running > 0;
      return `<tr class="${w.status === "online" ? "" : "row-off"}">
        <td><b>${esc(w.id)}</b><div class="muted small" title="${esc(w.platform)}">${esc(w.address)}</div></td>
        <td>${pill(w.status === "online" ? (busy ? "busy" : "online") : "offline")}</td>
        <td><span class="mini-bar">${bar(w.cpu)}</span> ${w.cpu.toFixed(0)}%</td>
        <td><span class="mini-bar">${bar(w.ram, "ram")}</span> ${w.ram.toFixed(0)}% <span class="muted small">of ${bytes(w.ram_total)}</span></td>
        <td>${w.cores}/${w.cpu_count}${w.gpu ? " · " + esc(w.gpu) : ""}</td>
        <td>${num(w.tasks_done)}</td>
        <td>${secs(w.busy_seconds)}</td>
        <td>${w.current_job ? esc(w.current_job) : '<span class="muted">-</span>'}</td>
      </tr>`;
    }).join("");
    $("#workers").innerHTML = `<table class="table"><thead><tr><th>PC</th><th>Status</th><th>CPU</th><th>RAM</th><th>Cores</th><th>Tasks done</th><th>Busy time</th><th>Job</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderActiveJob() {
    const jobs = [...state.jobs.values()];
    const job = jobs.find((j) => j.status === "running") || jobs.sort((a, b) => b.created_at - a.created_at)[0];
    if (!job) {
      $("#active-job").innerHTML = `<div class="empty">No jobs yet. <a href="#/new">Submit your first job &rarr;</a></div>`;
      return;
    }
    const workers = Object.entries(job.workers || {});
    $("#active-job").innerHTML = `
      <div class="detail-head" style="margin:0"><div><div class="job-title"><a href="#/jobs/${job.id}">${esc(job.name)}</a></div>
      <div class="muted small">${esc(TYPE_LABEL[job.type] || job.type)} · ${esc(job.phase)}</div></div>
      <div class="actions">${pill(job.status)}</div></div>
      <div class="bar progress-big ${job.status === "completed" ? "ok" : ""}"><span style="width:${pct(job.progress * 100)}"></span></div>
      <div class="meter-top"><span>${pct(job.progress * 100)} complete</span><b>${secs(job.duration)}</b></div>
      <div class="wp-row">${workers.map(([id, s]) => `<div class="wp-box"><div class="meter-top"><b>${esc(id)}</b><span>${s.state === "offline" ? '<span style="color:var(--err)">lost</span>' : s.tasks + " tasks"}</span></div>${bar((s.progress || 0) * 100)}</div>`).join("")}</div>
      ${job.headline ? `<p><b>${esc(job.headline)}</b></p>` : ""}${job.error ? `<p style="color:var(--err)">${esc(job.error)}</p>` : ""}`;
  }

  function renderLogs() {
    const item = (e) => `<li><span class="t">${time(e.ts)}</span><span class="${esc(e.level)}">${esc(e.message)}</span></li>`;
    const recent = state.logs.slice(-40).reverse();
    $("#log-mini").innerHTML = recent.map(item).join("") || '<li class="muted">No events yet</li>';
    if (state.route === "logs") $("#log-full").innerHTML = state.logs.slice().reverse().map(item).join("");
  }

  // -------------------------------------------------------------- jobs
  function renderJobsTable() {
    const jobs = [...state.jobs.values()].sort((a, b) => b.created_at - a.created_at);
    $("#jobs-empty").hidden = jobs.length > 0;
    $("#jobs-table tbody").innerHTML = jobs.map((j) => `
      <tr data-id="${j.id}">
        <td class="wrap"><b>${esc(j.name)}</b><div class="muted small">${j.id}</div></td>
        <td>${esc(TYPE_LABEL[j.type] || j.type)}</td>
        <td>${pill(j.status)}</td>
        <td><span class="mini-bar">${bar(j.progress * 100, j.status === "completed" ? "ok" : "")}</span> ${pct(j.progress * 100)}</td>
        <td class="wrap">${esc(j.headline || j.error || j.phase || "")}</td>
        <td>${secs(j.duration)}</td>
        <td>${dateTime(j.created_at)}</td>
      </tr>`).join("");
  }
  $("#jobs-table").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr) location.hash = `#/jobs/${tr.dataset.id}`;
  });

  let detailTimer = null;
  function scheduleDetail() {
    if (detailTimer) return;
    detailTimer = setTimeout(() => { detailTimer = null; renderDetail(state.jobs.get(state.jobId)); }, 300);
  }

  async function loadDetail(id) {
    const cached = state.jobs.get(id);
    if (cached && cached.spec) renderDetail(cached);
    else $("#job-detail").innerHTML = '<div class="card empty">Loading...</div>';
    try {
      const job = await api(`/api/jobs/${id}`);
      state.jobs.set(id, job);
      if (state.jobId === id) renderDetail(job);
    } catch (e) {
      $("#job-detail").innerHTML = `<div class="card empty">${esc(e.message)}</div>`;
    }
  }

  function statTile(label, value, sub) {
    return `<div class="cell"><div class="value">${value}</div><div class="label">${esc(label)}</div><div class="sub">${esc(sub || "")}</div></div>`;
  }

  function renderDetail(job) {
    if (!job || !job.spec) return;
    const r = job.result || {};
    const active = !["completed", "failed", "cancelled"].includes(job.status);
    const workers = Object.entries(job.workers || {});
    let html = `
      <div class="detail-head">
        <a href="#/jobs" class="btn small">&larr; Jobs</a>
        <h2>${esc(job.name)}</h2>${pill(job.status)}
        <div class="actions">
          ${active ? `<button class="btn small danger" data-act="cancel">Cancel</button>` : `<button class="btn small" data-act="rerun">Run again</button><button class="btn small danger" data-act="delete">Delete</button>`}
        </div>
      </div>
      <div class="card strip" style="margin-bottom:16px">
        ${statTile("Progress", pct(job.progress * 100), job.phase)}
        ${statTile("Duration", secs(job.duration), `started ${time(job.started_at)}`)}
        ${statTile("Workers", workers.length || "-", workers.map((w) => w[0]).join(", "))}
        ${statTile(job.type === "training" && r.metrics ? (r.metrics.test.accuracy != null ? "Test accuracy" : "Test R²") : "Result",
                   job.type === "training" && r.metrics ? (r.metrics.test.accuracy != null ? (r.metrics.test.accuracy * 100).toFixed(1) + "%" : r.metrics.test.r2.toFixed(3)) : esc(job.status),
                   r.headline || job.error || "")}
      </div>`;
    if (job.error) html += `<div class="card" style="border-color:var(--err);margin-bottom:16px"><b style="color:var(--err)">Error:</b> ${esc(job.error)}</div>`;

    if (job.type === "training") html += trainingSection(job);
    if (job.type === "compute") html += computeSection(job);
    if (job.type === "benchmark") html += benchmarkSection(job);

    if (workers.length && job.type !== "benchmark") {
      html += `<div class="card table-card" style="margin-bottom:16px"><table class="table"><thead><tr>
        <th>Worker</th><th>State</th><th>Tasks</th><th>${job.type === "training" ? "Rows held" : "Numbers"}</th><th>Compute time</th><th>${job.type === "training" ? "Epochs" : "Share of chunks"}</th><th>${job.type === "training" ? "Shard" : "Processes"}</th></tr></thead><tbody>
        ${workers.map(([id, s]) => `<tr><td><b>${esc(id)}</b></td><td>${pill(s.state)}</td><td>${num(s.tasks)}</td><td>${num(s.items)}</td><td>${secs(s.compute_time)}</td>
        <td><span class="mini-bar">${bar((s.progress || 0) * 100)}</span> ${job.type === "training" ? num(s.epochs || 0) : pct((s.progress || 0) * 100)}</td><td class="wrap">${esc(job.type === "training" ? s.shard || "" : (s.processes || 1) + " process(es)")}</td></tr>`).join("")}
        </tbody></table></div>`;
    }
    const resultCopy = Object.assign({}, r);
    if (resultCopy.model) resultCopy.model = Object.assign({}, resultCopy.model, { weights: "[...]" });
    const specRows = Object.entries(job.spec).map(([k, v]) => `<tr><th>${esc(k.replace(/_/g, " "))}</th><td>${esc(typeof v === "object" ? JSON.stringify(v) : String(v))}</td></tr>`).join("");
    html += `<div class="card" style="margin-bottom:16px"><h3>Parameters used</h3><table class="kv">${specRows}</table>
      <details class="raw"><summary>Show raw result (JSON)</summary><pre class="json">${esc(JSON.stringify(resultCopy, null, 2))}</pre></details></div>`;
    $("#job-detail").innerHTML = html;

    // charts after the DOM exists
    if (job.type === "training" && job.history.length) {
      const h = job.history;
      Charts.line($("#chart-loss"), [
        { name: "train loss", points: h.map((p) => [p.epoch, p.train_loss]) },
        { name: "test loss", points: h.map((p) => [p.epoch, p.loss]) },
      ], { xLabel: "epoch" });
      const isReg = h[0].accuracy === undefined;
      Charts.line($("#chart-metric"), [{ name: isReg ? "test R²" : "test accuracy %", points: h.map((p) => [p.epoch, isReg ? p.r2 : p.accuracy * 100]) }],
        { xLabel: "epoch", yMax: isReg ? 1 : 100, legend: true });
    }
    if (job.type === "compute" && workers.length) {
      Charts.bars($("#chart-chunks"), workers.map(([id, s]) => ({ label: id, value: s.tasks })), { yLabel: "chunks" });
      Charts.bars($("#chart-ctime"), workers.map(([id, s]) => ({ label: id, value: s.compute_time })), { yLabel: "seconds", format: (v) => v.toFixed(2) + "s" });
    }
    if (job.type === "benchmark" && r.runs) {
      Charts.bars($("#chart-time"), r.runs.map((x) => ({ label: `${x.workers} PC`, value: x.seconds })), { yLabel: "seconds", format: (v) => v.toFixed(2) + "s" });
      Charts.bars($("#chart-speedup"), r.runs.map((x) => ({ label: `${x.workers} PC`, value: x.speedup, color: "#0d9488" })),
        { yLabel: "speed-up", format: (v) => "x" + v.toFixed(2), overlay: { name: "ideal (linear)", points: r.runs.map((x) => x.workers) }, series: "measured speed-up", yMax: r.runs.length });
    }
  }

  function trainingSection(job) {
    const r = job.result;
    let html = `<div class="charts"><div class="card chart-card"><h3>Training &amp; test loss</h3><div class="chart" id="chart-loss"></div></div>
      <div class="card chart-card"><h3>Model quality on test data</h3><div class="chart" id="chart-metric"></div></div></div>`;
    if (r && r.timing) {
      const t = r.timing, other = Math.max(0, t.total - t.scatter - t.compute - t.communication);
      const segs = [["Scatter data", t.scatter, "#7c3aed"], ["Worker compute", t.compute, "#0d9488"], ["Communication & sync", t.communication, "#ea580c"], ["Coordinator (load, evaluate)", other, "#94a3b8"]];
      html += `<div class="card" style="margin-bottom:16px"><h3>Time breakdown</h3>
        <table class="kv breakdown">${segs.map(([n, v, c]) => `<tr><th><i style="background:${c}"></i>${n}</th><td>${secs(v)}</td>
          <td class="bar-cell"><div class="bar"><span style="width:${(v / t.total) * 100}%;background:${c}"></span></div></td><td class="muted">${((v / t.total) * 100).toFixed(0)}%</td></tr>`).join("")}</table>
        <p class="muted small">Dataset: ${esc(r.dataset.name)} · ${num(r.dataset.train_rows)} train / ${num(r.dataset.test_rows)} test rows · target <code>${esc(r.dataset.target)}</code> · ${esc(r.model.label)} · ${esc(r.model.strategy.replace("_", " "))}
        ${job.status === "completed" ? ` · <a href="#/predict" data-predict="${job.id}">Use this model &rarr;</a>` : ""}</p></div>`;
    }
    return html;
  }

  function computeSection(job) {
    const r = job.result;
    let html = `<div class="charts"><div class="card chart-card"><h3>Chunks done by each PC</h3><div class="chart" id="chart-chunks"></div></div>
      <div class="card chart-card"><h3>Time spent by each PC</h3><div class="chart" id="chart-ctime"></div></div></div>`;
    if (r && r.output) {
      html += `<div class="card" style="margin-bottom:16px"><h3>${esc(r.kernel_label)} - final (reduced) output</h3><div class="strip" style="margin:10px 0 0">
        ${Object.entries(r.output).map(([k, v]) => statTile(k.replace(/_/g, " "), esc(Array.isArray(v) ? v.map((p) => p.join("²=")).join(", ") : typeof v === "number" ? (Number.isInteger(v) ? num(v) : v.toFixed(6)) : String(v).length > 18 ? String(v).slice(0, 18) + "…" : v), "")).join("")}
      </div></div>`;
    }
    return html;
  }

  function benchmarkSection(job) {
    const r = job.result;
    let html = `<div class="charts"><div class="card chart-card"><h3>Run time with p PCs</h3><div class="chart" id="chart-time"></div></div>
      <div class="card chart-card"><h3>Speed-up = T(1) / T(p)</h3><div class="chart" id="chart-speedup"></div></div></div>`;
    if (r && r.runs) {
      html += `<div class="card table-card" style="margin-bottom:16px"><table class="table"><thead><tr><th>PCs (p)</th><th>Time T(p)</th><th>Speed-up</th><th>Efficiency</th><th>Run result</th></tr></thead><tbody>
        ${r.runs.map((x) => `<tr><td><b>${x.workers}</b></td><td>${secs(x.seconds)}</td><td>x${x.speedup.toFixed(2)}</td><td><span class="mini-bar">${bar(x.efficiency * 100, "ok")}</span> ${(x.efficiency * 100).toFixed(0)}%</td><td class="wrap">${esc(x.headline || "")}</td></tr>`).join("")}
        </tbody></table></div>
        <div class="card" style="margin-bottom:16px"><h3>Observation</h3><p class="muted">Why isn't the speed-up exactly p? Every extra PC adds network
        communication, synchronisation (the slowest worker sets the pace each round) and coordinator overhead. Compute-bound jobs (prime counting)
        scale well; communication-bound jobs (small datasets, many epochs) scale poorly - this is Amdahl's law in practice.</p></div>`;
    }
    return html;
  }

  $("#job-detail").addEventListener("click", async (e) => {
    const p = e.target.closest("[data-predict]");
    if (p) { state.predictJob = p.dataset.predict; return; }
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const id = state.jobId, job = state.jobs.get(id);
    try {
      if (btn.dataset.act === "cancel") { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); toast("Cancellation requested"); }
      if (btn.dataset.act === "delete") { await api(`/api/jobs/${id}`, { method: "DELETE" }); location.hash = "#/jobs"; }
      if (btn.dataset.act === "rerun") {
        const spec = Object.assign({ type: job.type }, job.spec);
        const j = await api("/api/jobs", { method: "POST", json: { name: job.name, spec } });
        location.hash = `#/jobs/${j.id}`;
      }
    } catch (err) { toast(err.message, true); }
  });

  // ---------------------------------------------------------- new job
  const EXPLAIN = {
    compute: `<h3>How Level 1 works</h3><pre>
          Web UI ── HTTP ──▶ Coordinator
                               │ split 1..N into chunks
          ┌────────────────────┼────────────────────┐
          ▼  TCP socket        ▼                    ▼
        PC-01               PC-02                PC-03
   multiprocessing      multiprocessing      multiprocessing
   (one per core)       (one per core)       (one per core)
          └────────────────────┼────────────────────┘
                               ▼
                       Reduce → Result</pre>
      <ul><li><b>Socket programming</b>: framed JSON + binary messages over TCP.</li>
      <li><b>Multithreading</b>: each worker has receiver, heartbeat &amp; executor threads.</li>
      <li><b>Multiprocessing</b>: every chunk is split again across the PC's CPU cores.</li>
      <li><b>Fault tolerance</b>: unplug a PC mid-job - its chunk is reassigned.</li></ul>`,
    training: `<h3>How Level 2 works (gradient sync)</h3><pre>
                 Coordinator (parameter server)
                       │ broadcast weights W
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
      Worker 1      Worker 2      Worker 3
     shard 1/3     shard 2/3     shard 3/3
         │ ∑ gradients │             │
         └─────────────┼─────────────┘
                       ▼
          W ← W − lr · (∑g / N)   → next epoch</pre>
      <p class="muted">Each PC keeps its data shard in memory (scattered once). Every epoch only the small weight matrix
      and gradients cross the network. Gradient sync gives the <b>same model</b> a single PC would train; model averaging trades a
      little accuracy for far less communication.</p>
      <p class="muted small">Scenario: <i>Smart Admission Predictor</i> - predict whether a student will pass from matric/inter marks,
      entry test, attendance, previous GPA, study hours and assignments.</p>`,
    benchmark: `<h3>How Level 3 works</h3>
      <p class="muted">The same workload is executed on 1, 2, … p PCs and timed.</p>
      <pre>
  Speedup(p)    = T(1) / T(p)
  Efficiency(p) = Speedup(p) / p

  1 PC  → 48 s
  2 PCs → 28 s   speed-up 1.7  efficiency 86%
  3 PCs → 19 s   speed-up 2.5  efficiency 84%</pre>
      <p class="muted">Ask students: <i>"With 3 PCs, why isn't it exactly 16 s?"</i> → communication overhead, synchronisation,
      load imbalance and the serial part of the program (Amdahl's law).</p>`,
  };
  function renderExplainer() {
    const tab = $("#job-tabs .active").dataset.tab;
    $("#explainer").innerHTML = EXPLAIN[tab];
  }
  $("#job-tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (!b) return;
    $$("#job-tabs button").forEach((x) => x.classList.toggle("active", x === b));
    // getAttribute, not .dataset: a form control named "dataset" shadows form.dataset
    $$("[data-tab-panel]").forEach((p) => (p.hidden = p.getAttribute("data-tab-panel") !== b.dataset.tab));
    renderExplainer();
  });

  function formData(form) {
    const out = {};
    $$("input[name], select[name]", form).forEach((i) => {
      if (i.type === "number") { if (i.value !== "") out[i.name] = Number(i.value); }
      else out[i.name] = i.value;
    });
    return out;
  }

  async function submit(name, spec) {
    try {
      const job = await api("/api/jobs", { method: "POST", json: { name: name || "", spec } });
      toast(`Job ${job.id} submitted`);
      location.hash = `#/jobs/${job.id}`;
    } catch (e) { toast(e.message, true); }
  }

  $("#form-compute").addEventListener("submit", (e) => {
    e.preventDefault();
    const d = formData(e.target), name = d.name; delete d.name;
    submit(name, Object.assign({ type: "compute" }, d));
  });

  $("#form-training").addEventListener("submit", (e) => {
    e.preventDefault();
    const d = formData(e.target), name = d.name; delete d.name;
    if (d.dataset !== "student_performance") delete d.rows;
    submit(name, Object.assign({ type: "training" }, d));
  });

  const BENCH = {
    primes: { type: "compute", kernel: "prime_count", n: 2000000, chunks_per_worker: 4 },
    pi: { type: "compute", kernel: "monte_carlo_pi", n: 20000000, chunks_per_worker: 4 },
    train: { type: "training", rows: 1000000, epochs: 40, model: "logistic_regression", target: "passed" },
    "train-small": { type: "training", rows: 20000, epochs: 40, model: "logistic_regression", target: "passed" },
  };
  $("#form-benchmark").addEventListener("submit", (e) => {
    e.preventDefault();
    const d = formData(e.target);
    const spec = { type: "benchmark", base: BENCH[d.base] };
    if (d.max_workers) spec.max_workers = d.max_workers;
    submit("", spec);
  });

  async function loadDatasets(selectId) {
    try { state.datasets = await api("/api/datasets"); } catch (e) { return; }
    const sel = $("#dataset-select");
    sel.innerHTML = state.datasets.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
    if (selectId) sel.value = selectId;
    onDatasetChange();
  }
  function onDatasetChange() {
    const d = state.datasets.find((x) => x.id === $("#dataset-select").value);
    if (!d) return;
    $("#rows-field").hidden = !d.builtin;
    const targets = d.builtin ? Object.keys(d.targets) : d.columns.slice().reverse();
    $("#target-select").innerHTML = targets.map((t) => `<option>${esc(t)}</option>`).join("");
    if (d.builtin) $("#target-select").value = "passed";
    onTargetChange();
  }
  function onTargetChange() {
    const d = state.datasets.find((x) => x.id === $("#dataset-select").value);
    if (!d || !d.builtin) return;
    const task = d.targets[$("#target-select").value];
    $("#model-select").value = task === "regression" ? "linear_regression" : task === "multiclass" ? "softmax_regression" : "logistic_regression";
    $("#form-training [name=lr]").value = task === "regression" ? 0.1 : 0.5;
  }
  $("#dataset-select").addEventListener("change", onDatasetChange);
  $("#target-select").addEventListener("change", onTargetChange);

  $("#csv-upload").addEventListener("click", async () => {
    const f = $("#csv-file").files[0];
    if (!f) return toast("Choose a CSV file first", true);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await api("/api/datasets", { method: "POST", body: fd });
      toast(`Uploaded ${res.name} (${res.columns.length} columns)`);
      loadDatasets(res.id);
    } catch (e) { toast(e.message, true); }
  });

  // ----------------------------------------------------------- predict
  const SAMPLE = { matric_marks: 82, inter_marks: 76, entry_test: 64, attendance: 88, prev_gpa: 3.1, study_hours: 18, assignments_done: 90 };
  async function renderPredict() {
    const done = [...state.jobs.values()].filter((j) => j.type === "training" && j.status === "completed").sort((a, b) => b.created_at - a.created_at);
    const sel = $("#predict-job");
    const keep = state.predictJob || sel.value;
    sel.innerHTML = done.map((j) => `<option value="${j.id}">${esc(j.name)} - ${esc(j.headline || "")}</option>`).join("") || "<option value=''>No completed training jobs yet</option>";
    if (keep && done.some((j) => j.id === keep)) sel.value = keep;
    state.predictJob = null;
    await renderPredictFields();
  }
  async function renderPredictFields() {
    const id = $("#predict-job").value;
    if (!id) { $("#predict-fields").innerHTML = ""; return; }
    let job = state.jobs.get(id);
    if (!job.result) { job = await api(`/api/jobs/${id}`); state.jobs.set(id, job); }
    const feats = job.result.model.features;
    $("#predict-fields").innerHTML = feats.map((f) => `<label>${esc(f.replace(/_/g, " "))}<input type="number" step="any" name="${esc(f)}" value="${SAMPLE[f] != null ? SAMPLE[f] : 0}"></label>`).join("");
  }
  $("#predict-job").addEventListener("change", renderPredictFields);
  $("#form-predict").addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = $("#predict-job").value;
    if (!id) return;
    const row = {};
    $$("#predict-fields input").forEach((i) => (row[i.name] = Number(i.value)));
    try {
      const res = (await api(`/api/jobs/${id}/predict`, { method: "POST", json: { rows: [row] } })).predictions[0];
      const p = res.prediction;
      let html = `<h3>Prediction</h3><div class="prediction">${esc(typeof p === "number" ? p.toFixed(2) : p)}</div>`;
      if (res.probabilities) {
        html += Object.entries(res.probabilities).map(([k, v]) => `<div class="prob"><span>${esc(k)}</span>${bar(v * 100, k === String(p) ? "ok" : "")}<b>${(v * 100).toFixed(1)}%</b></div>`).join("");
      }
      html += `<p class="muted small">Model from job <a href="#/jobs/${id}">${id}</a>. Features are standardised with the training mean/std before scoring.</p>`;
      $("#predict-result").innerHTML = html;
    } catch (err) { toast(err.message, true); }
  });

  // ------------------------------------------------------------- boot
  async function loadInitial() {
    try {
      state.info = await api("/api/info");
      $("#worker-port").textContent = state.info.worker_port;
      $("#version").textContent = "v" + state.info.version;
      if (state.info.auth_required && !state.token) askToken();
      const jobs = await api("/api/jobs");
      jobs.forEach((j) => state.jobs.set(j.id, Object.assign(state.jobs.get(j.id) || {}, j)));
      await loadDatasets();
      renderAll();
    } catch (e) { /* surfaced via toast/dialog */ }
  }

  route();
  loadInitial();
  connect();
})();
