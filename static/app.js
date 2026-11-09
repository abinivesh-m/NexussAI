/* ==========================================================================
   AstroHealth client
   Vanilla JS, no build step. State is explicit; every view is a pure render
   of the last successful API response.
   ========================================================================== */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    crew: [],
    mission: null,
    alerts: null,
    reference: null,
    model: null,
    selectedId: null,
    detail: null,
    checkinCrew: null,
    checkin: { symptoms: {}, sleep_quality: 4, mood: 4, workload: 2 },
    chartInstances: {},
    lastFetch: null,
  };

  const SCENARIOS = ["baseline", "fatigue", "sleep", "cardiac", "immune", "radiation"];

  /* ------------------------------------------------------------------ utils */
  const fmt = (n, d = 1) => (n === null || n === undefined || Number.isNaN(n) ? "—" : Number(n).toFixed(d));
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function toast(message, isError = false) {
    const el = $("#toast");
    el.textContent = message;
    el.className = "toast show" + (isError ? " error" : "");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (el.className = "toast"), 3600);
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch (_) { /* non-JSON error body */ }
      throw new Error(`${res.status} ${detail}`);
    }
    return res.json();
  }

  const levelClass = (level) => (level === "high" ? "high" : level === "moderate" ? "moderate" : "low");
  const levelColour = (level) =>
    level === "high" ? getVar("--danger") : level === "moderate" ? getVar("--warn") : getVar("--ok");
  function getVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ---------------------------------------------------------------- charts */
  function mountChart(key, canvasId, config) {
    if (state.chartInstances[key]) state.chartInstances[key].destroy();
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return null;
    const chart = new Chart(canvas, config);
    state.chartInstances[key] = chart;
    return chart;
  }

  function baseOptions(extra = {}) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#111a2e",
          borderColor: "#1d2a44",
          borderWidth: 1,
          titleColor: "#e7eefb",
          bodyColor: "#8ba0c0",
          padding: 10,
          displayColors: false,
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#5f7391", font: { size: 10 }, maxRotation: 0, autoSkipPadding: 18 },
          border: { color: "#16213a" },
        },
        y: {
          grid: { color: "rgba(29, 42, 68, 0.55)" },
          ticks: { color: "#5f7391", font: { size: 10 } },
          border: { display: false },
        },
      },
      ...extra,
    };
  }

  function lineDataset(label, data, colour, axis = "y", fill = false) {
    return {
      label,
      data,
      borderColor: colour,
      backgroundColor: fill ? colour.replace("rgb", "rgba").replace(")", ", 0.12)") : colour,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.32,
      fill,
      yAxisID: axis,
    };
  }

  /* ------------------------------------------------------------------ tabs */
  function setTab(name) {
    $$(".view").forEach((view) => (view.hidden = view.id !== `view-${name}`));
    $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
    moveIndicator();
    if (name === "model" && !state.model) loadModel();
    if (name === "checkin" && !state.reference) loadReference();
    if (name === "crew" && !state.crew.length) loadCrew();
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }

  function moveIndicator() {
    const active = $(".tab.active");
    const indicator = $("#tab-indicator");
    if (!active || !indicator) return;
    indicator.style.width = `${active.offsetWidth - 20}px`;
    indicator.style.transform = `translateX(${active.offsetLeft + 10}px)`;
  }

  function currentTab() {
    const hash = location.hash.replace("#", "");
    return ["overview", "crew", "checkin", "model"].includes(hash) ? hash : "overview";
  }

  /* --------------------------------------------------------------- overview */
  async function loadCrew() {
    try {
      const data = await api("/api/crew");
      state.crew = data.crew;
      state.mission = data.mission;
      state.lastFetch = Date.now();
      renderCrewBoard();
      renderCrewGrid();
      renderKpis();
      setLive(true);
    } catch (err) {
      setLive(false);
      toast(`Could not load crew: ${err.message}`, true);
    }
  }

  async function loadAlerts() {
    try {
      state.alerts = await api("/api/alerts");
      renderAlerts();
      renderAlertPill();
      renderKpis();
    } catch (err) {
      setLive(false);
    }
  }

  async function loadOverview() {
    await Promise.all([loadCrew(), loadAlerts(), loadTrends()]);
  }

  async function loadTrends() {
    try {
      const data = await api("/api/trends?days=21");
      const labels = data.dates.map((d) => d.slice(5));
      mountChart("mission", "mission-chart", {
        type: "line",
        data: {
          labels,
          datasets: [
            lineDataset("Heart rate", data.series.heart_rate, "rgb(255, 107, 107)", "y", true),
            lineDataset("Sleep", data.series.sleep_hours, "rgb(139, 147, 255)", "y1"),
          ],
        },
        options: baseOptions({
          scales: {
            x: baseOptions().scales.x,
            y: {
              ...baseOptions().scales.y,
              position: "left",
              title: { display: true, text: "bpm", color: "#5f7391", font: { size: 10 } },
            },
            y1: {
              ...baseOptions().scales.y,
              position: "right",
              grid: { drawOnChartArea: false },
              title: { display: true, text: "hours", color: "#5f7391", font: { size: 10 } },
            },
          },
        }),
      });
      $("#mission-legend").innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:#ff6b6b"></span>Heart rate (bpm)</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#8b93ff"></span>Sleep (h)</span>`;
    } catch (_) { /* chart is non-critical */ }
  }

  function setLive(ok) {
    const el = $("#live-indicator");
    if (!el) return;
    el.classList.toggle("is-stale", !ok);
    el.lastElementChild.textContent = ok ? "live" : "offline";
  }

  function renderKpis() {
    const crew = state.crew;
    if (!crew.length) return;
    const meanRisk = crew.reduce((sum, c) => sum + c.risk.overall, 0) / crew.length;
    const alerts = state.alerts || { count: 0, critical: 0, warning: 0 };
    $("#kpi-crew").textContent = crew.length;
    $("#kpi-crew-foot").textContent = `${crew.filter((c) => c.checked_in).length} self check-ins recorded`;
    $("#kpi-alerts").textContent = alerts.count;
    $("#kpi-alerts-foot").textContent = `${alerts.critical} critical · ${alerts.warning} warning`;
    $("#kpi-risk").textContent = fmt(meanRisk);
    $("#kpi-top").textContent = crew[0].name;
    $("#kpi-top-foot").textContent = `${crew[0].risk.overall.toFixed(0)}/100 · ${crew[0].risk.level}`;
    $("#mission-day").textContent = state.mission?.day ?? "—";
    if (state.mission) $("#mission-profile").textContent = state.mission.profile;
  }

  function renderAlertPill() {
    const pill = $("#alert-pill");
    if (!pill || !state.alerts) return;
    pill.classList.toggle("has-alerts", state.alerts.critical + state.alerts.warning > 0);
    pill.classList.toggle("is-clean", state.alerts.critical + state.alerts.warning === 0);
    $("#alert-pill-text").textContent = `${state.alerts.count} alert${state.alerts.count === 1 ? "" : "s"}`;
  }

  function renderAlerts() {
    const feed = $("#alert-feed");
    const data = state.alerts;
    if (!feed || !data) return;
    $("#alerts-hint").textContent = `${data.critical} critical · ${data.warning} warning`;
    if (!data.alerts.length) {
      feed.innerHTML = `<div class="empty-state">No active alerts. All measured signals are inside their operational bands.</div>`;
      return;
    }
    feed.innerHTML = data.alerts
      .slice(0, 14)
      .map(
        (a) => `
      <article class="alert-card ${a.severity}">
        <div class="alert-top">
          <span class="tag ${a.severity}">${esc(a.severity)}</span>
          <span class="alert-title">${esc(a.title)}</span>
        </div>
        <div class="alert-who">${esc(a.astronaut_name)} · ${esc(a.role)}</div>
        <div class="alert-detail">${esc(a.detail)}</div>
        <div class="alert-meta">
          <span class="tag">${esc(a.detector)}</span>
          <span class="tag">${esc(a.system)}</span>
          <span class="tag">${fmt(a.value, 1)} ${esc(a.unit)}</span>
        </div>
      </article>`
      )
      .join("");
  }

  function renderCrewBoard() {
    const board = $("#risk-board");
    if (!board) return;
    board.innerHTML = state.crew
      .map(
        (c) => `
      <div class="risk-row" data-id="${esc(c.id)}" role="button" tabindex="0">
        <div>
          <div class="risk-name">${esc(c.name)}</div>
          <div class="risk-role">${esc(c.role)}</div>
        </div>
        <div class="risk-bar"><div class="risk-bar-fill fill-${levelClass(c.risk.level)}" style="width:${Math.min(100, c.risk.overall)}%"></div></div>
        <div class="risk-score">${fmt(c.risk.overall)}</div>
        <div class="risk-level level-${levelClass(c.risk.level)}">${esc(c.risk.level)}</div>
      </div>`
      )
      .join("");
    $$(".risk-row", board).forEach((row) => {
      row.addEventListener("click", () => openCrew(row.dataset.id));
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openCrew(row.dataset.id);
        }
      });
    });
  }

  /* ------------------------------------------------------------------- crew */
  function renderCrewGrid() {
    const grid = $("#crew-grid");
    if (!grid) return;
    grid.innerHTML = state.crew
      .map(
        (c) => `
      <button class="crew-card ${state.selectedId === c.id ? "active" : ""}" data-id="${esc(c.id)}" type="button">
        <div class="crew-card-top">
          <div>
            <div class="crew-name">${esc(c.name)}</div>
            <div class="crew-role">${esc(c.role)} · age ${c.age}</div>
          </div>
          <div class="crew-risk">
            <div class="crew-risk-value level-${levelClass(c.risk.level)}">${fmt(c.risk.overall, 0)}</div>
            <div class="crew-risk-label level-${levelClass(c.risk.level)}">${esc(c.risk.level)}</div>
          </div>
        </div>
        <div class="vital-chips">
          <div class="vital-chip"><span class="vital-chip-label">HR</span><span class="vital-chip-value">${fmt(c.latest.heart_rate, 0)}</span></div>
          <div class="vital-chip"><span class="vital-chip-label">SpO₂</span><span class="vital-chip-value">${fmt(c.latest.spo2, 0)}%</span></div>
          <div class="vital-chip"><span class="vital-chip-label">Sleep</span><span class="vital-chip-value">${fmt(c.latest.sleep_hours)}h</span></div>
        </div>
        <div class="crew-card-foot">
          <span>${
            c.alerts.critical
              ? `${c.alerts.critical} critical${c.alerts.warning ? ` · ${c.alerts.warning} warning` : ""}`
              : `${c.alerts.total} alert${c.alerts.total === 1 ? "" : "s"}${c.alerts.warning ? ` · ${c.alerts.warning} warning` : ""}`
          }</span>
          <span class="tag ${c.risk.readiness === "cleared" ? "ok" : "warning"}">${esc(c.risk.readiness)}</span>
        </div>
      </button>`
      )
      .join("");
    $$(".crew-card", grid).forEach((card) => card.addEventListener("click", () => openCrew(card.dataset.id)));
  }

  async function openCrew(id) {
    state.selectedId = id;
    if (currentTab() !== "crew") {
      location.hash = "#crew";
      setTab("crew");
    }
    renderCrewGrid();
    const detail = $("#crew-detail");
    detail.hidden = false;
    detail.innerHTML = `<div class="panel"><div class="skeleton-list"><div class="skeleton"></div><div class="skeleton"></div></div></div>`;
    try {
      const data = await api(`/api/crew/${id}`);
      state.detail = data;
      renderCrewDetail(data);
      detail.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      detail.innerHTML = `<div class="panel"><div class="empty-state">Could not load this crew member: ${esc(err.message)}</div></div>`;
    }
  }

  function renderCrewDetail(data) {
    const a = data.astronaut;
    const p = data.prediction;
    const r = data.risk;
    const detail = $("#crew-detail");
    const readinessClass = r.readiness.cleared_for_eva ? "ok" : "bad";

    detail.innerHTML = `
      <div class="detail-head">
        <div>
          <div class="detail-title">${esc(a.name)}</div>
          <div class="detail-sub">${esc(a.role)} · ${esc(a.mission_profile_label)} · mission day ${a.mission_day}</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <span class="badge ${readinessClass}">${r.readiness.cleared_for_eva ? "EVA cleared" : "EVA not cleared"}</span>
          <button class="btn btn-ghost btn-sm" id="close-detail" type="button">Close</button>
        </div>
      </div>

      <div class="detail-grid">
        <article class="panel">
          <header class="panel-head"><h2>14-day vitals</h2><span class="panel-hint">Daily measured</span></header>
          <div class="spark-stack">
            <div class="spark">
              <div class="spark-head"><span class="spark-name">Heart rate</span><span class="spark-unit" id="spark-hr-now">—</span></div>
              <canvas id="chart-hr"></canvas>
            </div>
            <div class="spark">
              <div class="spark-head"><span class="spark-name">Oxygen saturation</span><span class="spark-unit" id="spark-spo2-now">—</span></div>
              <canvas id="chart-spo2"></canvas>
            </div>
            <div class="spark">
              <div class="spark-head"><span class="spark-name">Sleep</span><span class="spark-unit" id="spark-sleep-now">—</span></div>
              <canvas id="chart-sleep"></canvas>
            </div>
          </div>
        </article>

        <article class="panel">
          <header class="panel-head"><h2>Risk decomposition</h2><span class="panel-hint">${fmt(r.overall)} / 100</span></header>
          ${meter("Model objective", r.components.objective, 100, "--accent")}
          ${meter("Alert burden", r.components.alert_burden, 100, "--warn")}
          ${meter("Self-reported", r.components.subjective, 100, "--accent-2")}
          <div class="mini-table" style="margin-top:14px">
            <div class="mini-row"><span class="mini-key">Strain index</span><span class="mini-val">${fmt(p.strain_index)} / 100</span></div>
            <div class="mini-row"><span class="mini-key">Model class</span><span class="mini-val level-${levelClass(p.label)}">${esc(p.label)} (${fmt(p.confidence * 100, 0)}%)</span></div>
            <div class="mini-row"><span class="mini-key">Novel pattern</span><span class="mini-val">${p.is_novel ? "yes" : "no"}</span></div>
          </div>
          ${
            r.disagreement
              ? `<div class="notice" style="margin:14px 0 0"><strong>Disagreement.</strong><span>${esc(r.disagreement.note)}</span></div>`
              : ""
          }
        </article>
      </div>

      <div class="detail-grid-3">
        <article class="panel">
          <header class="panel-head"><h2>Why this assessment</h2><span class="panel-hint">Importance-weighted deviation</span></header>
          <div class="explain-list">
            ${data.explanation
              .map(
                (e) => `
              <div class="explain-item">
                <span class="explain-label">${esc(e.label)}</span>
                <span class="explain-delta ${e.direction === "high" ? "up" : "down"}">${e.deviation_sd > 0 ? "+" : ""}${fmt(e.deviation_sd, 2)} SD</span>
                <span class="explain-delta" style="grid-column:1/-1;font-weight:500;color:var(--faint);font-size:11px">
                  ${fmt(e.value)} ${esc(e.unit)} vs personal baseline ${fmt(e.baseline)} ${esc(e.unit)}
                </span>
                <div class="explain-bar"><div class="explain-bar-fill" style="width:${Math.min(100, e.contribution * 100)}%"></div></div>
              </div>`
              )
              .join("")}
          </div>
        </article>

        <article class="panel">
          <header class="panel-head"><h2>Active alerts</h2><span class="panel-hint">${data.alerts.length}</span></header>
          ${
            data.alerts.length
              ? `<div class="alert-feed" style="max-height:320px">${data.alerts
                  .map(
                    (al) => `
                <div class="alert-card ${al.severity}">
                  <div class="alert-top"><span class="tag ${al.severity}">${esc(al.severity)}</span><span class="alert-title">${esc(al.title)}</span></div>
                  <div class="alert-detail">${esc(al.detail)}</div>
                  <div class="alert-meta"><span class="tag">${esc(al.detector)}</span><span class="tag">${fmt(al.value)} ${esc(al.unit)}</span></div>
                </div>`
                  )
                  .join("")}</div>`
              : `<div class="empty-state">No alerts for this crew member.</div>`
          }
        </article>

        <article class="panel">
          <header class="panel-head"><h2>Next actions</h2><span class="panel-hint">Decision aid</span></header>
          <div class="action-list">
            ${r.actions
              .slice(0, 5)
              .map(
                (action, i) => `<div class="action-item"><span class="action-num">${i + 1}</span><span>${esc(action)}</span></div>`
              )
              .join("")}
          </div>
          ${
            !r.readiness.cleared_for_eva && r.readiness.blocking.length
              ? `<div style="margin-top:14px;font-size:12px;color:var(--danger)">
                   Blocking: ${r.readiness.blocking.map(esc).join("; ")}
                 </div>`
              : ""
          }
        </article>
      </div>

      <article class="panel">
        <header class="panel-head"><h2>What-if scenarios</h2><span class="panel-hint">Alters the last 5 days</span></header>
        <div class="scenario-row" id="scenario-row">
          ${SCENARIOS.map(
            (s) => `<button class="btn btn-sm" data-scenario="${s}" type="button">${s === "baseline" ? "Reset to baseline" : s}</button>`
          ).join("")}
        </div>
        <p style="margin-top:12px;font-size:12px;color:var(--faint)">
          Scenarios are a demonstration harness: they perturb measured signals so the detectors and model can be
          shown responding. They are not predictions of what will happen.
        </p>
      </article>
    `;

    $("#close-detail")?.addEventListener("click", () => {
      $("#crew-detail").hidden = true;
      state.selectedId = null;
      state.detail = null;
      renderCrewGrid();
    });

    $$("#scenario-row .btn").forEach((btn) =>
      btn.addEventListener("click", () => simulate(a.id, btn.dataset.scenario))
    );

    loadDetailChart(a.id);
  }

  function meter(name, value, max, colourVar) {
    const pct = Math.min(100, (value / max) * 100);
    return `
      <div class="meter">
        <div class="meter-head"><span class="meter-name">${esc(name)}</span><span class="meter-val">${fmt(value)}</span></div>
        <div class="meter-track"><div class="meter-fill" style="width:${pct}%;background:var(${colourVar})"></div></div>
      </div>`;
  }

  async function loadDetailChart(id) {
    try {
      const data = await api(`/api/crew/${id}/series?days=14`);
      const labels = data.series.map((d) => d.date.slice(5));
      const last = data.series[data.series.length - 1];

      const spark = (key, canvasId, colour, unit, valueLabel) => {
        mountChart(key, canvasId, {
          type: "line",
          data: {
            labels,
            datasets: [
              {
                data: data.series.map((d) => d[valueLabel]),
                borderColor: colour,
                backgroundColor:
                  colour.replace("rgb", "rgba").replace(")", ", 0.12)"),
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 3,
                tension: 0.3,
                fill: true,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: "#111a2e",
                borderColor: "#1d2a44",
                borderWidth: 1,
                titleColor: "#e7eefb",
                bodyColor: "#8ba0c0",
                displayColors: false,
                callbacks: { label: (ctx) => `${ctx.parsed.y} ${unit}` },
              },
            },
            scales: {
              x: {
                grid: { display: false },
                border: { display: false },
                ticks: { display: false },
              },
              y: {
                grid: { color: "rgba(29, 42, 68, 0.45)" },
                border: { display: false },
                ticks: { color: "#5f7391", font: { size: 9 }, maxTicksLimit: 3, padding: 4 },
              },
            },
            layout: { padding: { top: 2, right: 4, bottom: 0, left: 0 } },
          },
        });
      };

      spark("sparkHr", "chart-hr", "rgb(255, 107, 107)", "bpm", "heart_rate");
      spark("sparkSpo2", "chart-spo2", "rgb(61, 220, 151)", "%", "spo2");
      spark("sparkSleep", "chart-sleep", "rgb(139, 147, 255)", "h", "sleep_hours");

      $("#spark-hr-now").textContent = `${last.heart_rate} bpm`;
      $("#spark-spo2-now").textContent = `${last.spo2}%`;
      $("#spark-sleep-now").textContent = `${last.sleep_hours} h`;
    } catch (_) { /* non-critical */ }
  }

  async function simulate(id, scenario) {
    try {
      const data = await api(`/api/crew/${id}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, days: 5 }),
      });
      state.detail = data.assessment;
      renderCrewDetail(data.assessment);
      await Promise.all([loadCrew(), loadAlerts()]);
      toast(scenario === "baseline" ? "Reset to baseline signals" : `Applied scenario: ${scenario}`);
    } catch (err) {
      toast(`Scenario failed: ${err.message}`, true);
    }
  }

  /* ----------------------------------------------------------------- checkin */
  async function loadReference() {
    try {
      state.reference = await api("/api/reference");
      buildCheckinForm();
    } catch (err) {
      toast(`Could not load reference data: ${err.message}`, true);
    }
  }

  function buildCheckinForm() {
    if (!state.reference) return;
    if (!state.checkinCrew && state.crew.length) state.checkinCrew = state.crew[0].id;

    $("#checkin-crew").innerHTML = state.crew
      .map(
        (c) =>
          `<button class="chip ${state.checkinCrew === c.id ? "active" : ""}" data-id="${esc(c.id)}" type="button">${esc(c.name)}</button>`
      )
      .join("");
    $$("#checkin-crew .chip").forEach((chip) =>
      chip.addEventListener("click", () => {
        state.checkinCrew = chip.dataset.id;
        $$("#checkin-crew .chip").forEach((c) => c.classList.toggle("active", c.dataset.id === state.checkinCrew));
        loadCheckinPreview(state.checkinCrew);
      })
    );

    $("#symptom-grid").innerHTML = state.reference.symptoms
      .map(
        (s) => `
        <div class="symptom">
          <span class="symptom-name">${esc(s.label)}</span>
          <div class="symptom-scale" data-key="${esc(s.key)}">
            ${[0, 1, 2, 3]
              .map(
                (n) =>
                  `<button class="sev ${(state.checkin.symptoms[s.key] || 0) === n ? "active" : ""}" data-sev="${n}" type="button" aria-label="${esc(s.label)} severity ${n}">${n}</button>`
              )
              .join("")}
          </div>
        </div>`
      )
      .join("");
    $$(".symptom-scale").forEach((scale) => {
      scale.addEventListener("click", (e) => {
        const btn = e.target.closest(".sev");
        if (!btn) return;
        const key = scale.dataset.key;
        const sev = Number(btn.dataset.sev);
        if (sev === 0) delete state.checkin.symptoms[key];
        else state.checkin.symptoms[key] = sev;
        $$(".sev", scale).forEach((b) => b.classList.toggle("active", Number(b.dataset.sev) === sev));
      });
    });

    $$(".seg").forEach((seg) => {
      const field = seg.dataset.field;
      const labels = { sleep_quality: ["1", "2", "3", "4", "5"], mood: ["1", "2", "3", "4", "5"], workload: ["1", "2", "3", "4", "5"] };
      seg.innerHTML = labels[field]
        .map(
          (label, i) =>
            `<button type="button" data-value="${i + 1}" class="${state.checkin[field] === i + 1 ? "active" : ""}">${label}</button>`
        )
        .join("");
      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;
        state.checkin[field] = Number(btn.dataset.value);
        $$("button", seg).forEach((b) => b.classList.toggle("active", b === btn));
      });
    });

    if (state.checkinCrew) loadCheckinPreview(state.checkinCrew);
  }

  async function loadCheckinPreview(id) {
    const panel = $("#checkin-result");
    if (!panel || !id) return;
    panel.innerHTML = `<div class="skeleton-list"><div class="skeleton"></div><div class="skeleton"></div></div>`;
    try {
      const data = await api(`/api/crew/${id}`);
      renderCheckinResult(data, true);
    } catch (err) {
      panel.innerHTML = `<div class="empty-state">Could not load current state.</div>`;
    }
  }

  async function submitCheckin(event) {
    event.preventDefault();
    if (!state.checkinCrew) {
      toast("Select a crew member first", true);
      return;
    }
    const btn = $("#checkin-submit");
    btn.disabled = true;
    $("#checkin-status").textContent = "Submitting…";
    try {
      const payload = {
        ...state.checkin,
        hydration_l: Number($("#hydration").value),
        notes: $("#notes").value.trim() || null,
      };
      const data = await api(`/api/crew/${state.checkinCrew}/checkin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderCheckinResult(data.assessment);
      $("#checkin-status").textContent = "Assessment updated.";
      await Promise.all([loadCrew(), loadAlerts()]);
    } catch (err) {
      $("#checkin-status").textContent = "";
      toast(`Check-in failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
    }
  }

  function renderCheckinResult(data, preview = false) {
    const r = data.risk;
    const level = levelClass(r.level);
    $("#checkin-result").innerHTML = `
      <header class="panel-head"><h2>${preview ? "Current state" : "Assessment"}</h2><span class="panel-hint">${esc(data.astronaut.name)}</span></header>
      ${preview ? `<p style="font-size:12px;color:var(--faint);margin:-6px 0 14px">Sensors only — submit a check-in to fuse in how you feel.</p>` : ""}
      <div class="triage-head">
        <span class="triage-score level-${level}">${fmt(r.overall, 0)}</span>
        <div>
          <div class="crew-risk-label level-${level}">${esc(r.level)} composite risk</div>
          <div class="detail-sub">${r.readiness.cleared_for_eva ? "EVA cleared" : "EVA not cleared"}</div>
        </div>
      </div>
      ${meter("Model objective", r.components.objective, 100, "--accent")}
      ${meter("Alert burden", r.components.alert_burden, 100, "--warn")}
      ${meter("Self-reported", r.components.subjective, 100, "--accent-2")}
      ${
        r.disagreement
          ? `<div class="notice" style="margin:14px 0"><strong>Disagreement.</strong><span>${esc(r.disagreement.note)}</span></div>`
          : ""
      }
      <div class="mini-table" style="margin-bottom:14px">
        ${r.drivers
          .map((d) => `<div class="mini-row"><span class="mini-key">${esc(d.system)}</span><span class="mini-val">${fmt(d.weight)}</span></div>`)
          .join("")}
      </div>
      <h2 style="margin-bottom:10px">Actions</h2>
      <div class="action-list">
        ${r.actions
          .slice(0, 5)
          .map((action, i) => `<div class="action-item"><span class="action-num">${i + 1}</span><span>${esc(action)}</span></div>`)
          .join("")}
      </div>
      ${
        r.readiness.blocking.length
          ? `<div style="margin-top:14px;font-size:12px;color:var(--danger)">Blocking: ${r.readiness.blocking.map(esc).join("; ")}</div>`
          : ""
      }
    `;
  }

  /* ------------------------------------------------------------------- model */
  async function loadModel() {
    try {
      const [model] = await Promise.all([api("/api/model"), state.reference ? null : loadReference()]);
      state.model = model;
      renderModel(model);
    } catch (err) {
      toast(`Could not load model card: ${err.message}`, true);
    }
  }

  function renderModel(model) {
    const m = model.metrics;
    $("#data-statement").textContent = model.data_statement;

    const logo = m.leave_one_archetype_out;
    $("#metric-tiles").innerHTML = `
      ${kpiTile("Cross-validated accuracy", `${(m.stratified_5fold.accuracy * 100).toFixed(1)}%`, "5-fold stratified, out-of-fold")}
      ${kpiTile("Macro F1", m.stratified_5fold.macro_f1.toFixed(3), `across ${m.classes.length} strain classes`)}
      ${kpiTile("Unseen-archetype accuracy", `${(logo.accuracy * 100).toFixed(1)}%`, "leave-one-crew-type-out")}
      ${kpiTile("Training windows", m.n_samples.toLocaleString(), `${m.n_features} features · seed ${m.seed}`)}
    `;

    const importances = m.feature_importance.slice(0, 10);
    mountChart("importance", "importance-chart", {
      type: "bar",
      data: {
        labels: importances.map((f) => f.label),
        datasets: [
          {
            data: importances.map((f) => f.importance),
            backgroundColor: importances.map((_, i) => (i < 3 ? "rgba(76,194,255,0.85)" : "rgba(76,194,255,0.35)")),
            borderRadius: 4,
            barThickness: 14,
          },
        ],
      },
      options: baseOptions({
        indexAxis: "y",
        scales: {
          x: { ...baseOptions().scales.x, grid: { color: "rgba(29,42,68,0.55)" }, ticks: { color: "#5f7391", font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { color: "#8ba0c0", font: { size: 11 } }, border: { display: false } },
        },
      }),
    });

    $("#limitations").innerHTML = m.limitations.map((l) => `<li>${esc(l)}</li>`).join("");

    const cm = m.stratified_5fold.confusion_matrix;
    const labels = cm.labels;
    $("#confusion").innerHTML =
      `<div class="confusion-cell confusion-head"></div>` +
      labels.map((l) => `<div class="confusion-cell confusion-head">${esc(l)}</div>`).join("") +
      cm.matrix
        .map(
          (row, i) =>
            `<div class="confusion-cell confusion-head">${esc(labels[i])}</div>` +
            row
              .map((v, j) => `<div class="confusion-cell ${i === j ? "confusion-diag" : "confusion-off"}">${v}</div>`)
              .join("")
        )
        .join("");

    $("#logo-table").innerHTML = Object.entries(logo.per_archetype)
      .sort((a, b) => b[1].accuracy - a[1].accuracy)
      .map(
        ([name, v]) =>
          `<div class="mini-row"><span class="mini-key">${esc(name.replace(/_/g, " "))}</span><span class="mini-val">${(v.accuracy * 100).toFixed(1)}% <span style="color:var(--faint);font-weight:400">n=${v.n}</span></span></div>`
      )
      .join("");

    if (state.reference) {
      $("#bands-table").innerHTML = `
        <thead><tr><th>Signal</th><th>Band</th><th>Unit</th></tr></thead>
        <tbody>
          ${state.reference.vitals
            .map(
              (b) =>
                `<tr><td>${esc(b.label)}</td><td>${fmt(b.low)} – ${fmt(b.high)}</td><td>${esc(b.unit)}</td></tr>`
            )
            .join("")}
        </tbody>`;
      $("#sources").innerHTML = state.reference.citations
        .map(
          (c) => `
        <li class="source">
          <a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.name)}</a>
          <p>${esc(c.used_for)}</p>
        </li>`
        )
        .join("");
    }
  }

  function kpiTile(label, value, foot) {
    return `<div class="kpi"><span class="kpi-label">${esc(label)}</span><span class="kpi-value">${esc(value)}</span><span class="kpi-foot">${esc(foot)}</span></div>`;
  }

  /* -------------------------------------------------------------------- init */
  function init() {
    setTab(currentTab());
    loadOverview();

    window.addEventListener("hashchange", () => setTab(currentTab()));
    window.addEventListener("resize", moveIndicator);

    $("#refresh-btn")?.addEventListener("click", async () => {
      await loadOverview();
      if (state.selectedId) openCrew(state.selectedId);
      toast("Refreshed");
    });

    $("#alert-pill")?.addEventListener("click", () => {
      location.hash = "#overview";
      setTab("overview");
      setTimeout(() => $("#alerts-panel")?.scrollIntoView({ behavior: "smooth", block: "center" }), 60);
    });

    $("#checkin-form")?.addEventListener("submit", submitCheckin);

    const hydration = $("#hydration");
    hydration?.addEventListener("input", () => {
      $("#hydration-out").textContent = `${Number(hydration.value).toFixed(1)} L`;
    });

    // Keep the overview honest: refresh every 30 s, but only when visible.
    setInterval(() => {
      if (document.hidden) return;
      if (currentTab() === "overview") {
        loadCrew();
        loadAlerts();
      }
    }, 30000);

    // Chart.js may load after this script; re-render when it lands.
    if (typeof Chart === "undefined") {
      window.addEventListener("load", () => {
        if (typeof Chart !== "undefined") {
          loadTrends();
          if (state.model) renderModel(state.model);
          if (state.selectedId) loadDetailChart(state.selectedId);
        }
      });
    }

    setTimeout(moveIndicator, 60);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
