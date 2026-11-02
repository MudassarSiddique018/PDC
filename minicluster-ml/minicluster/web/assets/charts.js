/* Minimal dependency-free SVG charts (works offline in a lab without internet). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const PALETTE = ["#0d9488", "#2563eb", "#16a34a", "#ea580c", "#db2777", "#7c3aed"];

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function niceTicks(min, max, count) {
    if (min === max) { max = min + 1; }
    const span = max - min;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = (count * step0) / span;
    const step = step0 * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1);
    const ticks = [];
    const end = Math.ceil(max / step - 1e-9) * step;
    for (let v = Math.floor(min / step) * step; v <= end + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
    if (a >= 10 || a === 0) return String(+v.toFixed(1));
    return String(+v.toPrecision(3));
  }

  function frame(svg, W, H, pad, xTicks, yTicks, xScale, yScale, xLabel, yLabel) {
    yTicks.forEach((t) => {
      const y = yScale(t);
      el("line", { x1: pad.l, x2: W - pad.r, y1: y, y2: y, class: "gridline" }, svg);
      el("text", { x: pad.l - 6, y: y + 4, "text-anchor": "end" }, svg).textContent = fmt(t);
    });
    xTicks.forEach((t) => {
      el("text", { x: xScale(t.v), y: H - pad.b + 16, "text-anchor": "middle" }, svg).textContent = t.label;
    });
    el("line", { x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b, class: "axis" }, svg);
    if (xLabel) el("text", { x: (pad.l + W - pad.r) / 2, y: H - 2, "text-anchor": "middle" }, svg).textContent = xLabel;
    if (yLabel) {
      const t = el("text", { x: 12, y: (pad.t + H - pad.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(pad.t + H - pad.b) / 2})` }, svg);
      t.textContent = yLabel;
    }
  }

  function legend(container, series) {
    const lg = document.createElement("div");
    lg.className = "legend";
    series.forEach((s, i) => {
      const span = document.createElement("span");
      const sw = document.createElement("i");
      sw.style.background = s.color || PALETTE[i % PALETTE.length];
      if (s.dashed) sw.style.background = "repeating-linear-gradient(90deg," + (s.color || "#888") + " 0 4px,transparent 4px 7px)";
      span.appendChild(sw);
      span.appendChild(document.createTextNode(s.name));
      lg.appendChild(span);
    });
    container.appendChild(lg);
  }

  /** series: [{name, points:[[x,y],...], color?, dashed?}] */
  function line(container, series, opts = {}) {
    container.innerHTML = "";
    const W = 560, H = 240, pad = { l: 48, r: 14, t: 12, b: 34 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
    const all = series.flatMap((s) => s.points);
    if (!all.length) { container.innerHTML = '<div class="empty">No data yet</div>'; return; }
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (x0 === x1) x1 = x0 + 1;
    let y0 = opts.yMin !== undefined ? opts.yMin : Math.min(...ys), y1 = opts.yMax !== undefined ? opts.yMax : Math.max(...ys);
    const yt = niceTicks(y0, y1, 5);
    y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
    const xs_ = (v) => pad.l + ((v - x0) / (x1 - x0)) * (W - pad.l - pad.r);
    const ys_ = (v) => H - pad.b - ((v - y0) / (y1 - y0 || 1)) * (H - pad.t - pad.b);
    const xt = niceTicks(x0, x1, 6).filter((v) => v >= x0 && v <= x1).map((v) => ({ v, label: fmt(v) }));
    frame(svg, W, H, pad, xt, yt, xs_, ys_, opts.xLabel, opts.yLabel);
    series.forEach((s, i) => {
      const color = s.color || PALETTE[i % PALETTE.length];
      const d = s.points.map((p, j) => (j ? "L" : "M") + xs_(p[0]).toFixed(1) + " " + ys_(p[1]).toFixed(1)).join(" ");
      if (!s.dashed) {
        const base = ys_(y0).toFixed(1);
        el("path", { d: d + ` L${xs_(s.points[s.points.length - 1][0]).toFixed(1)} ${base} L${xs_(s.points[0][0]).toFixed(1)} ${base} Z`, fill: color, opacity: 0.08 }, svg);
      }
      el("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.dashed ? "6 5" : "none" }, svg);
      if (s.points.length <= 12 || s.markers) {
        s.points.forEach((p) => {
          const c = el("circle", { cx: xs_(p[0]), cy: ys_(p[1]), r: 4, fill: color }, svg);
          el("title", {}, c).textContent = `${s.name}: ${fmt(p[1])} @ ${fmt(p[0])}`;
        });
      }
    });
    if (series.length > 1 || opts.legend) legend(container, series);
  }

  /** bars: [{label, value, color?}] */
  function bars(container, data, opts = {}) {
    container.innerHTML = "";
    const W = 560, H = 240, pad = { l: 48, r: 14, t: 12, b: 34 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
    if (!data.length) { container.innerHTML = '<div class="empty">No data yet</div>'; return; }
    const max = Math.max(...data.map((d) => d.value), opts.yMax || 0);
    const yt = niceTicks(0, max, 5);
    const y1 = yt[yt.length - 1] || 1;
    const band = (W - pad.l - pad.r) / data.length;
    const ys_ = (v) => H - pad.b - (v / y1) * (H - pad.t - pad.b);
    const xs_ = (i) => pad.l + band * i + band / 2;
    frame(svg, W, H, pad, data.map((d, i) => ({ v: i, label: d.label })), yt, xs_, ys_, opts.xLabel, opts.yLabel);
    data.forEach((d, i) => {
      const bw = Math.min(70, band * 0.6);
      const y = ys_(d.value);
      const r = el("rect", { x: xs_(i) - bw / 2, y, width: bw, height: H - pad.b - y, rx: 0, fill: d.color || PALETTE[i % PALETTE.length] }, svg);
      el("title", {}, r).textContent = `${d.label}: ${fmt(d.value)}`;
      el("text", { x: xs_(i), y: y - 5, "text-anchor": "middle", style: "fill:#1f2937;font-weight:600" }, svg).textContent = opts.format ? opts.format(d.value) : fmt(d.value);
    });
    if (opts.overlay) {
      const pts = opts.overlay.points.map((v, i) => [xs_(i), ys_(v)]);
      el("path", { d: pts.map((p, j) => (j ? "L" : "M") + p[0] + " " + p[1]).join(" "), fill: "none", stroke: opts.overlay.color || "#ea580c", "stroke-width": 2, "stroke-dasharray": "6 5" }, svg);
      legend(container, [{ name: opts.series || "measured", color: PALETTE[0] }, { name: opts.overlay.name, color: opts.overlay.color || "#ea580c", dashed: true }]);
    }
  }

  window.Charts = { line, bars, PALETTE };
})();
