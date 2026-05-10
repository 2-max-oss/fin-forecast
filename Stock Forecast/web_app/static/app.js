const state = {
  data: null,
  activeTab: "summary",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const compactMoney = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 2,
});

function fmtMoney(value) {
  return Number.isFinite(value) ? money.format(value) : "N/A";
}

function fmtCompactMoney(value) {
  return Number.isFinite(value) ? compactMoney.format(value) : "N/A";
}

function fmtPct(value, decimals = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(decimals)}%` : "N/A";
}

function fmtSignedPct(value, decimals = 1) {
  return Number.isFinite(value)
    ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(decimals)}%`
    : "N/A";
}

function fmtNum(value, decimals = 2) {
  return Number.isFinite(value) ? value.toFixed(decimals) : "N/A";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function toneClass(score) {
  if (!Number.isFinite(score)) return "";
  if (score >= 65) return "good";
  if (score >= 45) return "watch";
  return "bad";
}

function setStatus(message, isError = false) {
  const status = $("#status");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

async function analyzeTicker(ticker) {
  setStatus(`Analyzing ${ticker.toUpperCase()}...`);
  $("#ticker-form button").disabled = true;

  try {
    const response = await fetch(`/api/analyze/${encodeURIComponent(ticker)}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis failed.");
    }
    state.data = payload;
    state.activeTab = "summary";
    render();
    setStatus("");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    $("#ticker-form button").disabled = false;
  }
}

function render() {
  const data = state.data;
  if (!data) return;

  $("#overview").classList.remove("is-empty");
  $("#visuals").classList.remove("is-hidden");
  $("#tabs-section").classList.remove("is-hidden");

  $("#company-meta").textContent = `${data.company.exchange || "Market"} · ${data.company.sector || "Sector"} · ${data.company.industry || "Industry"}`;
  $("#company-title").textContent = `${data.ticker} · ${data.company.name}`;
  $("#current-price").textContent = fmtMoney(data.currentPrice);
  $("#price-date").textContent = data.priceDate ? `As of ${data.priceDate.slice(0, 10)}` : "";

  const composite = data.composite;
  $("#overall-score").textContent = `${fmtNum(composite.overallScore, 1)}/100`;
  $("#rating").innerHTML = `<span class="pill ${toneClass(composite.overallScore)}">${escapeHtml(composite.rating)}</span>`;
  $("#confidence").textContent = fmtPct(composite.confidence, 0);
  $("#risk-free").textContent = fmtPct(data.macro.riskFreeRate, 2);

  $("#history-label").textContent = `${data.history.latest.length} trading days`;
  drawPriceChart($("#price-chart"), data.history.latest);

  const forecast12 = data.modules.forecast?.ensemble?.find((item) => item.horizonMonths === 12);
  $("#forecast-label").textContent = forecast12
    ? `${fmtSignedPct(forecast12.median / forecast12.currentPrice - 1)} median`
    : "";
  drawForecastChart($("#forecast-chart"), forecast12);

  renderTabs();
  renderTabContent();
}

function renderTabs() {
  $$(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.tab === state.activeTab);
  });
}

function metric(label, value) {
  return `
    <div class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${value}</strong>
    </div>
  `;
}

function renderWarnings(score) {
  const warnings = score?.warnings || [];
  if (!warnings.length) return "";
  return `
    <ul class="warning-list">
      ${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
  `;
}

function renderTabContent() {
  const data = state.data;
  const content = $("#tab-content");

  const renderers = {
    summary: renderSummary,
    forecast: renderForecast,
    valuation: renderValuation,
    fundamental: renderFundamental,
    technical: renderTechnical,
    risk: renderRisk,
  };

  content.innerHTML = renderers[state.activeTab](data);
}

function renderSummary(data) {
  const modules = data.composite.modules || [];
  const rows = modules
    .map(
      (module) => `
        <tr>
          <td>${escapeHtml(module.name)}</td>
          <td><span class="pill ${toneClass(module.score)}">${fmtNum(module.score, 1)}</span></td>
          <td>${fmtPct(module.confidence, 0)}</td>
          <td>${fmtNum(data.composite.weightsUsed[module.name] * 100, 0)}%</td>
        </tr>
      `,
    )
    .join("");

  const errors = Object.entries(data.errors || {});
  return `
    <div class="metric-grid">
      ${metric("Market Cap", fmtCompactMoney(data.company.marketCap))}
      ${metric("Dividend Yield", fmtPct(data.company.dividendYield, 2))}
      ${metric("Confidence Band", `${fmtNum(data.composite.confidenceBand[0], 1)}-${fmtNum(data.composite.confidenceBand[1], 1)}`)}
      ${metric("Sector ETF", escapeHtml(data.macro.sectorEtf || "N/A"))}
    </div>
    <div class="split" style="margin-top: 16px;">
      <div>
        <h3>Module Scores</h3>
        <table>
          <thead><tr><th>Module</th><th>Score</th><th>Confidence</th><th>Weight</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div>
        <h3>Company</h3>
        <table>
          <tbody>
            <tr><th>Country</th><td>${escapeHtml(data.company.country || "N/A")}</td></tr>
            <tr><th>Employees</th><td>${Number.isFinite(data.company.employees) ? data.company.employees.toLocaleString() : "N/A"}</td></tr>
            <tr><th>Enterprise Value</th><td>${fmtCompactMoney(data.company.enterpriseValue)}</td></tr>
            <tr><th>Errors</th><td>${errors.length ? errors.map(([k, v]) => `${escapeHtml(k)}: ${escapeHtml(v)}`).join("<br>") : "None"}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderForecast(data) {
  const forecast = data.modules.forecast;
  if (!forecast) return "<p>No forecast module available.</p>";

  const rows = forecast.ensemble
    .map((dist) => {
      const ret = dist.currentPrice ? dist.median / dist.currentPrice - 1 : null;
      return `
        <tr>
          <td>${dist.horizonMonths}M</td>
          <td>${fmtMoney(dist.median)}</td>
          <td>${fmtSignedPct(ret)}</td>
          <td>${fmtPct(dist.probPositive, 1)}</td>
          <td>${fmtMoney(dist.p5)} - ${fmtMoney(dist.p95)}</td>
        </tr>
      `;
    })
    .join("");

  const rnRows = forecast.riskNeutral
    .map(
      (dist) => `
        <tr>
          <td>${dist.horizonMonths}M</td>
          <td>${fmtMoney(dist.mean)}</td>
          <td>${fmtPct(dist.probPositive, 1)}</td>
          <td>${fmtMoney(dist.p10)} - ${fmtMoney(dist.p90)}</td>
        </tr>
      `,
    )
    .join("");

  const mf = forecast.mathFinance || {};
  return `
    <div class="metric-grid">
      ${metric("Forecast Score", `${fmtNum(forecast.score.score, 1)}/100`)}
      ${metric("Physical Drift", fmtPct(mf.physicalDrift, 1))}
      ${metric("Risk-Neutral Drift", fmtPct(mf.riskNeutralDrift, 1))}
      ${metric("Adjusted Volatility", fmtPct(mf.adjustedVolatility, 1))}
    </div>
    <div class="split" style="margin-top: 16px;">
      <div>
        <h3>Physical Forecast</h3>
        <table>
          <thead><tr><th>Horizon</th><th>Median</th><th>Return</th><th>P Positive</th><th>5-95 Range</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div>
        <h3>Risk-Neutral Distribution</h3>
        <table>
          <thead><tr><th>Horizon</th><th>Mean</th><th>P Positive</th><th>10-90 Range</th></tr></thead>
          <tbody>${rnRows}</tbody>
        </table>
      </div>
    </div>
    ${renderWarnings(forecast.score)}
  `;
}

function renderValuation(data) {
  const valuation = data.modules.valuation;
  if (!valuation) return "<p>No valuation module available.</p>";

  const rows = valuation.multiples
    .map(
      (m) => `
        <tr>
          <td>${escapeHtml(m.name)}</td>
          <td>${fmtNum(m.current, 1)}x</td>
          <td>${fmtNum(m.historical5yMedian, 1)}x</td>
          <td>${Number.isFinite(m.percentile) ? `${m.percentile.toFixed(0)}th` : "N/A"}</td>
        </tr>
      `,
    )
    .join("");

  return `
    <div class="metric-grid">
      ${metric("Valuation Score", `${fmtNum(valuation.score.score, 1)}/100`)}
      ${metric("Fair Value Mid", fmtMoney(valuation.fairValueMid))}
      ${metric("Upside", fmtSignedPct(valuation.upsidePct))}
      ${metric("Reverse DCF Growth", fmtPct(valuation.reverseDcf?.impliedGrowthRate, 1))}
    </div>
    <table style="margin-top: 16px;">
      <thead><tr><th>Multiple</th><th>Current</th><th>5Y Median</th><th>Percentile</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${renderWarnings(valuation.score)}
  `;
}

function renderFundamental(data) {
  const fundamental = data.modules.fundamental;
  if (!fundamental) return "<p>No fundamental module available.</p>";
  const m = fundamental.metrics;
  return `
    <div class="metric-grid">
      ${metric("Fundamental Score", `${fmtNum(fundamental.score.score, 1)}/100`)}
      ${metric("Revenue CAGR 5Y", fmtPct(m.revenueCagr5y, 1))}
      ${metric("Net Margin", fmtPct(m.netMargin, 1))}
      ${metric("ROIC", fmtPct(m.roic, 1))}
      ${metric("FCF Margin", fmtPct(m.fcfMargin, 1))}
      ${metric("FCF Conversion", fmtPct(m.fcfConversion, 1))}
      ${metric("Net Debt / EBITDA", fmtNum(m.netDebtToEbitda, 2))}
      ${metric("Interest Coverage", fmtNum(m.interestCoverage, 1))}
    </div>
    ${renderWarnings(fundamental.score)}
  `;
}

function renderTechnical(data) {
  const technical = data.modules.technical;
  if (!technical) return "<p>No technical module available.</p>";
  return `
    <div class="metric-grid">
      ${metric("Technical Score", `${fmtNum(technical.score.score, 1)}/100`)}
      ${metric("RSI 14", fmtNum(technical.rsi14, 1))}
      ${metric("SMA 50", fmtMoney(technical.sma50))}
      ${metric("SMA 200", fmtMoney(technical.sma200))}
      ${metric("MACD", fmtNum(technical.macdLine, 2))}
      ${metric("Bollinger %B", fmtNum(technical.bollingerPctB, 2))}
      ${metric("Support", (technical.supportLevels || []).slice(0, 2).map(fmtMoney).join(", ") || "N/A")}
      ${metric("Resistance", (technical.resistanceLevels || []).slice(0, 2).map(fmtMoney).join(", ") || "N/A")}
    </div>
    ${renderWarnings(technical.score)}
  `;
}

function renderRisk(data) {
  const risk = data.modules.risk;
  if (!risk) return "<p>No risk module available.</p>";
  return `
    <div class="metric-grid">
      ${metric("Risk Score", `${fmtNum(risk.score.score, 1)}/100`)}
      ${metric("Realized Vol 365D", fmtPct(risk.realizedVol365d, 1))}
      ${metric("Beta vs SPY", fmtNum(risk.betaSpy, 2))}
      ${metric("Sharpe 1Y", fmtNum(risk.sharpe1y, 2))}
      ${metric("Max Drawdown 1Y", fmtSignedPct(risk.maxDrawdown1y, 1))}
      ${metric("Current Drawdown", fmtSignedPct(risk.currentDrawdown, 1))}
      ${metric("Quarter Kelly", fmtPct(risk.quarterKelly, 1))}
      ${metric("Kelly Edge", fmtSignedPct(risk.kellyEdge, 1))}
    </div>
    ${renderWarnings(risk.score)}
  `;
}

function drawPriceChart(canvas, history) {
  drawLineChart(canvas, history.map((p) => p.price), {
    line: "#1f6feb",
    fill: "rgba(31, 111, 235, 0.10)",
    label: "Price",
  });
}

function drawForecastChart(canvas, dist) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!dist) return;

  const values = [dist.currentPrice, dist.p5, dist.p25, dist.median, dist.p75, dist.p95];
  const min = Math.min(...values) * 0.96;
  const max = Math.max(...values) * 1.04;
  const pad = { left: 56, right: 18, top: 16, bottom: 38 };
  const x0 = pad.left;
  const x1 = width - pad.right;
  const y = (value) => height - pad.bottom - ((value - min) / (max - min || 1)) * (height - pad.top - pad.bottom);

  drawGrid(ctx, width, height, pad);

  function band(low, high, color) {
    ctx.beginPath();
    ctx.moveTo(x0, y(dist.currentPrice));
    ctx.lineTo(x1, y(low));
    ctx.lineTo(x1, y(high));
    ctx.lineTo(x0, y(dist.currentPrice));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  band(dist.p5, dist.p95, "rgba(31, 111, 235, 0.08)");
  band(dist.p25, dist.p75, "rgba(31, 111, 235, 0.18)");

  ctx.beginPath();
  ctx.moveTo(x0, y(dist.currentPrice));
  ctx.lineTo(x1, y(dist.median));
  ctx.strokeStyle = "#1f6feb";
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.fillStyle = "#111827";
  ctx.font = "18px system-ui";
  ctx.fillText(fmtMoney(dist.median), x1 - 112, y(dist.median) - 10);
  drawYAxisLabels(ctx, min, max, width, height, pad);
}

function drawLineChart(canvas, values, options) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!values.length) return;

  const min = Math.min(...values) * 0.96;
  const max = Math.max(...values) * 1.04;
  const pad = { left: 56, right: 18, top: 16, bottom: 38 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const x = (idx) => pad.left + (idx / Math.max(values.length - 1, 1)) * chartWidth;
  const y = (value) => height - pad.bottom - ((value - min) / (max - min || 1)) * chartHeight;

  drawGrid(ctx, width, height, pad);

  ctx.beginPath();
  values.forEach((value, idx) => {
    if (idx === 0) ctx.moveTo(x(idx), y(value));
    else ctx.lineTo(x(idx), y(value));
  });
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = options.fill;
  ctx.fill();

  ctx.beginPath();
  values.forEach((value, idx) => {
    if (idx === 0) ctx.moveTo(x(idx), y(value));
    else ctx.lineTo(x(idx), y(value));
  });
  ctx.strokeStyle = options.line;
  ctx.lineWidth = 3;
  ctx.stroke();

  drawYAxisLabels(ctx, min, max, width, height, pad);
}

function drawGrid(ctx, width, height, pad) {
  ctx.strokeStyle = "#d8e1ea";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }
}

function drawYAxisLabels(ctx, min, max, width, height, pad) {
  ctx.fillStyle = "#637083";
  ctx.font = "14px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const value = max - ((max - min) * i) / 4;
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.fillText(fmtMoney(value), pad.left - 8, y);
  }
  ctx.textAlign = "left";
}

$("#ticker-form").addEventListener("submit", (event) => {
  event.preventDefault();
  analyzeTicker($("#ticker-input").value.trim() || "AAPL");
});

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    state.activeTab = tab.dataset.tab;
    renderTabs();
    renderTabContent();
  });
});

