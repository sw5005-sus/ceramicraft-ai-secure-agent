// ===== base elements =====
const mockBtn = document.getElementById("mockBtn");
const riskBtn = document.getElementById("riskBtn");
const statusEl = document.getElementById("status");
const featureResult = document.getElementById("featureResult");
const riskResult = document.getElementById("riskResult");

let confirmedUserId = null;

// ===== utils =====
function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.className = isError ? "status error" : "status";
}

function safeParseJSON(str) {
  try {
    return typeof str === "string" ? JSON.parse(str) : str;
  } catch {
    return null;
  }
}

// ===== feature table (保持原来的简单表格) =====
function renderTable(target, data) {
  target.innerHTML = "";
  target.className = "result";

  const rows = Object.entries(data || {});
  if (!rows.length) {
    target.textContent = "No data";
    return;
  }

  const table = document.createElement("table");
  table.innerHTML = `
    <thead><tr><th>key</th><th>value</th></tr></thead>
    <tbody>
      ${rows
        .map(
          ([k, v]) =>
            `<tr><td>${k}</td><td>${typeof v === "object" ? JSON.stringify(v) : v}</td></tr>`
        )
        .join("")}
    </tbody>
  `;
  target.appendChild(table);
}

// ===== ⭐ 核心：好看的 Risk UI =====
function renderRiskAssessment(target, data) {
    target.innerHTML = "";
  
    const rec = safeParseJSON(data.recommendation);
  
    target.innerHTML = `
      <div class="risk-section">
        <div class="version-row">
          <span class="version-badge">Model ${data.model_version || "-"}</span>
          <span class="version-badge">Prompt ${data.prompt_version || "-"}</span>
        </div>
        <div class="risk-section-title">Risk Summary</div>
        <span class="risk-badge risk-badge-${data.risk_level?.toLowerCase()}">
          ${data.risk_level}
        </span>
  
        <div class="score-row">
          <div class="score-item">Risk Score: <span>${Number(data.risk_score).toFixed(2)}</span></div>
          <div class="score-item">Fraud Prob: <span>${(Number(data.fraud_probability) * 100).toFixed(2)}%</span></div>
        </div>
      </div>
  
      <div class="risk-section">
        <div class="risk-section-title">Recommendation</div>
  
        <span class="action-badge action-badge-${rec?.recommended_action}">
          ${rec?.recommended_action || "N/A"}
        </span>
  
        <div class="score-row">
          <div class="score-item">Confidence: <span>${rec?.confidence || "N/A"}</span></div>
        </div>
  
        <div class="risk-section-title">Reason</div>
        <ul class="bullet-list">
          ${(rec?.reason || "")
            .split(".")
            .filter(Boolean)
            .map((r) => `<li>${r.trim()}.</li>`)
            .join("")}
        </ul>
  
        <div class="risk-section-title">Analyst Summary</div>
        <ul class="bullet-list">
          ${(rec?.analyst_summary || "")
            .split(".")
            .filter(Boolean)
            .map((r) => `<li>${r.trim()}.</li>`)
            .join("")}
        </ul>
  
        <div class="risk-section-title">Evidence</div>
        <div class="chip-list">
          ${(rec?.evidence || [])
            .map((e) => `<span class="chip">${e}</span>`)
            .join("")}
        </div>
      </div>
  
      <div class="risk-section">
        <div class="risk-section-title">Top Contributions</div>
        <table class="contrib-table">
          <thead>
            <tr><th>Feature</th><th>Impact</th><th>Ratio</th></tr>
          </thead>
          <tbody>
            ${(data.ml_top_contribution || [])
              .map(
                (c) => `
                <tr>
                  <td>${c.name}</td>
                  <td class="${c.impact > 0 ? "impact-positive" : "impact-negative"}">
                    ${Number(c.impact).toFixed(4)}
                  </td>
                  <td>${c.ratio}</td>
                </tr>
              `
              )
              .join("")}
          </tbody>
        </table>
      </div>
  
      <div class="risk-section">
        <div class="risk-section-title">Triggered Rules</div>
        <div class="chip-list">
          ${(data.triggered_rules || [])
            .map((r) => `<span class="chip">${r}</span>`)
            .join("")}
        </div>
      </div>
    `;
  }

// ===== API =====
async function requestJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || "Request failed");
  return data;
}

// ===== mock =====
mockBtn.onclick = async () => {
  const userId = document.getElementById("userId").value;
  const opType = document.getElementById("opType").value;

  setStatus("Generating...");
  riskBtn.disabled = true;

  try {
    const query = new URLSearchParams({ user_id: userId, op_type: opType });

    const data = await requestJson(
      `/ai-secure-agent/v1/demo/mock-data?${query}`,
      { method: "POST" }
    );

    renderTable(featureResult, data);
    confirmedUserId = userId;
    riskBtn.disabled = false;

    setStatus("Mock ready");
  } catch (e) {
    setStatus(e.message, true);
  }
};

// ===== risk =====
riskBtn.onclick = async () => {
  setStatus("Running risk...");

  try {
    const query = new URLSearchParams({ user_id: confirmedUserId });

    const data = await requestJson(
      `/ai-secure-agent/v1/demo/risk-access?${query}`,
      { method: "POST" }
    );

    // ⭐ 用这个，不要再用 renderTable
    renderRiskAssessment(riskResult, data);

    setStatus("Done");
  } catch (e) {
    setStatus(e.message, true);
  }
};