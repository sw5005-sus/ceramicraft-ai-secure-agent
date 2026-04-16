from enum import Enum

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(
    prefix="/demo",
    tags=["demo", "mock"],
)

DEMO_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Demo API 交互页</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f6f8fb; margin: 0; padding: 24px; }
    .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
    h1 { margin-top: 0; color: #1f2937; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .field { display: flex; flex-direction: column; gap: 6px; min-width: 220px; }
    input, select, button { padding: 10px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 14px; }
    button { cursor: pointer; border: none; background: #2563eb; color: #fff; }
    button:disabled { background: #94a3b8; cursor: not-allowed; }
    .secondary { background: #0f766e; }
    .panel { margin-top: 14px; padding: 12px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
    .status { font-size: 13px; color: #334155; margin-top: 8px; }
    .error { color: #b91c1c; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Demo API 交互页面</h1>
    <p>流程：先生成 Mock 数据并查看 Feature，确认后再进行风险评估。</p>

    <div class="row">
      <div class="field">
        <label for="userId">User ID</label>
        <input id="userId" type="number" min="1" value="1001" />
      </div>
      <div class="field">
        <label for="opType">Mock 类型</label>
        <select id="opType">
          <option value="normal">normal</option>
          <option value="block">block</option>
          <option value="manual_review">manual_review</option>
          <option value="watchlist">watchlist</option>
          <option value="clear">clear</option>
        </select>
      </div>
    </div>

    <div class="row">
      <button id="mockBtn">1) 生成 Mock 并提取 Feature</button>
      <button id="riskBtn" class="secondary" disabled>2) 确认 Feature 后评估风险</button>
    </div>

    <div class="status" id="status"></div>

    <div class="panel">
      <strong>Feature 结果</strong>
      <pre id="featureResult">暂无数据</pre>
    </div>

    <div class="panel">
      <strong>风险评估结果</strong>
      <pre id="riskResult">暂无数据</pre>
    </div>
  </div>

  <script>
    const mockBtn = document.getElementById("mockBtn");
    const riskBtn = document.getElementById("riskBtn");
    const statusEl = document.getElementById("status");
    const featureResult = document.getElementById("featureResult");
    const riskResult = document.getElementById("riskResult");

    let confirmedUserId = null;

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? "status error" : "status";
    }

    async function requestJson(url, options = {}) {
      const res = await fetch(url, options);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(JSON.stringify(data));
      }
      return data;
    }

    mockBtn.addEventListener("click", async () => {
      const userId = document.getElementById("userId").value;
      const opType = document.getElementById("opType").value;

      if (!userId) {
        setStatus("请先输入 user_id", true);
        return;
      }

      setStatus("正在生成 Mock 数据并提取 Feature...");
      riskBtn.disabled = true;
      riskResult.textContent = "暂无数据";

      try {
        const query = new URLSearchParams({ user_id: userId, op_type: opType });
        const data = await requestJson(`/demo/mock-data?${query.toString()}`, { method: "POST" });
        featureResult.textContent = JSON.stringify(data, null, 2);
        confirmedUserId = userId;
        riskBtn.disabled = false;
        setStatus("Feature 已生成，请确认后点击风险评估。");
      } catch (error) {
        featureResult.textContent = "请求失败";
        setStatus(`生成失败: ${error.message}`, true);
      }
    });

    riskBtn.addEventListener("click", async () => {
      if (!confirmedUserId) {
        setStatus("请先生成并确认 Feature", true);
        return;
      }

      setStatus("正在进行风险评估...");

      try {
        const query = new URLSearchParams({ user_id: confirmedUserId });
        const data = await requestJson(`/demo/risk-access?${query.toString()}`, { method: "POST" });
        riskResult.textContent = JSON.stringify(data, null, 2);
        setStatus("风险评估完成。");
      } catch (error) {
        riskResult.textContent = "请求失败";
        setStatus(`评估失败: ${error.message}`, true);
      }
    });
  </script>
</body>
</html>
"""


class OpType(str, Enum):
    clear = "clear"
    normal = "normal"
    block = "block"
    manual_review = "manual_review"
    watchlist = "watchlist"


@router.post("/mock-data")
def gen_mock_data(user_id: int, op_type: OpType):
    """Generate mock data for testing."""
    import ceramicraft_ai_secure_agent.mock.gen_mock_data as gen_mock_data
    from ceramicraft_ai_secure_agent.service.feature_service import extract_features

    if op_type == OpType.clear:
        gen_mock_data.clear_mock_data(user_id=user_id)
    elif op_type == OpType.normal:
        gen_mock_data.gen_mock_normal(user_id=user_id)
    elif op_type == OpType.block:
        gen_mock_data.gen_mock_block(user_id=user_id)
    elif op_type == OpType.manual_review:
        gen_mock_data.gen_mock_manual_review(user_id=user_id)
    elif op_type == OpType.watchlist:
        gen_mock_data.gen_mock_watchlist(user_id=user_id)
    return extract_features(user_id=user_id)


@router.post("/risk-access")
def risk_access(user_id: int):
    """Assess risk for the specified user."""
    from ceramicraft_ai_secure_agent.service.agent_service import assess_risk

    return assess_risk(user_id=user_id)


@router.get("/page", response_class=HTMLResponse)
def demo_page():
    """Interactive HTML page for demo APIs."""
    return HTMLResponse(content=DEMO_HTML)
