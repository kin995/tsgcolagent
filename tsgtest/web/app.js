(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { jobId: null, job: null, items: [], active: 0, timer: null, browsePath: null, root: "", runName: null };

  const pct = (v) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : (Number(v) * 100).toFixed(1) + "%");
  const num = (v, digits = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  function setPill(el, text, tone) {
    el.textContent = text;
    el.className = "pill" + (tone ? " " + tone : "");
  }

  // ---------- 初始化 ----------
  async function init() {
    bind();
    try {
      const health = await fetchJSON("/api/health");
      const deps = health.dependencies || {};
      const missing = Object.entries(deps).filter(([, ok]) => !ok).map(([name]) => name);
      if (missing.length === 0) setPill($("dep-pill"), "引擎就绪", "ok");
      else {
        setPill($("dep-pill"), `缺少依赖 ${missing.length} 项`, "warn");
        $("dep-pill").title = "未安装：" + missing.join(", ") + "（pip install -r requirements.txt）";
      }
      $("base-url").value = health.ollama_base_url || "";
      state.root = health.root || "";
      if (health.provider) $("provider").value = health.provider;
      toggleApiKey();
      if (health.provider === "openai" && health.has_api_key === false) {
        setPill($("ollama-pill"), "API 已选 · 缺少 key", "warn");
        $("ollama-pill").title = "把 key 写入 api_key.txt，或设置环境变量";
      }
    } catch (err) {
      setPill($("dep-pill"), "服务异常", "bad");
    }
    await loadModels();
    await loadHistory();
    await loadRuns();
  }

  function toggleApiKey() {
    const field = $("api-key-field");
    if (field) field.hidden = $("provider").value !== "openai";
  }

  function bind() {
    $("start-btn").addEventListener("click", start);
    $("demo-btn").addEventListener("click", loadDemo);
    $("cancel-btn").addEventListener("click", cancelJob);
    $("export-md").addEventListener("click", () => exportReport("markdown"));
    $("export-html").addEventListener("click", () => exportReport("html"));
    $("refresh-models").addEventListener("click", loadModels);
    $("api-key").addEventListener("blur", loadModels);
    $("provider").addEventListener("change", () => {
      if ($("provider").value === "openai") {
        $("base-url").value = "https://api.deepseek.com";
      } else {
        $("base-url").value = "http://127.0.0.1:11434";
      }
      toggleApiKey();
      loadModels();
    });
    $("browse-btn").addEventListener("click", () => openBrowse($("source").value.trim() || null));
    $("sample-btn").addEventListener("click", () => {
      if (!state.root) return;
      $("source").value = state.root + "\\examples\\sample_python\\inventory.py";
      $("rounds").value = "1";
    });
    $("browse-close").addEventListener("click", () => $("browse-modal").hidden = true);
    $("browse-up").addEventListener("click", () => openBrowse(state.browsePath ? dirname(state.browsePath) : null));
    $("browse-home").addEventListener("click", () => openBrowse(null));
    $("copy").addEventListener("click", copyCode);
    $("download").addEventListener("click", downloadCode);
    $("source").addEventListener("keydown", (e) => { if (e.key === "Enter") start(); });
  }

  function dirname(path) {
    const normalized = path.replace(/[\\/]+$/, "");
    const index = Math.max(normalized.lastIndexOf("\\"), normalized.lastIndexOf("/"));
    return index > 0 ? normalized.slice(0, index) : normalized;
  }

  // ---------- 模型 ----------
  function fillModels(names, selected) {
    const select = $("model");
    select.innerHTML = "";
    if (!names || names.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "（未发现模型）";
      select.appendChild(option);
      return;
    }
    for (const name of names) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    if (selected && names.includes(selected)) select.value = selected;
  }

  async function loadModels() {
    const provider = $("provider").value;
    if (provider === "openai") {
      if (!$("base-url").value.trim() || $("base-url").value.includes("11434")) {
        $("base-url").value = "https://api.deepseek.com";
      }
      setPill($("ollama-pill"), "API 模式", "ok");
      $("ollama-pill").title = "key 从环境变量或 api_key.txt 读取";
      const params = new URLSearchParams({ provider: "openai", base_url: $("base-url").value.trim() });
      const key = $("api-key").value.trim();
      if (key) params.set("api_key", key);
      try {
        const data = await fetchJSON("/api/models?" + params.toString());
        if (data.models && data.models.length > 0) {
          fillModels(data.models, data.models.includes("deepseek-chat") ? "deepseek-chat" : data.models[0]);
        } else {
          fillModels(["deepseek-chat", "deepseek-reasoner"], "deepseek-chat");
        }
      } catch (err) {
        fillModels(["deepseek-chat", "deepseek-reasoner"], "deepseek-chat");
      }
      return;
    }
    setPill($("ollama-pill"), "Ollama 检测中…", "");
    try {
      const base = $("base-url").value.trim();
      const data = await fetchJSON("/api/models" + (base ? "?base_url=" + encodeURIComponent(base) : ""));
      fillModels(data.models, $("model").value);
      if (!data.models || data.models.length === 0) {
        setPill($("ollama-pill"), "Ollama 未连接", "bad");
        $("ollama-pill").title = data.error || "请先启动 ollama serve 并 ollama pull 模型";
        return;
      }
      setPill($("ollama-pill"), `Ollama · ${data.models.length} 个模型`, "ok");
    } catch (err) {
      setPill($("ollama-pill"), "Ollama 未连接", "bad");
    }
  }

  // ---------- 启动任务 ----------
  async function start() {
    const source = $("source").value.trim();
    const error = $("form-error");
    if (!source) {
      error.textContent = "请先填写源文件或目录。";
      error.hidden = false;
      return;
    }
    error.hidden = true;
    $("start-btn").disabled = true;
    $("metrics").hidden = true;
    $("test-block").hidden = true;
    $("job-log").textContent = "";
    try {
      const payload = {
        source,
        provider: $("provider").value,
        model: $("model").value || undefined,
        ollama_base_url: $("base-url").value.trim() || undefined,
        temperature: Number($("temperature").value),
        max_rounds: Number($("rounds").value),
        api_key: $("provider").value === "openai" ? ($("api-key").value.trim() || undefined) : undefined,
      };
      const data = await fetchJSON("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.jobId = data.job.id;
      renderJob(data.job);
      schedulePoll();
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
      $("start-btn").disabled = false;
    }
  }

  function schedulePoll() {
    clearTimeout(state.timer);
    state.timer = setTimeout(poll, 1200);
  }

  async function poll() {
    if (!state.jobId) return;
    try {
      const job = await fetchJSON("/api/jobs/" + state.jobId);
      renderJob(job);
      if (job.status === "queued" || job.status === "running") schedulePoll();
      else {
        $("start-btn").disabled = false;
        $("cancel-btn").hidden = true;
        loadHistory();
        loadRuns();
        if (job.status === "done" && job.out_dir) {
          const parts = String(job.out_dir).split(/[\\/]/).filter(Boolean);
          loadRun(parts[parts.length - 1]);
        }
      }
    } catch (err) {
      schedulePoll();
    }
  }

  function renderJob(job) {
    state.job = job;
    const badge = $("job-status");
    const labels = { queued: "排队中", running: "生成中", done: "已完成", failed: "失败", cancelled: "已取消" };
    badge.textContent = labels[job.status] || job.status;
    badge.className = "badge " + job.status;
    $("cancel-btn").hidden = !(job.status === "queued" || job.status === "running");
    const elapsed = job.started_at ? Math.max(0, (Date.now() - new Date(job.started_at).getTime()) / 1000) : 0;
    $("job-meta").textContent = `${job.source}  →  ${job.out_dir}` + (job.status === "running" ? `  ·  ${elapsed.toFixed(0)}s` : "");
    $("job-log").textContent = (job.log || []).join("\n");
    $("job-log").scrollTop = $("job-log").scrollHeight;
    if (job.status === "failed") {
      const error = $("form-error");
      error.textContent = job.error || "任务失败";
      error.hidden = false;
    }
    if (job.result && job.status === "done") {
      renderResult(job.result);
      if (job.out_dir) {
        const parts = String(job.out_dir).split(/[\\/]/).filter(Boolean);
        state.runName = parts[parts.length - 1] || null;
        $("export-block").hidden = !state.runName;
        $("rounds-block").hidden = true;
        $("llm-block").hidden = false;
      }
    }
  }

  function renderResult(result) {
    state.items = result.items || [];
    if (state.items.length === 0) return;
    state.active = 0;
    renderMetrics();
    renderTabs();
    renderCode();
  }

  function renderMetrics() {
    const item = state.items[state.active] || {};
    const m = item.metrics || {};
    const cells = [
      ["行覆盖 LC", pct(m.line_coverage)],
      ["分支覆盖 BC", pct(m.branch_coverage)],
      ["断言有效性 AE", pct(m.ae)],
      ["变异得分 MS", pct(m.mutation_score)],
      ["通过率", pct(m.pass_rate)],
      ["目标改善 TIR", m.tir === null || m.tir === undefined ? "—" : pct(m.tir)],
      ["平均执行", num(m.avg_exec)],
      ["耗时", m.duration_seconds === null || m.duration_seconds === undefined ? "—" : num(m.duration_seconds, 1) + "s"],
    ];
    $("metrics").innerHTML = cells
      .map(([k, v]) => `<div class="metric"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`)
      .join("");
    $("metrics").hidden = false;
  }

  function renderTabs() {
    const tabs = $("test-tabs");
    tabs.innerHTML = "";
    if (state.items.length <= 1) return;
    state.items.forEach((item, index) => {
      const button = document.createElement("button");
      button.textContent = (item.source || "").split(/[\\/]/).pop() || `#${index + 1}`;
      button.className = index === state.active ? "active" : "";
      button.addEventListener("click", () => {
        state.active = index;
        renderMetrics();
        renderTabs();
        renderCode();
      });
      tabs.appendChild(button);
    });
  }

  function renderCode() {
    const item = state.items[state.active] || {};
    $("test-code").querySelector("code").textContent = item.test_text || "（没有测试内容）";
    $("test-block").hidden = false;
  }

  async function copyCode() {
    const item = state.items[state.active] || {};
    try {
      await navigator.clipboard.writeText(item.test_text || "");
      $("copy").textContent = "已复制";
      setTimeout(() => ($("copy").textContent = "复制"), 1200);
    } catch (err) {
      $("copy").textContent = "复制失败";
    }
  }

  function downloadCode() {
    const item = state.items[state.active] || {};
    if (!item.test_text) return;
    const name = (item.test_path || "test.py").split(/[\\/]/).pop();
    const blob = new Blob([item.test_text], { type: "text/plain;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }

  // ---------- 离线示例 ----------
  async function loadDemo() {
    try {
      const data = await fetchJSON("/api/demo");
      state.items = data.items || [];
      state.active = 0;
      const badge = $("job-status");
      badge.textContent = "示例运行";
      badge.className = "badge done";
      $("job-meta").textContent = "内置示例（已脱敏）：" + ((state.items[0] || {}).source || "");
      $("job-log").textContent = "离线示例：展示真实运行产出的指标与测试代码，不需要 Ollama。";
      renderResult(data);
    } catch (err) {
      const error = $("form-error");
      error.textContent = "示例加载失败：" + err.message;
      error.hidden = false;
    }
  }

  // ---------- 本地运行 ----------
  async function loadRuns() {
    try {
      const data = await fetchJSON("/api/runs");
      const list = $("runs");
      list.innerHTML = "";
      if (!data.runs || data.runs.length === 0) {
        list.innerHTML = '<li class="muted">暂无本地运行</li>';
        return;
      }
      for (const run of data.runs.slice(0, 20)) {
        const li = document.createElement("li");
        const label = (run.source || "").split(/[\\/]/).filter(Boolean).pop() || run.name;
        const stateLabel = run.status || run.quality || "";
        li.innerHTML = `<span class="name">${esc(label)}</span><span class="state">${esc(pct(run.line_coverage))} · ${esc(stateLabel)}</span>`;
        li.title = `${run.name} · ${run.created_at || ""}`;
        li.addEventListener("click", () => loadRun(run.name));
        list.appendChild(li);
      }
    } catch (err) {
      $("runs").innerHTML = '<li class="muted">读取失败</li>';
    }
  }

  async function loadRun(name) {
    try {
      const run = await fetchJSON("/api/runs/" + encodeURIComponent(name));
      state.runName = name;
      state.jobId = null;
      state.items = [{ source: run.source, test_text: run.test_text, metrics: run.metrics }];
      state.active = 0;
      const badge = $("job-status");
      badge.textContent = "历史运行";
      badge.className = "badge done";
      $("cancel-btn").hidden = true;
      $("job-meta").textContent = `${run.source || ""}  →  ${run.status || ""}${run.stop_reason ? "（" + run.stop_reason + "）" : ""}`;
      renderMetrics();
      renderCode();
      renderRounds(run.rounds || []);
      renderLlm(run.llm || {});
      $("export-block").hidden = false;
      $("job-log").textContent = run.error || "历史运行：无实时日志，可查看轮次与指标。";
    } catch (err) {
      const error = $("form-error");
      error.textContent = "加载运行失败：" + err.message;
      error.hidden = false;
    }
  }

  function renderRounds(rounds) {
    const body = $("rounds-body");
    body.innerHTML = "";
    for (const item of rounds) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${item.round ?? ""}</td><td>${item.passed ?? ""}</td>` +
        `<td>${pct(item.line_coverage)}</td><td>${pct(item.branch_coverage)}</td>` +
        `<td>${pct(item.ae)}</td><td>${pct(item.mutation_score)}</td><td>${item.tests ?? ""}</td>`;
      body.appendChild(tr);
    }
    $("rounds-block").hidden = rounds.length === 0;
  }

  function renderLlm(llm) {
    const seconds = llm.duration_seconds;
    const cells = [
      ["AI 调用", llm.calls ?? "—"],
      ["输入 tokens", llm.prompt_tokens ?? "—"],
      ["输出 tokens", llm.completion_tokens ?? "—"],
      ["模型耗时", seconds == null ? "—" : Number(seconds).toFixed(1) + "s"],
    ];
    $("llm-block").innerHTML = cells
      .map(([key, value]) => `<div class="metric"><div class="k">${esc(key)}</div><div class="v">${esc(value)}</div></div>`)
      .join("");
    $("llm-block").hidden = false;
  }

  async function cancelJob() {
    if (!state.jobId) return;
    try {
      await fetchJSON("/api/jobs/" + state.jobId + "/cancel", { method: "POST" });
    } catch (err) {
      /* 忽略取消失败 */
    }
  }

  function exportReport(format) {
    if (!state.runName) return;
    window.location = `/api/runs/${encodeURIComponent(state.runName)}/report?format=${format}`;
  }

  // ---------- 历史 ----------
  async function loadHistory() {
    try {
      const data = await fetchJSON("/api/jobs");
      const list = $("history");
      list.innerHTML = "";
      if (!data.jobs || data.jobs.length === 0) {
        list.innerHTML = '<li class="muted">暂无任务</li>';
        return;
      }
      for (const job of data.jobs.slice(0, 12)) {
        const li = document.createElement("li");
        const name = (job.source || "").split(/[\\/]/).pop() || job.id;
        const labels = { queued: "排队", running: "运行", done: "完成", failed: "失败", cancelled: "取消" };
        li.innerHTML = `<span class="name">${esc(name)}</span><span class="state">${esc(labels[job.status] || job.status)}</span>`;
        li.title = job.out_dir;
        li.addEventListener("click", async () => {
          state.jobId = job.id;
          renderJob(await fetchJSON("/api/jobs/" + job.id));
        });
        list.appendChild(li);
      }
    } catch (err) {
      /* 忽略历史加载错误 */
    }
  }

  // ---------- 文件浏览 ----------
  async function openBrowse(path) {
    try {
      const data = await fetchJSON("/api/browse" + (path ? "?path=" + encodeURIComponent(path) : ""));
      state.browsePath = data.path;
      $("browse-path").textContent = data.path;
      const list = $("browse-list");
      list.innerHTML = "";
      for (const dir of data.dirs || []) {
        const li = document.createElement("li");
        li.innerHTML = `<span>📁 ${esc(dir.name)}</span><span class="tag">目录</span>`;
        li.addEventListener("click", () => openBrowse(dir.path));
        list.appendChild(li);
      }
      for (const file of data.files || []) {
        const li = document.createElement("li");
        li.innerHTML = `<span>${esc(file.name)}</span><span class="tag">${esc(file.language)}</span>`;
        li.addEventListener("click", () => {
          $("source").value = file.path;
          $("browse-modal").hidden = true;
        });
        list.appendChild(li);
      }
      if (!list.children.length) list.innerHTML = '<li class="muted">此目录没有可用的源文件</li>';
      $("browse-modal").hidden = false;
    } catch (err) {
      const error = $("form-error");
      error.textContent = err.message;
      error.hidden = false;
    }
  }

  init();
})();
