(function () {
  const config = window.SUMMARY_FRONTEND_CONFIG || {};
  const backendHttpUrl = config.backendHttpUrl || "http://localhost:8088";
  const backendWsUrl = config.backendWsUrl || backendHttpUrl.replace(/^http/, "ws");

  const storage = {
    jobId: "summaryFrontend.jobId",
    title: "summaryFrontend.title",
    inputText: "summaryFrontend.inputText",
    inputMode: "summaryFrontend.inputMode",
    queryText: "summaryFrontend.queryText",
    uploadFormat: "summaryFrontend.uploadFormat",
    uploadRawLineColumn: "summaryFrontend.uploadRawLineColumn",
    reuseFormat: "summaryFrontend.reuseFormat",
    reuseRawLineColumn: "summaryFrontend.reuseRawLineColumn",
    queryRawLineColumn: "summaryFrontend.queryRawLineColumn",
    autoStart: "summaryFrontend.autoStart",
  };

  const state = {
    jobId: localStorage.getItem(storage.jobId) || "",
    snapshot: null,
    websocket: null,
    pollTimer: null,
    inputMode: localStorage.getItem(storage.inputMode) || "text",
    uploads: [],
  };

  window.__summaryWsMessages = 0;

  const el = {
    form: document.getElementById("createJobForm"),
    title: document.getElementById("jobTitle"),
    inputText: document.getElementById("inputText"),
    modeTabs: Array.from(document.querySelectorAll(".mode-tab")),
    sourcePanels: {
      text: document.getElementById("textPanel"),
      upload: document.getElementById("uploadPanel"),
      reuse: document.getElementById("reusePanel"),
      query: document.getElementById("queryPanel"),
    },
    uploadFile: document.getElementById("uploadFile"),
    uploadFormat: document.getElementById("uploadFormat"),
    uploadRawLineColumn: document.getElementById("uploadRawLineColumn"),
    refreshUploadsButton: document.getElementById("refreshUploadsButton"),
    uploadedFileSelect: document.getElementById("uploadedFileSelect"),
    uploadedFileDetails: document.getElementById("uploadedFileDetails"),
    reuseFormat: document.getElementById("reuseFormat"),
    reuseRawLineColumn: document.getElementById("reuseRawLineColumn"),
    queryText: document.getElementById("queryText"),
    queryRawLineColumn: document.getElementById("queryRawLineColumn"),
    autoStart: document.getElementById("autoStart"),
    createButton: document.getElementById("createJobButton"),
    reloadButton: document.getElementById("reloadJobButton"),
    clearButton: document.getElementById("clearJobButton"),
    pauseButton: document.getElementById("pauseJobButton"),
    resumeButton: document.getElementById("resumeJobButton"),
    cancelButton: document.getElementById("cancelJobButton"),
    connectionBadge: document.getElementById("connectionBadge"),
    jobSubtitle: document.getElementById("jobSubtitle"),
    jobStatus: document.getElementById("jobStatus"),
    nodesDone: document.getElementById("nodesDone"),
    artifactTotal: document.getElementById("artifactTotal"),
    serverTime: document.getElementById("serverTime"),
    progressBar: document.getElementById("progressBar"),
    nodesBody: document.getElementById("nodesBody"),
    artifactsList: document.getElementById("artifactsList"),
    eventsList: document.getElementById("eventsList"),
    finalSummary: document.getElementById("finalSummary"),
  };

  function setText(node, value) {
    node.textContent = value == null || value === "" ? "-" : String(value);
  }

  function setBadge(text, className) {
    el.connectionBadge.className = "badge " + className;
    setText(el.connectionBadge, text);
  }

  function persistDraft() {
    localStorage.setItem(storage.title, el.title.value);
    localStorage.setItem(storage.inputText, el.inputText.value);
    localStorage.setItem(storage.inputMode, state.inputMode);
    localStorage.setItem(storage.queryText, el.queryText.value);
    localStorage.setItem(storage.uploadFormat, el.uploadFormat.value);
    localStorage.setItem(storage.uploadRawLineColumn, el.uploadRawLineColumn.value);
    localStorage.setItem(storage.reuseFormat, el.reuseFormat.value);
    localStorage.setItem(storage.reuseRawLineColumn, el.reuseRawLineColumn.value);
    localStorage.setItem(storage.queryRawLineColumn, el.queryRawLineColumn.value);
    localStorage.setItem(storage.autoStart, el.autoStart.checked ? "1" : "0");
  }

  function restoreDraft() {
    el.title.value = localStorage.getItem(storage.title) || "";
    el.inputText.value = localStorage.getItem(storage.inputText) || "";
    el.queryText.value = localStorage.getItem(storage.queryText) || "";
    el.uploadFormat.value = localStorage.getItem(storage.uploadFormat) || "auto";
    el.uploadRawLineColumn.value = localStorage.getItem(storage.uploadRawLineColumn) || "";
    el.reuseFormat.value = localStorage.getItem(storage.reuseFormat) || "";
    el.reuseRawLineColumn.value = localStorage.getItem(storage.reuseRawLineColumn) || "";
    el.queryRawLineColumn.value = localStorage.getItem(storage.queryRawLineColumn) || "";
    el.autoStart.checked = (localStorage.getItem(storage.autoStart) || "1") === "1";
  }

  async function requestJson(path, options) {
    const response = await fetch(backendHttpUrl + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    return response.json();
  }

  async function requestForm(path, formData) {
    const response = await fetch(backendHttpUrl + path, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    return response.json();
  }

  async function loadSnapshot() {
    if (!state.jobId) {
      renderEmpty();
      return;
    }
    const snapshot = await requestJson(`/summary-jobs/${state.jobId}/snapshot`);
    state.snapshot = snapshot;
    render(snapshot);
  }

  function setInputMode(mode) {
    state.inputMode = mode;
    for (const tab of el.modeTabs) {
      tab.classList.toggle("active", tab.dataset.mode === mode);
    }
    for (const [panelMode, panel] of Object.entries(el.sourcePanels)) {
      panel.classList.toggle("active", panelMode === mode);
    }
    persistDraft();
    if (mode === "reuse") {
      loadUploads().catch(() => setBadge("uploads error", "badge-bad"));
    }
  }

  function formatBytes(value) {
    const bytes = Number(value || 0);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  function renderUploads() {
    el.uploadedFileSelect.replaceChildren();
    if (!state.uploads.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No saved uploads";
      el.uploadedFileSelect.appendChild(option);
      el.uploadedFileDetails.textContent = "No saved uploads loaded";
      return;
    }

    for (const upload of state.uploads) {
      const option = document.createElement("option");
      option.value = upload.upload_id;
      option.disabled = !upload.available;
      const suffix = upload.available ? "" : " (missing)";
      option.textContent = `${upload.filename || upload.upload_id} · ${upload.source_format || "unknown"} · ${formatBytes(upload.size_bytes)}${suffix}`;
      el.uploadedFileSelect.appendChild(option);
    }
    renderSelectedUploadDetails();
  }

  function renderSelectedUploadDetails() {
    const upload = state.uploads.find((item) => item.upload_id === el.uploadedFileSelect.value);
    if (!upload) {
      el.uploadedFileDetails.textContent = "No upload selected";
      return;
    }
    const stagedAt = upload.staged_at ? new Date(upload.staged_at).toLocaleString() : "-";
    el.uploadedFileDetails.textContent = [
      `job=${upload.source_job_id}`,
      `status=${upload.job_status || "-"}`,
      `format=${upload.source_format || "-"}`,
      `size=${formatBytes(upload.size_bytes)}`,
      `staged=${stagedAt}`,
    ].join(" · ");
  }

  async function loadUploads() {
    const uploads = await requestJson("/summary-uploads?limit=200");
    state.uploads = uploads || [];
    renderUploads();
  }

  function startPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
    }
    state.pollTimer = setInterval(() => {
      loadSnapshot().catch(() => setBadge("poll error", "badge-bad"));
    }, 1500);
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function connectWebSocket() {
    if (!state.jobId) {
      return;
    }
    if (state.websocket) {
      state.websocket.close();
    }
    setBadge("connecting", "badge-warn");
    const socket = new WebSocket(`${backendWsUrl}/ws/summary-jobs/${encodeURIComponent(state.jobId)}`);
    state.websocket = socket;

    socket.addEventListener("open", () => {
      setBadge("live", "badge-live");
      stopPolling();
    });

    socket.addEventListener("message", (event) => {
      window.__summaryWsMessages += 1;
      const payload = JSON.parse(event.data);
      if (payload.type === "snapshot") {
        state.snapshot = payload.snapshot;
        render(payload.snapshot);
      }
      if (payload.type === "error") {
        setBadge("ws error", "badge-bad");
      }
    });

    socket.addEventListener("close", () => {
      if (state.jobId) {
        setBadge("polling", "badge-warn");
        startPolling();
      }
    });

    socket.addEventListener("error", () => {
      setBadge("ws error", "badge-bad");
      startPolling();
    });
  }

  function setActiveJob(jobId) {
    state.jobId = jobId;
    localStorage.setItem(storage.jobId, jobId);
    loadSnapshot().catch((error) => {
      setBadge("load error", "badge-bad");
      setText(el.finalSummary, error.message);
    });
    connectWebSocket();
  }

  function renderEmpty() {
    setText(el.jobSubtitle, "No active job");
    setText(el.jobStatus, "-");
    setText(el.nodesDone, "0/0");
    setText(el.artifactTotal, "0");
    setText(el.serverTime, "-");
    el.progressBar.style.width = "0%";
    el.nodesBody.replaceChildren();
    el.artifactsList.replaceChildren();
    el.eventsList.replaceChildren();
    setText(el.finalSummary, "-");
    setBadge("offline", "badge-muted");
  }

  function statusClass(status) {
    const normalized = String(status || "").toLowerCase();
    if (normalized.includes("done")) return "status-done";
    if (normalized.includes("running")) return "status-running";
    if (normalized.includes("failed")) return "status-failed";
    if (normalized.includes("cancel")) return "status-cancelled";
    if (normalized.includes("queue")) return "status-queued";
    return "status-pending";
  }

  function makeStatusPill(status) {
    const span = document.createElement("span");
    span.className = `status-pill ${statusClass(status)}`;
    span.textContent = status || "-";
    return span;
  }

  function render(snapshot) {
    const nodes = snapshot.nodes || [];
    const artifacts = snapshot.artifacts || [];
    const doneNodes = nodes.filter((node) => node.node_status === "DONE").length;
    const totalNodes = nodes.length;
    const progress = totalNodes ? Math.round((doneNodes / totalNodes) * 100) : 0;

    setText(el.jobSubtitle, snapshot.job && snapshot.job.job_id ? snapshot.job.job_id : state.jobId);
    setText(el.jobStatus, snapshot.job ? snapshot.job.job_status : "-");
    setText(el.nodesDone, `${doneNodes}/${totalNodes}`);
    setText(el.artifactTotal, artifacts.length);
    setText(el.serverTime, snapshot.server_time ? new Date(snapshot.server_time).toLocaleTimeString() : "-");
    el.progressBar.style.width = `${progress}%`;

    renderNodes(nodes);
    renderArtifacts(artifacts);
    renderEvents(snapshot.job_events || [], snapshot.node_events || []);
    renderFinal(snapshot.final_artifact);
  }

  function renderNodes(nodes) {
    el.nodesBody.replaceChildren();
    if (!nodes.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.textContent = "No nodes";
      row.appendChild(cell);
      el.nodesBody.appendChild(row);
      return;
    }
    for (const node of nodes) {
      const row = document.createElement("tr");
      const type = document.createElement("td");
      const level = document.createElement("td");
      const index = document.createElement("td");
      const status = document.createElement("td");
      const nodeId = document.createElement("td");
      type.textContent = node.node_type;
      level.textContent = node.level;
      index.textContent = node.node_index;
      status.appendChild(makeStatusPill(node.node_status));
      nodeId.textContent = node.node_id;
      nodeId.className = "node-id";
      row.append(type, level, index, status, nodeId);
      el.nodesBody.appendChild(row);
    }
  }

  function renderArtifacts(artifacts) {
    el.artifactsList.replaceChildren();
    if (!artifacts.length) {
      const empty = document.createElement("div");
      empty.className = "artifact-row";
      empty.textContent = "No artifacts";
      el.artifactsList.appendChild(empty);
      return;
    }
    for (const artifact of artifacts) {
      const row = document.createElement("div");
      const title = document.createElement("strong");
      const details = document.createElement("span");
      row.className = "artifact-row";
      title.textContent = `${artifact.artifact_type} L${artifact.level}`;
      details.textContent = `${artifact.node_id || "input"} ${String(artifact.content_hash || "").slice(0, 16)}`;
      row.append(title, details);
      el.artifactsList.appendChild(row);
    }
  }

  function renderEvents(jobEvents, nodeEvents) {
    el.eventsList.replaceChildren();
    const events = [
      ...jobEvents.map((event) => ({ ...event, scope: "JOB" })),
      ...nodeEvents.map((event) => ({ ...event, scope: "NODE" })),
    ].sort((a, b) => String(b.event_time || "").localeCompare(String(a.event_time || ""))).slice(0, 80);

    if (!events.length) {
      const empty = document.createElement("div");
      empty.className = "event-row";
      empty.textContent = "No events";
      el.eventsList.appendChild(empty);
      return;
    }

    for (const event of events) {
      const row = document.createElement("div");
      const title = document.createElement("strong");
      const details = document.createElement("span");
      row.className = "event-row";
      title.textContent = `${event.scope} ${event.event_type}`;
      details.textContent = `${event.status || ""} ${event.node_id || ""} ${event.actor || ""}`;
      row.append(title, details);
      el.eventsList.appendChild(row);
    }
  }

  function renderFinal(artifact) {
    if (!artifact || !artifact.content) {
      setText(el.finalSummary, "-");
      return;
    }
    try {
      const parsed = JSON.parse(artifact.content);
      const lines = [parsed.summary || artifact.content];
      if (Array.isArray(parsed.key_points) && parsed.key_points.length) {
        lines.push("", "Key points:");
        for (const point of parsed.key_points) {
          lines.push(`- ${point}`);
        }
      }
      if (Array.isArray(parsed.warnings) && parsed.warnings.length) {
        lines.push("", "Warnings:");
        for (const warning of parsed.warnings) {
          lines.push(`- ${warning}`);
        }
      }
      setText(el.finalSummary, lines.join("\n"));
    } catch {
      setText(el.finalSummary, artifact.content);
    }
  }

  el.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    persistDraft();
    el.createButton.disabled = true;
    try {
      const created = await createJobFromCurrentMode();
      setActiveJob(created.job_id);
    } catch (error) {
      setBadge("create error", "badge-bad");
      setText(el.finalSummary, error.message);
    } finally {
      el.createButton.disabled = false;
    }
  });

  async function createJobFromCurrentMode() {
    const title = el.title.value || null;
    const autoStart = el.autoStart.checked;
    if (state.inputMode === "text") {
      const inputText = el.inputText.value.trim();
      if (!inputText) {
        throw new Error("Input context is empty");
      }
      return requestJson("/summary-jobs", {
        method: "POST",
        body: JSON.stringify({
          title,
          input_text: inputText,
          metadata: { source: "summary_frontend_text" },
          auto_start: autoStart,
        }),
      });
    }

    if (state.inputMode === "upload") {
      const file = el.uploadFile.files && el.uploadFile.files[0];
      if (!file) {
        throw new Error("Choose a file to upload");
      }
      const formData = new FormData();
      formData.append("file", file);
      formData.append("title", title || file.name);
      formData.append("metadata", JSON.stringify({ source: "summary_frontend_upload" }));
      formData.append("auto_start", autoStart ? "true" : "false");
      formData.append("source_format", el.uploadFormat.value || "auto");
      if (el.uploadRawLineColumn.value.trim()) {
        formData.append("raw_line_column", el.uploadRawLineColumn.value.trim());
      }
      setBadge("uploading", "badge-warn");
      const created = await requestForm("/summary-jobs/upload", formData);
      loadUploads().catch(() => undefined);
      return created;
    }

    if (state.inputMode === "reuse") {
      const uploadId = el.uploadedFileSelect.value;
      if (!uploadId) {
        throw new Error("Choose a saved upload");
      }
      const payload = {
        upload_id: uploadId,
        title,
        metadata: { source: "summary_frontend_reuse" },
        auto_start: autoStart,
        source_format: el.reuseFormat.value || null,
        raw_line_column: el.reuseRawLineColumn.value.trim() || null,
      };
      return requestJson("/summary-jobs/from-upload", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    }

    if (state.inputMode === "query") {
      const query = el.queryText.value.trim();
      if (!query) {
        throw new Error("SQL query is empty");
      }
      return requestJson("/summary-jobs/clickhouse-query", {
        method: "POST",
        body: JSON.stringify({
          title,
          query,
          metadata: { source: "summary_frontend_query" },
          auto_start: autoStart,
          raw_line_column: el.queryRawLineColumn.value.trim() || null,
        }),
      });
    }

    throw new Error(`Unsupported input mode: ${state.inputMode}`);
  }

  for (const input of [
    el.title,
    el.inputText,
    el.uploadFormat,
    el.uploadRawLineColumn,
    el.reuseFormat,
    el.reuseRawLineColumn,
    el.queryText,
    el.queryRawLineColumn,
    el.autoStart,
  ]) {
    input.addEventListener("input", persistDraft);
    input.addEventListener("change", persistDraft);
  }

  for (const tab of el.modeTabs) {
    tab.addEventListener("click", () => setInputMode(tab.dataset.mode || "text"));
  }

  el.uploadedFileSelect.addEventListener("change", renderSelectedUploadDetails);
  el.refreshUploadsButton.addEventListener("click", () => {
    loadUploads().catch((error) => {
      setBadge("uploads error", "badge-bad");
      setText(el.finalSummary, error.message);
    });
  });

  el.reloadButton.addEventListener("click", () => {
    loadSnapshot().catch((error) => setText(el.finalSummary, error.message));
  });

  el.clearButton.addEventListener("click", () => {
    state.jobId = "";
    localStorage.removeItem(storage.jobId);
    if (state.websocket) {
      state.websocket.close();
    }
    stopPolling();
    renderEmpty();
  });

  el.pauseButton.addEventListener("click", () => {
    if (state.jobId) requestJson(`/summary-jobs/${state.jobId}/pause`, { method: "POST" }).catch(() => setBadge("pause error", "badge-bad"));
  });

  el.resumeButton.addEventListener("click", () => {
    if (state.jobId) requestJson(`/summary-jobs/${state.jobId}/resume`, { method: "POST" }).catch(() => setBadge("resume error", "badge-bad"));
  });

  el.cancelButton.addEventListener("click", () => {
    if (state.jobId) requestJson(`/summary-jobs/${state.jobId}/cancel`, { method: "POST" }).catch(() => setBadge("cancel error", "badge-bad"));
  });

  restoreDraft();
  setInputMode(state.inputMode);
  loadUploads().catch(() => undefined);
  if (state.jobId) {
    loadSnapshot().catch(() => renderEmpty());
    connectWebSocket();
  } else {
    renderEmpty();
  }
})();
