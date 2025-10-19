const summarySelectors = {
  root: document.getElementById("dataset-root"),
  total: document.getElementById("dataset-total"),
  real: document.getElementById("dataset-real"),
  fake: document.getElementById("dataset-fake"),
  unknown: document.getElementById("dataset-unknown"),
};

const evaluationSection = document.getElementById("evaluation-result");
const evalTotalEl = document.getElementById("eval-total");
const evalPredFakeEl = document.getElementById("eval-pred-fake");
const evalPredRealEl = document.getElementById("eval-pred-real");
const evalGtFakeEl = document.getElementById("eval-gt-fake");
const evalGtRealEl = document.getElementById("eval-gt-real");
const evalGtUnknownEl = document.getElementById("eval-gt-unknown");
const evalAccuracyEl = document.getElementById("eval-accuracy");
const evalFakeRateEl = document.getElementById("eval-fake-rate");
const evalThresholdEl = document.getElementById("eval-threshold");
const evalLimitEl = document.getElementById("eval-limit");
const evalGroupLimitEl = document.getElementById("eval-group-limit");
const evalRunIdEl = document.getElementById("eval-run-id");
const evalModelVersionEl = document.getElementById("eval-model-version");
const evalAucEl = document.getElementById("eval-auc");
const evalBalancedAccEl = document.getElementById("eval-balanced-acc");
const evalFnrEl = document.getElementById("eval-fnr");
const evalGroupTableBody = document.querySelector("#eval-group-table tbody");
const evalFakeSourceBody = document.querySelector("#eval-fake-source-table tbody");
const evalRealSourceBody = document.querySelector("#eval-real-source-table tbody");
const evalTableBody = document.querySelector("#eval-table tbody");
const evalRiskHighEl = document.getElementById("eval-risk-high");
const evalRiskMediumEl = document.getElementById("eval-risk-medium");
const evalRiskLowEl = document.getElementById("eval-risk-low");
const paginationEl = document.getElementById("eval-pagination");
const pagePrevBtn = document.getElementById("page-prev");
const pageNextBtn = document.getElementById("page-next");
const pageInfoEl = document.getElementById("page-info");
const runEvaluationBtn = document.getElementById("run-evaluation");
const progressContainer = document.getElementById("evaluation-progress");
const progressFill = document.getElementById("evaluation-progress-fill");
const progressTextEl = document.getElementById("evaluation-progress-text");

const sampleSelect = document.getElementById("sample-select");
const sampleDetails = document.getElementById("sample-details");
const samplePathEl = document.getElementById("sample-path");
const sampleGroupEl = document.getElementById("sample-group");
const sampleProbEl = document.getElementById("sample-prob");
const sampleProbVideoEl = document.getElementById("sample-prob-video");
const sampleProbAudioEl = document.getElementById("sample-prob-audio");
const sampleProbSyncEl = document.getElementById("sample-prob-sync");
const samplePredEl = document.getElementById("sample-pred");
const sampleGtEl = document.getElementById("sample-gt");
const sampleRiskEl = document.getElementById("sample-risk");

const evalModalAccEls = {
  video: {
    real: document.getElementById("eval-modal-acc-video-real"),
    fake: document.getElementById("eval-modal-acc-video-fake"),
  },
  audio: {
    real: document.getElementById("eval-modal-acc-audio-real"),
    fake: document.getElementById("eval-modal-acc-audio-fake"),
  },
  sync: {
    real: document.getElementById("eval-modal-acc-sync-real"),
    fake: document.getElementById("eval-modal-acc-sync-fake"),
  },
};

const MOCK_DATA_URL = "/static/mock/mock.json";

let cachedSamples = [];
let currentPage = 1;
const PAGE_SIZE = 50;
let progressTimer = null;
let mockPayload = null;
let useMockData = false;
let mockReason = "";

async function ensureMockPayload() {
  if (mockPayload) {
    return mockPayload;
  }
  try {
    const response = await fetch(MOCK_DATA_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Mock data request failed with status ${response.status}`);
    }
    mockPayload = await response.json();
    return mockPayload;
  } catch (error) {
    console.error("Unable to load mock data:", error);
    return null;
  }
}

function updateMockBanner(reason) {
  const banner = document.getElementById("mock-banner");
  if (!banner) {
    return;
  }
  banner.classList.remove("hidden");
  const messageEl = document.getElementById("mock-banner-message");
  if (messageEl) {
    const fallbackMessage =
      messageEl.textContent?.trim() ||
      "Sample evaluation results are displayed while backend services are offline.";
    messageEl.textContent = reason || fallbackMessage;
  }
  const timestampEl = document.getElementById("mock-banner-updated");
  if (timestampEl && mockPayload?.last_updated) {
    const date = new Date(mockPayload.last_updated);
    if (!Number.isNaN(date.getTime())) {
      timestampEl.textContent = date.toLocaleString();
    } else {
      timestampEl.textContent = mockPayload.last_updated;
    }
  }
  document.body?.classList?.add("mock-mode");
}

function enableMockMode(reason) {
  useMockData = true;
  if (!mockReason) {
    mockReason = reason || "Backend offline. Showing mock evaluation preview.";
  }
  updateMockBanner(mockReason);
}

async function tryGetMockSection(key, reason) {
  const payload = await ensureMockPayload();
  if (payload && payload[key]) {
    enableMockMode(reason);
    return payload[key];
  }
  return null;
}

function updateSummaryView(data) {
  if (!data) {
    return;
  }
  summarySelectors.root.textContent = data.root_dir ?? "Unknown";
  summarySelectors.total.textContent = data.total ?? 0;
  summarySelectors.real.textContent = data.real ?? 0;
  summarySelectors.fake.textContent = data.fake ?? 0;
  summarySelectors.unknown.textContent = data.unknown ?? 0;
}

async function refreshSummary({ showError = true } = {}) {
  if (useMockData) {
    const fallback = mockPayload?.summary ?? (await tryGetMockSection("summary"));
    if (fallback) {
      updateSummaryView(fallback);
      return fallback;
    }
    if (showError) {
      alert("Refresh failed. Mock summary unavailable.");
    }
    return null;
  }
  try {
    const response = await fetch("/dataset/summary");
    if (!response.ok) {
      throw new Error(`Dataset summary responded with status ${response.status}`);
    }
    const data = await response.json();
    updateSummaryView(data);
    return data;
  } catch (error) {
    console.warn("Dataset summary unavailable, attempting mock fallback:", error);
    const fallback = await tryGetMockSection(
      "summary",
      "Dataset summary API unreachable. Displaying mock statistics."
    );
    if (fallback) {
      updateSummaryView(fallback);
      return fallback;
    }
    if (showError) {
      alert("Refresh failed. Please check server logs.");
    }
    return null;
  }
}

function formatLabel(label) {
  if (label === 1) return "Fake";
  if (label === 0) return "Real";
  return "Unknown";
}

function formatRisk(risk) {
  if (!risk) return "Unknown";
  return risk.charAt(0).toUpperCase() + risk.slice(1);
}

function formatGroup(label, folder) {
  const labelText = label ? label.charAt(0).toUpperCase() + label.slice(1) : "Unknown";
  if (!folder) {
    return labelText;
  }
  return `${labelText} / ${folder}`;
}

function formatPercentage(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatProbability(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toFixed(3);
}

function updateProgressView(snapshot) {
  if (!progressContainer || !progressFill || !progressTextEl) {
    return;
  }
  if (!snapshot) {
    progressContainer.classList.add("hidden");
    return;
  }

  const status = snapshot.status ?? "idle";
  const processed = snapshot.processed ?? 0;
  const total = snapshot.total ?? 0;
  const message = snapshot.message ?? "";

  let percentage = 0;
  if (total > 0) {
    percentage = Math.round((processed / total) * 100);
  } else if (status === "completed" || status === "error") {
    percentage = 100;
  }
  percentage = Math.max(0, Math.min(100, percentage));

  progressFill.style.width = `${percentage}%`;
  progressFill.classList.toggle("complete", status === "completed");
  progressFill.classList.toggle("error", status === "error");

  const textParts = [];
  if (message) {
    textParts.push(message);
  }
  if (status === "running" && total > 0) {
    textParts.push(`${percentage}%`);
  }
  if ((status === "completed" || status === "error") && !message) {
    textParts.push(status === "completed" ? "Evaluation complete." : "Evaluation failed.");
  }
  progressTextEl.textContent = textParts.join(" | ") || "Waiting to start...";

  if (status === "idle") {
    progressContainer.classList.add("hidden");
  } else {
    progressContainer.classList.remove("hidden");
  }

  if (runEvaluationBtn) {
    if (status === "running") {
      runEvaluationBtn.disabled = true;
      runEvaluationBtn.textContent = "Evaluating...";
    } else if (status === "completed") {
      runEvaluationBtn.disabled = false;
      runEvaluationBtn.textContent = "Re-run Evaluation";
    } else if (status === "error" || status === "idle") {
      runEvaluationBtn.disabled = false;
      runEvaluationBtn.textContent = "Start Evaluation";
    }
  }
}

async function fetchProgressSnapshot() {
  if (!progressContainer) {
    return null;
  }
  if (useMockData) {
    const progress =
      mockPayload?.progress ?? (await tryGetMockSection("progress", "Using mock progress data."));
    if (progress) {
      updateProgressView(progress);
      return progress;
    }
    return null;
  }
  try {
    const response = await fetch("/evaluation/progress");
    if (!response.ok) {
      throw new Error("Failed to fetch evaluation progress");
    }
    const data = await response.json();
    updateProgressView(data);
    return data;
  } catch (error) {
    console.warn("Unable to load evaluation progress, attempting mock fallback:", error);
    const fallback = await tryGetMockSection(
      "progress",
      "Evaluation progress API unreachable. Showing mock progress."
    );
    if (fallback) {
      updateProgressView(fallback);
      return fallback;
    }
    return null;
  }
}

function stopProgressPolling() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function startProgressPolling() {
  if (!progressContainer) {
    return;
  }
  if (useMockData) {
    fetchProgressSnapshot();
    return;
  }
  if (progressTimer) {
    return;
  }
  const handleSnapshot = (snapshot) => {
    if (!snapshot) {
      return;
    }
    if (snapshot.status === "completed" || snapshot.status === "error") {
      stopProgressPolling();
    }
  };
  progressContainer.classList.remove("hidden");
  fetchProgressSnapshot().then(handleSnapshot);
  progressTimer = setInterval(() => {
    fetchProgressSnapshot().then(handleSnapshot);
  }, 1000);
}

function renderTablePage(page = currentPage) {
  if (!Array.isArray(cachedSamples)) {
    return;
  }
  if (!paginationEl || !pageInfoEl || !pagePrevBtn || !pageNextBtn) {
    return;
  }
  if (cachedSamples.length === 0) {
    evalTableBody.innerHTML = "";
    paginationEl.classList.add("hidden");
    pageInfoEl.textContent = "Page 0 / 0";
    return;
  }

  const totalPages = Math.max(1, Math.ceil(cachedSamples.length / PAGE_SIZE));
  currentPage = Math.min(Math.max(page, 1), totalPages);
  const start = (currentPage - 1) * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, cachedSamples.length);

  evalTableBody.innerHTML = "";
  for (let i = start; i < end; i += 1) {
    const item = cachedSamples[i];
    const videoName = item.video_path ? item.video_path.split(/[/\\]/).pop() : "Unknown";
    const probVideo = formatProbability(item.prob_video ?? item.prob_fake ?? 0);
    const probAudio = formatProbability(item.prob_audio ?? item.prob_fake ?? 0);
    const probSync = formatProbability(item.prob_sync ?? item.prob_fake ?? 0);
    const probCombined = formatProbability(item.prob_fake ?? 0);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td title="${item.video_path}">${videoName}</td>
      <td>${formatGroup(item.group_label, item.group_folder)}</td>
      <td>${probVideo}</td>
      <td>${probAudio}</td>
      <td>${probSync}</td>
      <td>${probCombined}</td>
      <td>${formatLabel(item.pred_label)}</td>
      <td>${formatLabel(item.true_label)}</td>
      <td>${formatRisk(item.risk_level)}</td>
    `;
    evalTableBody.appendChild(tr);
  }

  pageInfoEl.textContent = `Page ${currentPage} / ${totalPages}`;
  pagePrevBtn.disabled = currentPage <= 1;
  pageNextBtn.disabled = currentPage >= totalPages;
  paginationEl.classList.toggle("hidden", totalPages <= 1);
}

function populateSampleDropdown(details) {
  cachedSamples = details ?? [];
  sampleSelect.innerHTML = '<option value="">Select a sample...</option>';
  cachedSamples.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = index.toString();
    const name =
      (item.video_path && item.video_path.split(/[/\\]/).pop()) ||
      `Sample ${index + 1}`;
    option.textContent = `${name} [${formatGroup(item.group_label, item.group_folder)}]`;
    sampleSelect.appendChild(option);
  });
  sampleDetails.classList.add("hidden");
}

function showSampleDetail(index) {
  const item = cachedSamples[index];
  if (!item) {
    sampleDetails.classList.add("hidden");
    return;
  }
  samplePathEl.textContent = item.video_path;
  if (sampleGroupEl) {
    sampleGroupEl.textContent = formatGroup(item.group_label, item.group_folder);
  }
  const probCombined = formatProbability(item.prob_fake ?? 0);
  const probVideo = formatProbability(item.prob_video ?? item.prob_fake ?? 0);
  const probAudio = formatProbability(item.prob_audio ?? item.prob_fake ?? 0);
  const probSync = formatProbability(item.prob_sync ?? item.prob_fake ?? 0);
  sampleProbEl.textContent = probCombined;
  if (sampleProbVideoEl) sampleProbVideoEl.textContent = probVideo;
  if (sampleProbAudioEl) sampleProbAudioEl.textContent = probAudio;
  if (sampleProbSyncEl) sampleProbSyncEl.textContent = probSync;
  samplePredEl.textContent = formatLabel(item.pred_label);
  sampleGtEl.textContent = formatLabel(item.true_label);
  sampleRiskEl.textContent = formatRisk(item.risk_level);
  sampleDetails.classList.remove("hidden");
}

function updateEvaluationView(data) {
  if (evalRunIdEl) {
    evalRunIdEl.textContent = data.run_id ?? "N/A";
  }
  if (evalModelVersionEl) {
    evalModelVersionEl.textContent = data.model_version ?? "N/A";
  }
  evalTotalEl.textContent = data.total ?? 0;
  evalPredFakeEl.textContent = data.pred_fake ?? 0;
  evalPredRealEl.textContent = data.pred_real ?? 0;
  evalGtFakeEl.textContent = data.gt_fake ?? 0;
  evalGtRealEl.textContent = data.gt_real ?? 0;
  evalGtUnknownEl.textContent = data.gt_unknown ?? 0;
  evalRiskHighEl.textContent = data.risk_high ?? 0;
  evalRiskMediumEl.textContent = data.risk_medium ?? 0;
  evalRiskLowEl.textContent = data.risk_low ?? 0;
  evalAccuracyEl.textContent = formatPercentage(data.accuracy);
  evalFakeRateEl.textContent = formatPercentage(data.fake_rate);
  evalThresholdEl.textContent = data.threshold !== undefined ? data.threshold.toFixed(2) : "N/A";
  if (evalAucEl) {
    const aucValue = Number(data.auc);
    evalAucEl.textContent =
      data.auc !== undefined && data.auc !== null && !Number.isNaN(aucValue)
        ? aucValue.toFixed(3)
        : "N/A";
  }
  if (evalBalancedAccEl) {
    evalBalancedAccEl.textContent = formatPercentage(data.balanced_accuracy);
  }
  if (evalFnrEl) {
    evalFnrEl.textContent = formatPercentage(data.fnr);
  }
  if (evalLimitEl) {
    const datasetTotal = data.dataset_total ?? data.total ?? 0;
    const totalText =
      datasetTotal && datasetTotal !== data.total
        ? `${data.total ?? 0} / ${datasetTotal}`
        : `${data.total ?? 0}`;
    evalLimitEl.textContent = totalText;
  }
  if (evalGroupLimitEl) {
    const limit = data.per_group_limit;
    if (typeof limit === "number" && limit > 0) {
      evalGroupLimitEl.textContent = `${limit}`;
    } else if (limit === 0) {
      evalGroupLimitEl.textContent = "0";
    } else {
      evalGroupLimitEl.textContent = "Unlimited";
    }
  }
  if (evalGroupTableBody) {
    evalGroupTableBody.innerHTML = "";
    if (Array.isArray(data.group_breakdown) && data.group_breakdown.length > 0) {
      data.group_breakdown.forEach((group) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${formatLabel(group.label === "real" ? 0 : group.label === "fake" ? 1 : null)}</td>
          <td>${group.folder ?? "Unknown"}</td>
          <td>${group.selected ?? 0}</td>
          <td>${group.available ?? 0}</td>
        `;
        evalGroupTableBody.appendChild(tr);
      });
    } else {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="4">No folders sampled.</td>`;
      evalGroupTableBody.appendChild(tr);
    }
  }
  if (evalFakeSourceBody) {
    evalFakeSourceBody.innerHTML = "";
    if (Array.isArray(data.fake_sources) && data.fake_sources.length > 0) {
      data.fake_sources.forEach((source) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${source.source ?? "Unknown"}</td>
          <td>${source.count ?? 0}</td>
          <td>${formatProbability(source.auc)}</td>
          <td>${formatPercentage(source.tpr)}</td>
          <td>${formatPercentage(source.fnr)}</td>
          <td>${formatPercentage(source.fpr)}</td>
        `;
        evalFakeSourceBody.appendChild(tr);
      });
    } else {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="6">No fake source metrics.</td>`;
      evalFakeSourceBody.appendChild(tr);
    }
  }
  if (evalRealSourceBody) {
    evalRealSourceBody.innerHTML = "";
    if (Array.isArray(data.real_sources) && data.real_sources.length > 0) {
      data.real_sources.forEach((source) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${source.source ?? "Unknown"}</td>
          <td>${source.count ?? 0}</td>
          <td>${formatPercentage(source.fpr)}</td>
          <td>${formatPercentage(source.tnr)}</td>
        `;
        evalRealSourceBody.appendChild(tr);
      });
    } else {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="4">No real source metrics.</td>`;
      evalRealSourceBody.appendChild(tr);
    }
  }
  if (data.modal_accuracy && evalModalAccEls) {
    ["video", "audio", "sync"].forEach((modal) => {
      const stats = data.modal_accuracy[modal] || {};
      const realEl = evalModalAccEls[modal]?.real;
      const fakeEl = evalModalAccEls[modal]?.fake;
      if (realEl) {
        const value = stats.real;
        realEl.textContent =
          value === null || value === undefined ? "N/A" : `${(value * 100).toFixed(2)}%`;
      }
      if (fakeEl) {
        const value = stats.fake;
        fakeEl.textContent =
          value === null || value === undefined ? "N/A" : `${(value * 100).toFixed(2)}%`;
      }
    });
  }

  populateSampleDropdown(data.details);
  currentPage = 1;
  renderTablePage(currentPage);
  evaluationSection.classList.remove("hidden");
}

async function fetchEvaluationSummary() {
  if (useMockData) {
    const mockEval = mockPayload?.evaluation ?? (await tryGetMockSection("evaluation"));
    if (mockEval) {
      updateEvaluationView(mockEval);
      return mockEval;
    }
    return null;
  }
  try {
    const response = await fetch("/evaluation/summary");
    if (!response.ok) {
      throw new Error(`Evaluation summary responded with status ${response.status}`);
    }
    const data = await response.json();
    if (data.message) {
      return null;
    }
    updateEvaluationView(data);
    return data;
  } catch (error) {
    console.warn("Evaluation summary unavailable, attempting mock fallback:", error);
    const fallback = await tryGetMockSection(
      "evaluation",
      "Evaluation API unreachable. Showing mock evaluation output."
    );
    if (fallback) {
      updateEvaluationView(fallback);
      return fallback;
    }
    return null;
  }
}

async function runEvaluation() {
  if (!runEvaluationBtn) {
    return;
  }
  const button = runEvaluationBtn;

  if (useMockData) {
    const payload = mockPayload ?? (await ensureMockPayload());
    const evaluation = payload?.evaluation;
    if (!evaluation) {
      alert("Mock evaluation data unavailable.");
      return;
    }
    button.disabled = true;
    button.textContent = "Evaluating...";
    const cachedProgress = payload.progress || {};
    const total =
      cachedProgress.total ?? evaluation.total ?? evaluation.dataset_total ?? cachedSamples.length;
    const runningSnapshot = {
      status: "running",
      processed: 0,
      total: total ?? 0,
      message: "Simulating evaluation...",
    };
    updateProgressView(runningSnapshot);
    progressContainer?.classList?.remove("hidden");
    await new Promise((resolve) => setTimeout(resolve, 600));
    const finalSnapshot = {
      status: "completed",
      processed: cachedProgress.processed ?? total ?? evaluation.total ?? 0,
      total: total ?? cachedProgress.total ?? evaluation.total ?? 0,
      message: cachedProgress.message ?? "Mock evaluation complete",
    };
    updateProgressView(finalSnapshot);
    updateEvaluationView(evaluation);
    stopProgressPolling();
    button.disabled = false;
    button.textContent = "Re-run Evaluation";
    return;
  }

  let evaluationSucceeded = false;
  try {
    button.disabled = true;
    button.textContent = "Evaluating...";
    startProgressPolling();
    const response = await fetch("/evaluation/run", { method: "POST" });
    if (!response.ok) {
      let errorMessage = "Evaluation call failed";
      try {
        const payload = await response.json();
        errorMessage = payload.detail ?? payload.message ?? errorMessage;
      } catch (parseError) {
        const fallback = await response.text();
        if (fallback) {
          errorMessage = fallback;
        }
      }
      throw new Error(errorMessage);
    }
    const data = await response.json();
    updateEvaluationView(data);
    evaluationSucceeded = true;
  } catch (error) {
    console.error(error);
    alert(error.message || "Evaluation failed. See server logs for details.");
  } finally {
    await fetchProgressSnapshot();
    stopProgressPolling();
    button.disabled = false;
    button.textContent = evaluationSucceeded ? "Re-run Evaluation" : "Start Evaluation";
  }
}

const refreshSummaryBtn = document.getElementById("refresh-summary");
if (refreshSummaryBtn) {
  refreshSummaryBtn.addEventListener("click", () => refreshSummary());
}
if (runEvaluationBtn) {
  runEvaluationBtn.addEventListener("click", runEvaluation);
}
if (pagePrevBtn && pageNextBtn) {
  pagePrevBtn.addEventListener("click", () => renderTablePage(currentPage - 1));
  pageNextBtn.addEventListener("click", () => renderTablePage(currentPage + 1));
}
sampleSelect.addEventListener("change", (event) => {
  const value = event.target.value;
  if (value === "") {
    sampleDetails.classList.add("hidden");
    return;
  }
  showSampleDetail(Number(value));
});

fetchProgressSnapshot().then((snapshot) => {
  if (snapshot && snapshot.status === "running") {
    startProgressPolling();
  }
});

refreshSummary();
fetchEvaluationSummary();
