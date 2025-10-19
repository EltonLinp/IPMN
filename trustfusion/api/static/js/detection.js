const startBtn = document.getElementById("start-detection");
const stopBtn = document.getElementById("stop-detection");
const riskEl = document.getElementById("status-risk-level");
const confidenceEl = document.getElementById("status-confidence");
const alertsEl = document.getElementById("status-alerts");
const tableBody = document.getElementById("detections-body");

const MOCK_EVENTS = [
  {
    risk: "High",
    confidence: 0.91,
    highlights: ["Facial texture drift", "Lip-sync mismatch"],
  },
  {
    risk: "Medium",
    confidence: 0.68,
    highlights: ["Audio cadence anomaly"],
  },
  {
    risk: "Low",
    confidence: 0.32,
    highlights: ["Minimal artefacts"],
  },
  {
    risk: "Medium",
    confidence: 0.57,
    highlights: ["Blink frequency outlier"],
  },
  {
    risk: "High",
    confidence: 0.95,
    highlights: ["Identity embedding divergence", "Voice timbre shift"],
  },
];

let simulationTimer = null;
let eventIndex = 0;

function formatTimestamp() {
  return new Date().toLocaleTimeString();
}

function renderRow(event) {
  const row = document.createElement("tr");
  const highlights = Array.isArray(event.highlights) ? event.highlights.join(", ") : "N/A";
  row.innerHTML = `
    <td>${formatTimestamp()}</td>
    <td>${event.risk}</td>
    <td>${event.confidence.toFixed(2)}</td>
    <td>${highlights}</td>
  `;
  return row;
}

function updateStatus(event) {
  riskEl.textContent = event.risk;
  confidenceEl.textContent = event.confidence.toFixed(2);
  const alertsCount = Array.isArray(event.highlights) ? event.highlights.length : 0;
  alertsEl.textContent = `${alertsCount}`;
}

function clearTable() {
  tableBody.innerHTML = "";
}

function resetDashboard() {
  riskEl.textContent = "Idle";
  confidenceEl.textContent = "0.00";
  alertsEl.textContent = "0";
  tableBody.innerHTML = `<tr><td colspan="4">Simulator idle.</td></tr>`;
}

function emitNextEvent() {
  const event = MOCK_EVENTS[eventIndex % MOCK_EVENTS.length];
  eventIndex += 1;
  if (tableBody.children.length === 1 && tableBody.children[0].textContent === "Simulator idle.") {
    clearTable();
  }
  tableBody.prepend(renderRow(event));
  while (tableBody.children.length > 8) {
    tableBody.removeChild(tableBody.lastChild);
  }
  updateStatus(event);
}

function startSimulation() {
  if (simulationTimer) {
    return;
  }
  clearTable();
  emitNextEvent();
  simulationTimer = setInterval(emitNextEvent, 2000);
  startBtn.disabled = true;
  stopBtn.classList.remove("hidden");
  stopBtn.disabled = false;
  startBtn.textContent = "Running...";
}

function stopSimulation() {
  if (simulationTimer) {
    clearInterval(simulationTimer);
    simulationTimer = null;
  }
  eventIndex = 0;
  startBtn.disabled = false;
  startBtn.textContent = "Start Mock Detection";
  stopBtn.classList.add("hidden");
  resetDashboard();
}

if (startBtn) {
  startBtn.addEventListener("click", startSimulation);
}
if (stopBtn) {
  stopBtn.addEventListener("click", stopSimulation);
}

resetDashboard();
