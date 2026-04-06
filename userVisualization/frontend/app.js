const statusPill = document.getElementById("status-pill");
const summaryChip = document.getElementById("summary-chip");
const errorMessage = document.getElementById("error-message");
const loadingBackdrop = document.getElementById("loading-backdrop");
const resultGrid = document.getElementById("result-grid");
const idResultGrid = document.getElementById("id-result-grid");
const idSummaryChip = document.getElementById("id-summary-chip");
const fileInput = document.getElementById("video-input");
const chooseBtn = document.getElementById("choose-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const uploadBox = document.getElementById("upload-box");
const fileChip = document.getElementById("file-chip");
const idPhotoInput = document.getElementById("id-photo-input");
const idPhotoBtn = document.getElementById("id-photo-btn");
const idPhotoChip = document.getElementById("id-photo-chip");
const idPhotoPreview = document.getElementById("id-photo-preview");
const idFacePreview = document.getElementById("id-face-preview");
const selfieInput = document.getElementById("selfieInput");
const selfieBtn = document.getElementById("selfieBtn");
const selfieChip = document.getElementById("selfieChip");
const selfiePreview = document.getElementById("selfiePreview");
const idMatchReal = document.getElementById("id-match-real");
const idMatchFake = document.getElementById("id-match-fake");
const idMatchRealLabel = document.getElementById("id-match-real-label");
const idMatchFakeLabel = document.getElementById("id-match-fake-label");
const fusionCard = document.getElementById("fusion-card");
const gaugeLabel = document.getElementById("gauge-label");
const gaugeScore = document.getElementById("gauge-score");
const contrastBars = document.getElementById("contrast-bars");
const nameInput = document.getElementById("user-name");
const phoneInput = document.getElementById("user-phone");
const cameraPreview = document.getElementById("camera-preview");
const cameraStatus = document.getElementById("camera-status");
const cameraTimer = document.getElementById("camera-timer");
const cameraStartBtn = document.getElementById("camera-start-btn");
const cameraStopBtn = document.getElementById("camera-stop-btn");
const cameraResetBtn = document.getElementById("camera-reset-btn");
const ekycEvaluateBtn = document.getElementById("ekycEvaluateBtn");
const ekycVideoPlayer = document.getElementById("ekycVideoPlayer");
const ekycIdFaceImg = document.getElementById("ekycIdFaceImg");
const ekycSelfieFaceImg = document.getElementById("ekycSelfieFaceImg");
const ekycVideoFrameImg = document.getElementById("ekycVideoFrameImg");
const ekycIdFaceHint = document.getElementById("ekycIdFaceHint");
const ekycSelfieFaceHint = document.getElementById("ekycSelfieFaceHint");
const ekycVideoFrameHint = document.getElementById("ekycVideoFrameHint");
const ekycVideoHint = document.getElementById("ekycVideoHint");
const idDocumentStatus = document.getElementById("idDocumentStatus");
const idDocumentRiskLevel = document.getElementById("idDocumentRiskLevel");
const idDocumentSummary = document.getElementById("idDocumentSummary");
const idDocumentReupload = document.getElementById("idDocumentReupload");
const idDocumentIssues = document.getElementById("idDocumentIssues");
const scoreDeepfake = document.getElementById("scoreDeepfake");
const scoreDeepfakeMeta = document.getElementById("scoreDeepfakeMeta");
const scoreIdSelfie = document.getElementById("scoreIdSelfie");
const scoreIdVideo = document.getElementById("scoreIdVideo");
const scoreIdSelfieMeta = document.getElementById("scoreIdSelfieMeta");
const scoreIdVideoMeta = document.getElementById("scoreIdVideoMeta");
const scoreDeepfakeBar = document.getElementById("scoreDeepfakeBar");
const scoreIdSelfieBar = document.getElementById("scoreIdSelfieBar");
const scoreIdVideoBar = document.getElementById("scoreIdVideoBar");
const scoreFusionThreshold = document.getElementById("scoreFusionThreshold");
const scoreFusionStrategy = document.getElementById("scoreFusionStrategy");
const scoreFusionWeights = document.getElementById("scoreFusionWeights");
const ekycDecision = document.getElementById("ekycDecision");
const ekycDecisionChip = document.getElementById("ekycDecisionChip");
const ekycDecisionSummary = document.getElementById("ekycDecisionSummary");
const ekycDecisionMeta = document.getElementById("ekycDecisionMeta");
const ekycReasons = document.getElementById("ekycReasons");
const ekycError = document.getElementById("ekycError");
const ekycQualityFlags = document.getElementById("ekycQualityFlags");
const ekycSyncMismatch = document.getElementById("ekycSyncMismatch");
const ekycSyncInterpolated = document.getElementById("ekycSyncInterpolated");
const ekycSyncLengthBad = document.getElementById("ekycSyncLengthBad");
const ekycAudioFake = document.getElementById("ekycAudioFake");
const ekycVideoFake = document.getElementById("ekycVideoFake");
const ekycSyncFake = document.getElementById("ekycSyncFake");

let selectedFile = null;
let selectedIdPhoto = null;
let selectedSelfie = null;
let idPhotoPreviewUrl = null;
let selfiePreviewUrl = null;
let fusionGauge = null;
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordTimerId = null;
let recordStartAt = null;
let ekycVideoUrl = null;
const MAX_RECORD_SECONDS = 15;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function updateAnalyzeReadyState() {
  const ready = Boolean(selectedFile && selectedIdPhoto && selectedSelfie);
  analyzeBtn.disabled = !ready;
  statusPill.textContent = ready ? "Ready to evaluate" : "Waiting for upload";
}

function updateEkycReadyState() {
  return;
}

function setSelectedFile(file) {
  if (!file) {
    selectedFile = null;
    fileChip.hidden = true;
    resetIdPanels();
    updateAnalyzeReadyState();
    updateEkycReadyState();
    return;
  }
  selectedFile = file;
  fileChip.textContent = `${file.name} - ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
  fileChip.hidden = false;
  resetIdPanels();
  updateAnalyzeReadyState();
  updateEkycReadyState();
}

function setIdPhotoFile(file) {
  if (!file) {
    selectedIdPhoto = null;
    if (idPhotoChip) {
      idPhotoChip.hidden = true;
      idPhotoChip.textContent = "";
    }
    if (idPhotoPreviewUrl) {
      URL.revokeObjectURL(idPhotoPreviewUrl);
      idPhotoPreviewUrl = null;
    }
    if (idPhotoPreview) {
      idPhotoPreview.removeAttribute("src");
    }
    if (idFacePreview) {
      idFacePreview.removeAttribute("src");
    }
    resetIdPanels();
    updateAnalyzeReadyState();
    updateEkycReadyState();
    return;
  }
  selectedIdPhoto = file;
  if (idPhotoChip) {
    idPhotoChip.textContent = `${file.name} - ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    idPhotoChip.hidden = false;
  }
  if (idPhotoPreviewUrl) {
    URL.revokeObjectURL(idPhotoPreviewUrl);
  }
  idPhotoPreviewUrl = URL.createObjectURL(file);
  if (idPhotoPreview) {
    idPhotoPreview.src = idPhotoPreviewUrl;
  }
  if (idFacePreview) {
    idFacePreview.removeAttribute("src");
  }
  resetIdPanels();
  updateAnalyzeReadyState();
  updateEkycReadyState();
}

function setSelfieFile(file) {
  if (!file) {
    selectedSelfie = null;
    if (selfieChip) {
      selfieChip.hidden = true;
      selfieChip.textContent = "";
    }
    if (selfiePreviewUrl) {
      URL.revokeObjectURL(selfiePreviewUrl);
      selfiePreviewUrl = null;
    }
    if (selfiePreview) {
      selfiePreview.removeAttribute("src");
      selfiePreview.hidden = true;
    }
    updateEkycReadyState();
    return;
  }
  selectedSelfie = file;
  if (selfieChip) {
    selfieChip.textContent = `${file.name} - ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    selfieChip.hidden = false;
  }
  if (selfiePreviewUrl) {
    URL.revokeObjectURL(selfiePreviewUrl);
  }
  selfiePreviewUrl = URL.createObjectURL(file);
  if (selfiePreview) {
    selfiePreview.src = selfiePreviewUrl;
    selfiePreview.hidden = false;
  }
  updateEkycReadyState();
}

function setCameraStatus(message) {
  if (cameraStatus) {
    cameraStatus.textContent = message;
  }
}

function updateCameraTimer() {
  if (!cameraTimer) {
    return;
  }
  if (!recordStartAt) {
    cameraTimer.textContent = "00:00";
    return;
  }
  const elapsed = Math.min((Date.now() - recordStartAt) / 1000, MAX_RECORD_SECONDS);
  const minutes = Math.floor(elapsed / 60);
  const seconds = Math.floor(elapsed % 60);
  cameraTimer.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

async function ensureCameraStream() {
  if (mediaStream) {
    return mediaStream;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access is not supported in this browser.");
  }
  mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  if (cameraPreview) {
    cameraPreview.srcObject = mediaStream;
    cameraPreview.muted = true;
    try {
      await cameraPreview.play();
    } catch (err) {
      console.warn("Unable to autoplay camera preview.", err);
    }
  }
  return mediaStream;
}

function stopCameraStream() {
  if (!mediaStream) {
    return;
  }
  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  if (cameraPreview) {
    cameraPreview.srcObject = null;
  }
}

function setCameraControls({ recording, hasClip }) {
  if (cameraStartBtn) {
    cameraStartBtn.disabled = recording;
  }
  if (cameraStopBtn) {
    cameraStopBtn.disabled = !recording;
  }
  if (cameraResetBtn) {
    cameraResetBtn.disabled = recording || !hasClip;
  }
}

async function startRecording() {
  try {
    clearError();
    setCameraStatus("Requesting camera...");
    const stream = await ensureCameraStream();
    recordedChunks = [];
    let options = { mimeType: "video/webm;codecs=vp8,opus" };
    if (options.mimeType && !MediaRecorder.isTypeSupported(options.mimeType)) {
      options = {};
    }
    mediaRecorder = new MediaRecorder(stream, options);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };
    mediaRecorder.onstop = () => {
      const blobType = mediaRecorder?.mimeType || "video/webm";
      const blob = new Blob(recordedChunks, { type: blobType });
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const file = new File([blob], `webcam-${timestamp}.webm`, { type: blobType });
      setSelectedFile(file);
      setCameraStatus("Recording saved");
      setCameraControls({ recording: false, hasClip: true });
      updateCameraTimer();
    };
    mediaRecorder.start(200);
    recordStartAt = Date.now();
    updateCameraTimer();
    recordTimerId = window.setInterval(() => {
      updateCameraTimer();
      if (recordStartAt && Date.now() - recordStartAt >= MAX_RECORD_SECONDS * 1000) {
        stopRecording();
      }
    }, 200);
    setCameraStatus("Recording...");
    setCameraControls({ recording: true, hasClip: false });
  } catch (err) {
    console.error(err);
    showError(err.message || "Unable to access camera.");
    setCameraStatus("Camera error");
  }
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") {
    return;
  }
  mediaRecorder.stop();
  mediaRecorder = null;
  if (recordTimerId) {
    window.clearInterval(recordTimerId);
    recordTimerId = null;
  }
  recordStartAt = null;
}

function resetRecording() {
  recordedChunks = [];
  setSelectedFile(null);
  stopRecording();
  stopCameraStream();
  setCameraStatus("Camera idle");
  setCameraControls({ recording: false, hasClip: false });
  updateCameraTimer();
}

async function uploadVideo(file) {
  if (!file) {
    showError("Please choose a video file first.");
    return;
  }
  if (!selectedIdPhoto) {
    showError("Please upload an ID photo first.");
    return;
  }
  const nameValue = nameInput?.value?.trim() ?? "";
  const phoneValue = phoneInput?.value?.trim() ?? "";
  if (!nameValue || !phoneValue) {
    showError("Please enter your name and phone number.");
    return;
  }
  clearError();
  statusPill.textContent = "Uploading...";
  loadingBackdrop.hidden = false;
  analyzeBtn.disabled = true;
  chooseBtn.disabled = true;
  try {
    const formData = new FormData();
    formData.append("video", file, file.name || "upload.mp4");
    formData.append("id_photo", selectedIdPhoto, selectedIdPhoto.name || "id_photo.jpg");
    formData.append("user_name", nameValue);
    formData.append("user_phone", phoneValue);
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.detail || "Inference failed, please try again.");
    }
    updateIdPanels(payload.id_face, payload.id_match);
    updateResults(payload.result);
    statusPill.textContent = "Analysis complete";
  } catch (err) {
    console.error(err);
    showError(err.message || "Upload failed, please retry.");
    statusPill.textContent = "Error";
  } finally {
    loadingBackdrop.hidden = true;
    updateAnalyzeReadyState();
    chooseBtn.disabled = false;
  }
}

function showEkycError(message) {
  if (!ekycError) {
    return;
  }
  ekycError.textContent = message;
  ekycError.hidden = false;
}

function clearEkycError() {
  if (!ekycError) {
    return;
  }
  ekycError.hidden = true;
  ekycError.textContent = "";
}

async function uploadEvaluate() {
  if (!selectedFile) {
    showEkycError("Please choose a video file first.");
    return;
  }
  if (!selectedIdPhoto) {
    showEkycError("Please upload an ID photo first.");
    return;
  }
  if (!selectedSelfie) {
    showEkycError("Please upload a selfie photo first.");
    return;
  }
  clearEkycError();
  statusPill.textContent = "Evaluating eKYC...";
  loadingBackdrop.hidden = false;
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
  }
  try {
    const formData = new FormData();
    formData.append("video", selectedFile, selectedFile.name || "upload.mp4");
    formData.append("id_image", selectedIdPhoto, selectedIdPhoto.name || "id_image.jpg");
    formData.append("selfie_image", selectedSelfie, selectedSelfie.name || "selfie.jpg");
    const response = await fetch("/api/ekyc/evaluate", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const code = payload?.error_code || "error";
      const msg = payload?.message || "Evaluation failed.";
      showEkycError(`${code}: ${msg}`);
      statusPill.textContent = "Error";
      return;
    }
    updateEkycResults(payload);
    statusPill.textContent = "eKYC complete";
  } catch (err) {
    console.error(err);
    showEkycError(err.message || "Evaluation failed.");
    statusPill.textContent = "Error";
  } finally {
    loadingBackdrop.hidden = true;
    updateAnalyzeReadyState();
  }
}

function updateResults(result) {
  if (!result) {
    return;
  }
  updateCard("final", result.final);
  updateCard("audio", result.audio);
  updateCard("video", result.video);
  updateCard("sync", result.sync);
  updateFusionGauge(result.final);
  updateContrastBars(result);
  const finalLabel = result.final?.label === "Fake" ? "Suspicious (fake)" : "Likely genuine";
  const fakeProb = clamp01(result.final?.fake ?? 0);
  const confidence = Math.max(result.final?.real ?? 0, result.final?.fake ?? 0);
  if (summaryChip) {
    summaryChip.textContent = `${finalLabel} - Fake ${(fakeProb * 100).toFixed(1)}% (Confidence ${(confidence * 100).toFixed(1)}%)`;
  }
}

function updateEkycResults(payload) {
  if (!payload) {
    return;
  }
  const artifacts = payload.artifacts || {};
  const documentCheck = payload.document_check || {};
  const deepfakeScore = payload.deepfake?.score;
  const deepfake = payload.deepfake || {};
  const details = deepfake.details || {};
  const calibration = deepfake.calibration || {};
  const quality = deepfake.quality || {};
  const syncQuality = quality.sync || {};
  const idSelfie = payload.match?.id_selfie;
  const idVideo = payload.match?.id_video;
  const matchThresholds = payload.match?.thresholds || {};
  const idSelfiePassThreshold =
    typeof matchThresholds.id_selfie_pass === "number" ? matchThresholds.id_selfie_pass : 0.6;
  const idVideoRejectThreshold =
    typeof matchThresholds.id_video_reject === "number" ? matchThresholds.id_video_reject : 0.458;
  const idVideoPassThreshold =
    typeof matchThresholds.id_video_pass === "number" ? matchThresholds.id_video_pass : 0.78;
  const decision = payload.fusion?.decision || "--";
  const reasons = Array.isArray(payload.fusion?.reason) ? payload.fusion.reason : [];
  const explanation = payload.fusion?.explanation || {};

  if (ekycDecision) {
    ekycDecision.textContent = decision;
  }
  if (ekycDecisionChip) {
    ekycDecisionChip.textContent = decision;
  }
  renderIdDocumentCheck(documentCheck);
  renderDecisionExplanation(explanation, decision, reasons);

  if (ekycReasons) {
    ekycReasons.innerHTML = "";
    const items = reasons.length ? reasons : ["No reasons provided."];
    items.forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      ekycReasons.appendChild(li);
    });
  }

  const deepfakeValue = typeof deepfakeScore === "number" ? deepfakeScore : null;
  setScorePercent(scoreDeepfake, scoreDeepfakeBar, deepfakeValue, "fake");
  renderFusionCalibration(calibration, deepfakeValue);
  updateExplainability({
    qualityFlags: deepfake.quality_flags,
    syncQuality,
    branchDetails: details,
  });

  const idSelfieOk = idSelfie?.ok === true;
  const idSelfieProb = idSelfieOk && typeof idSelfie?.prob === "number" ? idSelfie.prob : null;
  setMatchProbScore(scoreIdSelfie, scoreIdSelfieBar, idSelfieProb, { passThreshold: idSelfiePassThreshold });
  setMatchThresholdMeta(scoreIdSelfieMeta, { passThreshold: idSelfiePassThreshold });

  const idVideoOk = idVideo?.ok === true;
  const idVideoProb = idVideoOk && typeof idVideo?.prob === "number" ? idVideo.prob : null;
  setMatchProbScore(scoreIdVideo, scoreIdVideoBar, idVideoProb, {
    rejectThreshold: idVideoRejectThreshold,
    passThreshold: idVideoPassThreshold,
  });
  setMatchThresholdMeta(scoreIdVideoMeta, {
    rejectThreshold: idVideoRejectThreshold,
    passThreshold: idVideoPassThreshold,
  });

  setImage(ekycIdFaceImg, ekycIdFaceHint, artifacts?.id_face_path, "ID face not available");
  setImage(ekycSelfieFaceImg, ekycSelfieFaceHint, artifacts?.selfie_face_path, "Selfie face not available");
  setImage(ekycVideoFrameImg, ekycVideoFrameHint, artifacts?.video_best_frame_path, "Video frame not available");

  if (selectedFile && ekycVideoPlayer) {
    if (ekycVideoUrl) {
      URL.revokeObjectURL(ekycVideoUrl);
    }
    ekycVideoUrl = URL.createObjectURL(selectedFile);
    ekycVideoPlayer.src = ekycVideoUrl;
    ekycVideoPlayer.load();
    if (ekycVideoHint) {
      ekycVideoHint.textContent = "";
    }
  }
}

function renderIdDocumentCheck(documentCheck) {
  const status = normalizeDocumentCheckStatus(documentCheck?.status);
  const riskLevel = documentCheck?.risk_level ? String(documentCheck.risk_level) : "--";
  const summary =
    typeof documentCheck?.summary === "string" && documentCheck.summary.trim().length > 0
      ? documentCheck.summary
      : "No document screening result.";
  const userMessage =
    typeof documentCheck?.user_message === "string" && documentCheck.user_message.trim().length > 0
      ? documentCheck.user_message
      : "";
  const issues = Array.isArray(documentCheck?.issues) ? documentCheck.issues : [];

  if (idDocumentStatus) {
    idDocumentStatus.textContent = status;
  }
  if (idDocumentRiskLevel) {
    idDocumentRiskLevel.textContent = riskLevel;
    idDocumentRiskLevel.className = `document-risk-pill ${status.toLowerCase()}`;
  }
  if (idDocumentSummary) {
    idDocumentSummary.textContent = summary;
  }
  if (idDocumentReupload) {
    if (documentCheck?.needs_reupload) {
      idDocumentReupload.hidden = false;
      idDocumentReupload.textContent = userMessage || "Please re-upload the ID photo.";
    } else {
      idDocumentReupload.hidden = true;
      idDocumentReupload.textContent = "";
    }
  }
  if (!idDocumentIssues) {
    return;
  }
  idDocumentIssues.innerHTML = "";

  if (issues.length === 0) {
    const clean = document.createElement("div");
    clean.className = "document-issue-card pass";
    clean.appendChild(buildDocumentIssueBadge("PASS"));

    const title = document.createElement("p");
    title.className = "document-issue-title";
    title.textContent = "No major issue detected";
    clean.appendChild(title);

    const message = document.createElement("p");
    message.className = "document-issue-message";
    message.textContent = "The uploaded ID photo passed the current document quality screening.";
    clean.appendChild(message);

    idDocumentIssues.appendChild(clean);
    return;
  }

  issues.forEach((issue) => {
    const severity = normalizeIssueSeverity(issue?.severity);
    const card = document.createElement("article");
    card.className = `document-issue-card ${severity}`;

    const header = document.createElement("div");
    header.className = "document-issue-header";

    const title = document.createElement("p");
    title.className = "document-issue-title";
    title.textContent = issue?.title ? String(issue.title) : "Issue";
    header.appendChild(title);
    header.appendChild(buildDocumentIssueBadge(severity.toUpperCase()));
    card.appendChild(header);

    const message = document.createElement("p");
    message.className = "document-issue-message";
    message.textContent =
      issue?.message && String(issue.message).trim().length > 0
        ? String(issue.message)
        : "No further detail.";
    card.appendChild(message);

    idDocumentIssues.appendChild(card);
  });
}

function normalizeDocumentCheckStatus(value) {
  const normalized = value == null ? "" : String(value).trim().toUpperCase();
  if (normalized === "PASS" || normalized === "REVIEW" || normalized === "REUPLOAD") {
    return normalized;
  }
  return "--";
}

function normalizeIssueSeverity(value) {
  const normalized = value == null ? "" : String(value).trim().toLowerCase();
  if (normalized === "high") {
    return "high";
  }
  if (normalized === "medium") {
    return "medium";
  }
  return "pass";
}

function buildDocumentIssueBadge(text) {
  const badge = document.createElement("span");
  const tone = normalizeDocumentIssueBadgeTone(text);
  badge.className = `document-issue-badge ${tone}`;
  badge.textContent = text;
  return badge;
}

function normalizeDocumentIssueBadgeTone(value) {
  const normalized = value == null ? "" : String(value).trim().toUpperCase();
  if (normalized === "HIGH" || normalized === "REUPLOAD") {
    return "high";
  }
  if (normalized === "MEDIUM" || normalized === "REVIEW") {
    return "medium";
  }
  return "pass";
}

function renderDecisionExplanation(explanation, decision, reasons) {
  if (ekycDecisionSummary) {
    const summary =
      typeof explanation?.summary === "string" && explanation.summary.trim().length > 0
        ? explanation.summary
        : buildDecisionSummaryFallback(decision, reasons);
    ekycDecisionSummary.textContent = summary;
  }
}

function buildDecisionSummaryFallback(decision, reasons) {
  const normalized = normalizeDecisionStatus(decision);
  const reasonText = Array.isArray(reasons) && reasons.length ? reasons.join("; ") : "No reasons provided.";
  return `${normalized}: ${reasonText}`;
}

function normalizeDecisionStatus(value) {
  const normalized = value == null ? "" : String(value).trim().toUpperCase();
  if (normalized === "PASS" || normalized === "REVIEW" || normalized === "REJECT") {
    return normalized;
  }
  return "REVIEW";
}

function buildDecisionStatusBadge(status) {
  const badge = document.createElement("span");
  badge.className = `decision-status-badge ${status.toLowerCase()}`;
  badge.textContent = status;
  return badge;
}

function updateExplainability({ qualityFlags, syncQuality, branchDetails }) {
  renderCodeTags(ekycQualityFlags, qualityFlags, "None");

  renderBoolBadge(ekycSyncMismatch, toOptionalBoolean(syncQuality?.mismatch));
  renderBoolBadge(ekycSyncInterpolated, toOptionalBoolean(syncQuality?.interpolated));
  renderBoolBadge(ekycSyncLengthBad, toOptionalBoolean(syncQuality?.length_bad));

  renderPercentValue(ekycAudioFake, branchDetails?.audio?.fake);
  renderPercentValue(ekycVideoFake, branchDetails?.video?.fake);
  renderPercentValue(ekycSyncFake, branchDetails?.sync?.fake);
}

function renderFusionCalibration(calibration, deepfakeValue) {
  const threshold =
    typeof calibration?.threshold === "number" && !Number.isNaN(calibration.threshold)
      ? calibration.threshold
      : null;
  const strategy =
    typeof calibration?.strategy === "string" && calibration.strategy.trim().length > 0
      ? calibration.strategy.trim()
      : "weighted";
  const weights = calibration?.weights || {};

  if (scoreFusionThreshold) {
    scoreFusionThreshold.textContent = threshold !== null ? `${(clamp01(threshold) * 100).toFixed(1)}%` : "N/A";
  }
  if (scoreFusionStrategy) {
    scoreFusionStrategy.textContent = `Strategy: ${strategy}`;
  }
  if (scoreFusionWeights) {
    const audio = typeof weights.audio === "number" ? weights.audio.toFixed(2) : "--";
    const video = typeof weights.video === "number" ? weights.video.toFixed(2) : "--";
    const sync = typeof weights.sync === "number" ? weights.sync.toFixed(2) : "--";
    scoreFusionWeights.textContent = `Weights: audio ${audio} / video ${video} / sync ${sync}`;
  }
  if (scoreDeepfakeMeta) {
    if (threshold !== null && typeof deepfakeValue === "number") {
      const relation = deepfakeValue >= threshold ? "above" : "below";
      scoreDeepfakeMeta.textContent = `Calibrated tri-modal fake score. Current score is ${relation} the threshold.`;
    } else {
      scoreDeepfakeMeta.textContent = "Calibrated tri-modal fake score.";
    }
  }
  if (ekycDecisionMeta) {
    if (threshold !== null && typeof deepfakeValue === "number") {
      ekycDecisionMeta.textContent = `Deepfake score ${(clamp01(deepfakeValue) * 100).toFixed(1)}% vs threshold ${(clamp01(threshold) * 100).toFixed(1)}%.`;
    } else {
      ekycDecisionMeta.textContent = "Deepfake score will be compared against the calibrated threshold after evaluation.";
    }
  }
}

function renderCodeTags(container, values, emptyText = "None") {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  const normalized = normalizeStringList(values);
  if (normalized.length === 0) {
    container.appendChild(buildBadge(emptyText, "muted"));
    return;
  }
  normalized.forEach((item) => {
    container.appendChild(buildBadge(item));
  });
}

function buildBadge(text, tone = "") {
  const badge = document.createElement("span");
  badge.className = `ekyc-badge${tone ? ` ${tone}` : ""}`;
  badge.textContent = text;
  return badge;
}

function normalizeStringList(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  return values
    .map((item) => (item == null ? "" : String(item).trim()))
    .filter((item) => item.length > 0);
}

function toOptionalBoolean(value) {
  if (typeof value === "boolean") {
    return value;
  }
  if (value === 0 || value === 1) {
    return Boolean(value);
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true" || normalized === "1" || normalized === "yes" || normalized === "on") {
      return true;
    }
    if (normalized === "false" || normalized === "0" || normalized === "no" || normalized === "off") {
      return false;
    }
  }
  return null;
}

function renderBoolBadge(element, value) {
  if (!element) {
    return;
  }
  element.classList.remove("true", "false", "unknown");
  if (typeof value === "boolean") {
    element.classList.add(value ? "true" : "false");
    element.textContent = value ? "True" : "False";
    return;
  }
  element.classList.add("unknown");
  element.textContent = "N/A";
}

function renderPercentValue(element, value) {
  if (!element) {
    return;
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    element.textContent = "N/A";
    return;
  }
  const clamped = clamp01(value);
  element.textContent = `${(clamped * 100).toFixed(1)}%`;
}

function updateLegacyPanelsFromEkyc(payload) {
  if (!payload) {
    return;
  }
  const details = payload.deepfake?.details || {};
  const matchThresholds = payload.match?.thresholds || {};
  const idVideoRejectThreshold =
    typeof matchThresholds.id_video_reject === "number" ? matchThresholds.id_video_reject : 0.458;
  const idVideoPassThreshold =
    typeof matchThresholds.id_video_pass === "number" ? matchThresholds.id_video_pass : 0.78;
  const fakeScore = typeof payload.deepfake?.score === "number" ? payload.deepfake.score : null;
  const label = payload.deepfake?.label || (fakeScore !== null && fakeScore >= 0.5 ? "Fake" : "Real");
  if (fakeScore !== null) {
    const final = {
      label,
      fake: fakeScore,
      real: 1 - fakeScore,
      confidence: Math.abs(fakeScore - (1 - fakeScore)),
    };
    updateResults({
      final,
      audio: details.audio,
      video: details.video,
      sync: details.sync,
    });
  }

  const artifacts = payload.artifacts || {};
  const idFacePath = artifacts.id_face_path;
  if (idFacePath) {
    updateIdFaceCard({
      ok: true,
      crop_url: idFacePath,
      photo_url: idPhotoPreviewUrl || undefined,
    });
  }

  const idVideo = payload.match?.id_video;
  if (idVideo) {
    updateIdMatchCard(idVideo, {
      rejectThreshold: idVideoRejectThreshold,
      passThreshold: idVideoPassThreshold,
    });
  }
}

function setImage(imgEl, hintEl, src, placeholder) {
  if (!imgEl) {
    return;
  }
  if (typeof src === "string" && src.length > 0) {
    imgEl.src = `${src}?t=${Date.now()}`;
    imgEl.hidden = false;
    if (hintEl) {
      hintEl.textContent = "";
    }
  } else {
    imgEl.removeAttribute("src");
    imgEl.hidden = true;
    if (hintEl) {
      hintEl.textContent = placeholder || "Unavailable";
    }
  }
}

function setScorePercent(textEl, barEl, value, type) {
  if (!textEl) {
    return;
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    textEl.textContent = "N/A";
    if (barEl) {
      barEl.style.width = "0%";
    }
    return;
  }
  const clamped = Math.max(0, Math.min(1, value));
  textEl.textContent = `${(clamped * 100).toFixed(1)}%`;
  if (barEl) {
    barEl.classList.toggle("fake", type === "fake");
    barEl.classList.toggle("real", type === "real");
    barEl.style.width = `${clamped * 100}%`;
    barEl.style.left = "0%";
  }
}

function setMatchScore(textEl, barEl, value) {
  if (!textEl) {
    return;
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    textEl.textContent = "N/A";
    if (barEl) {
      barEl.style.width = "0%";
    }
    return;
  }
  textEl.textContent = value.toFixed(3);
  const normalized = Math.max(0, Math.min(1, (value + 1) / 2));
  if (barEl) {
    barEl.classList.add("real");
    barEl.classList.remove("fake");
    barEl.style.width = `${normalized * 100}%`;
    barEl.style.left = "0%";
  }
}

function updateIdPanels(idFace, idMatch) {
  updateIdFaceCard(idFace);
  updateIdMatchCard(idMatch);
}

function updateIdFaceCard(data) {
  if (!data) {
    resetIdPanels();
    return;
  }
  if (data.ok) {
    if (idFacePreview && data.crop_url) {
      idFacePreview.src = `${data.crop_url}?t=${Date.now()}`;
    }
    if (idPhotoPreview && data.photo_url) {
      idPhotoPreview.src = `${data.photo_url}?t=${Date.now()}`;
    }
    return;
  }
  const error = data.error || "unknown_error";
  if (idFacePreview) {
    idFacePreview.removeAttribute("src");
  }
  showError(formatIdError(error));
}

function getMatchBand(prob, { rejectThreshold = null, passThreshold = 0.6 } = {}) {
  if (typeof prob !== "number" || Number.isNaN(prob)) {
    return "unknown";
  }
  if (typeof rejectThreshold === "number" && prob < rejectThreshold) {
    return "reject";
  }
  if (prob >= passThreshold) {
    return "pass";
  }
  if (typeof rejectThreshold === "number") {
    return "review";
  }
  return "reject";
}

function setMatchProbScore(textEl, barEl, value, { rejectThreshold = null, passThreshold = 0.6 } = {}) {
  if (!textEl) {
    return;
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    textEl.textContent = "N/A";
    if (barEl) {
      barEl.style.width = "0%";
    }
    return;
  }
  const prob = Math.max(0, Math.min(1, value));
  const band = getMatchBand(prob, { rejectThreshold, passThreshold });
  textEl.textContent = `${(prob * 100).toFixed(1)}%`;
  if (barEl) {
    barEl.classList.toggle("real", band === "pass");
    barEl.classList.toggle("review", band === "review");
    barEl.classList.toggle("fake", band === "reject");
    barEl.style.width = `${prob * 100}%`;
    barEl.style.left = "0%";
  }
}

function setMatchThresholdMeta(element, { rejectThreshold = null, passThreshold = 0.6 } = {}) {
  if (!element) {
    return;
  }
  if (typeof passThreshold !== "number" || Number.isNaN(passThreshold)) {
    element.textContent = "Match probability";
    return;
  }
  if (typeof rejectThreshold === "number" && !Number.isNaN(rejectThreshold)) {
    element.textContent =
      `Match probability · reject < ${(clamp01(rejectThreshold) * 100).toFixed(1)}% · pass >= ${(clamp01(passThreshold) * 100).toFixed(1)}%`;
    return;
  }
  element.textContent = `Match probability · pass line ${(clamp01(passThreshold) * 100).toFixed(1)}%`;
}

function updateIdMatchCard(data, { rejectThreshold = null, passThreshold = 0.6 } = {}) {
  if (!idResultGrid) {
    return;
  }
  const card = idResultGrid.querySelector('[data-key="id-match"]');
  if (!card) {
    return;
  }
  if (!data) {
    card.querySelector('[data-role="label"]').textContent = "Pending";
    card.querySelector('[data-role="score"]').textContent = "--";
    card.querySelector('[data-role="detail"]').textContent = "Awaiting comparison";
    resetMatchBar();
    return;
  }
  if (data.ok) {
    const score = typeof data.score === "number" ? data.score.toFixed(3) : "--";
    const prob = typeof data.prob === "number" ? (data.prob * 100).toFixed(1) : "--";
    const probValue = typeof data.prob === "number" ? data.prob : 0;
    const band =
      typeof data.decision === "string" && data.decision.trim().length > 0
        ? data.decision.trim().toUpperCase()
        : getMatchBand(probValue, { rejectThreshold, passThreshold }).toUpperCase();
    let label = "Review range";
    if (band === "PASS") {
      label = "Likely match";
    } else if (band === "REJECT") {
      label = "Mismatch";
    }
    card.querySelector('[data-role="label"]').textContent = label;
    card.querySelector('[data-role="score"]').textContent = `${prob}%`;
    if (typeof rejectThreshold === "number" && !Number.isNaN(rejectThreshold)) {
      card.querySelector('[data-role="detail"]').textContent =
        `Score ${score} · reject < ${(clamp01(rejectThreshold) * 100).toFixed(1)}% · pass >= ${(clamp01(passThreshold) * 100).toFixed(1)}%`;
    } else {
      card.querySelector('[data-role="detail"]').textContent =
        `Score ${score} · pass line ${(clamp01(passThreshold) * 100).toFixed(1)}%`;
    }
    updateMatchBar(probValue);
    if (idSummaryChip) {
      idSummaryChip.textContent = "Match complete";
    }
    return;
  }
  const error = data.error || "unknown_error";
  card.querySelector('[data-role="label"]').textContent = "Match failed";
  card.querySelector('[data-role="score"]').textContent = "--";
  card.querySelector('[data-role="detail"]').textContent = formatMatchError(error);
  resetMatchBar();
  if (idSummaryChip) {
    idSummaryChip.textContent = "Match error";
  }
}

function resetIdPanels() {
  if (!idResultGrid) {
    return;
  }
  const idFaceCard = idResultGrid.querySelector('[data-key="id-face"]');
  if (idFaceCard) {
    idFaceCard.querySelector('[data-role="label"]').textContent = "--";
    idFaceCard.querySelector('[data-role="score"]').textContent = "--";
    idFaceCard.querySelector('[data-role="detail"]').textContent = "Waiting for upload";
  }
  const idMatchCard = idResultGrid.querySelector('[data-key="id-match"]');
  if (idMatchCard) {
    idMatchCard.querySelector('[data-role="label"]').textContent = "Pending";
    idMatchCard.querySelector('[data-role="score"]').textContent = "--";
    idMatchCard.querySelector('[data-role="detail"]').textContent = "Awaiting comparison";
  }
  if (idSummaryChip) {
    idSummaryChip.textContent = "Awaiting upload";
  }
  if (idFacePreview) {
    idFacePreview.removeAttribute("src");
  }
  resetMatchBar();
}

function formatIdError(error) {
  switch (error) {
    case "no_face_detected":
      return "No face detected in ID photo.";
    case "face_too_small":
      return "Face is too small in ID photo.";
    case "read_error":
      return "ID photo could not be processed.";
    default:
      return `ID photo error: ${error}`;
  }
}

function formatMatchError(error) {
  switch (error) {
    case "id_face_unavailable":
      return "ID face extraction failed.";
    case "no_face_detected":
      return "No face detected in video.";
    case "face_too_small":
      return "Video face is too small.";
    case "read_error":
      return "Video could not be processed.";
    default:
      return `Match error: ${error}`;
  }
}

function updateMatchBar(probValue) {
  const matchPct = Math.min(Math.max(probValue, 0), 1) * 100;
  const nonMatchPct = 100 - matchPct;
  if (idMatchReal) {
    idMatchReal.style.width = `${matchPct}%`;
    idMatchReal.style.left = "0%";
  }
  if (idMatchFake) {
    idMatchFake.style.width = `${nonMatchPct}%`;
    idMatchFake.style.left = `${matchPct}%`;
  }
  if (idMatchRealLabel) {
    idMatchRealLabel.textContent = `Match ${matchPct.toFixed(1)}%`;
  }
  if (idMatchFakeLabel) {
    idMatchFakeLabel.textContent = `Non-match ${nonMatchPct.toFixed(1)}%`;
  }
}

function resetMatchBar() {
  if (idMatchReal) {
    idMatchReal.style.width = "0%";
    idMatchReal.style.left = "0%";
  }
  if (idMatchFake) {
    idMatchFake.style.width = "0%";
    idMatchFake.style.left = "0%";
  }
  if (idMatchRealLabel) {
    idMatchRealLabel.textContent = "Match --%";
  }
  if (idMatchFakeLabel) {
    idMatchFakeLabel.textContent = "Non-match --%";
  }
}

function updateCard(key, data) {
  const card = resultGrid?.querySelector(`[data-key="${key}"]`);
  if (!card || !data) {
    return;
  }
  const friendly = data.label === "Fake" ? "Suspicious" : "Likely genuine";
  const fake = clamp01(data.fake ?? 0);
  const real = clamp01(data.real ?? 0);
  const confidence = Math.max(fake, real);
  card.querySelector('[data-role="label"]').textContent = friendly;
  card.querySelector('[data-role="score"]').textContent = `Fake ${(fake * 100).toFixed(1)}%`;
  card.querySelector('[data-role="detail"]').textContent = `Real ${(real * 100).toFixed(1)}% | Confidence ${(confidence * 100).toFixed(1)}%`;
  card.classList.toggle("fake", data.label === "Fake");
  card.classList.toggle("real", data.label === "Real");
}

function clamp01(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return 0;
  }
  return Math.min(Math.max(value, 0), 1);
}

function initFusionGauge() {
  const ctx = document.getElementById("fusion-gauge");
  if (!ctx || typeof Chart === "undefined") {
    console.warn("Chart.js not available; fusion gauge disabled.");
    return;
  }
  fusionGauge = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Fake", "Real"],
      datasets: [
        {
          data: [0.5, 0.5],
          backgroundColor: ["#f43f5e", "#10b981"],
          borderWidth: 0,
          cutout: "70%",
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => {
              const label = context.label || "";
              const value = context.parsed * 100;
              return `${label}: ${value.toFixed(1)}%`;
            },
          },
        },
      },
    },
  });
}

function updateFusionGauge(data) {
  if (!fusionGauge || !data) {
    return;
  }
  const fake = clamp01(data.fake ?? 0);
  const real = clamp01(data.real ?? 0);
  fusionGauge.data.datasets[0].data = [fake, real];
  fusionGauge.update();
  const friendly = data.label === "Fake" ? "Suspicious" : "Likely genuine";
  if (gaugeLabel) {
    gaugeLabel.textContent = friendly;
  }
  if (gaugeScore) {
    gaugeScore.textContent = `Fake ${(fake * 100).toFixed(1)}%`;
  }
  fusionCard?.classList.toggle("fake", data.label === "Fake");
  fusionCard?.classList.toggle("real", data.label === "Real");
}

function updateContrastBars(result) {
  if (!contrastBars || !result) {
    return;
  }
  ["audio", "video", "sync"].forEach((key) => {
    const row = contrastBars.querySelector(`.bar-row[data-bar-key="${key}"]`);
    const data = result[key];
    if (!row || !data) {
      return;
    }
    const real = clamp01(data.real ?? 0);
    const fake = clamp01(data.fake ?? 0);
    const total = real + fake || 1;
    const realPct = (real / total) * 100;
    const fakePct = (fake / total) * 100;
    const realSeg = row.querySelector(".bar-segment.real");
    const fakeSeg = row.querySelector(".bar-segment.fake");
    if (realSeg) {
      realSeg.style.width = `${realPct}%`;
      realSeg.style.left = "0%";
    }
    if (fakeSeg) {
      fakeSeg.style.width = `${fakePct}%`;
      fakeSeg.style.left = `${realPct}%`;
    }
    const realLabel = row.querySelector("[data-real-value]");
    const fakeLabel = row.querySelector("[data-fake-value]");
    if (realLabel) {
      realLabel.textContent = `Real ${(real * 100).toFixed(1)}%`;
    }
    if (fakeLabel) {
      fakeLabel.textContent = `Fake ${(fake * 100).toFixed(1)}%`;
    }
  });
}

chooseBtn?.addEventListener("click", () => {
  fileInput?.click();
});

idPhotoBtn?.addEventListener("click", () => {
  idPhotoInput?.click();
});

fileInput?.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) {
    setSelectedFile(file);
  }
});

idPhotoInput?.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) {
    setIdPhotoFile(file);
  }
});

selfieBtn?.addEventListener("click", () => {
  selfieInput?.click();
});

selfieInput?.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) {
    setSelfieFile(file);
  }
});

uploadBox?.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadBox.classList.add("dragging");
});

uploadBox?.addEventListener("dragleave", () => {
  uploadBox.classList.remove("dragging");
});

uploadBox?.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadBox.classList.remove("dragging");
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    setSelectedFile(file);
  }
});

uploadBox?.addEventListener("click", (event) => {
  const target = event.target;
  if (target instanceof Element) {
    if (target.closest("button") || target.closest("input")) {
      return;
    }
  }
  fileInput?.click();
});

analyzeBtn?.addEventListener("click", () => {
  if (!selectedFile) {
    showError("Please choose a video file first.");
    return;
  }
  if (!selectedIdPhoto) {
    showError("Please upload an ID photo first.");
    return;
  }
  if (!selectedSelfie) {
    showError("Please upload a selfie photo first.");
    return;
  }
  uploadEvaluate();
});

cameraStartBtn?.addEventListener("click", () => {
  startRecording();
});

cameraStopBtn?.addEventListener("click", () => {
  stopRecording();
});

cameraResetBtn?.addEventListener("click", () => {
  resetRecording();
});

initFusionGauge();
