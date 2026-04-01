/**
 * app.js — Breast Cancer Image Classification Based on Deep Transfer Learning — Frontend Logic
 * Handles drag-drop upload, API calls, and animated results display
 */

'use strict';

const API_BASE = '';  // Same origin — FastAPI serves both API and static files

// ── DOM References ──────────────────────────────────────────────────────────
const dropZone         = document.getElementById('drop-zone');
const fileInput        = document.getElementById('file-input');
const browseBtn        = document.getElementById('browse-btn');
const changeBtn        = document.getElementById('change-btn');
const analyzeBtn       = document.getElementById('analyze-btn');
const reanalyzeBtn     = document.getElementById('reanalyze-btn');
const retryBtn         = document.getElementById('retry-btn');

const dropZoneContent  = document.getElementById('drop-zone-content');
const previewContent   = document.getElementById('preview-content');
const previewImg       = document.getElementById('preview-img');
const previewFilename  = document.getElementById('preview-filename');
const previewFilesize  = document.getElementById('preview-filesize');

const modelStatusBadge = document.getElementById('model-status-badge');
const statusDot        = modelStatusBadge.querySelector('.status-dot');
const statusText       = modelStatusBadge.querySelector('.status-text');

const resultsEmpty     = document.getElementById('results-empty');
const resultsLoading   = document.getElementById('results-loading');
const resultsContent   = document.getElementById('results-content');
const resultsError     = document.getElementById('results-error');
const errorMessage     = document.getElementById('error-message');

const btnText          = analyzeBtn.querySelector('.btn-text');
const btnSpinner       = analyzeBtn.querySelector('.btn-spinner');
const toast            = document.getElementById('toast');
const locationInput    = document.getElementById('location-input');
const recommendBtn     = document.getElementById('recommend-btn');
const hospitalStatus   = document.getElementById('hospital-status');
const hospitalSummary  = document.getElementById('hospital-summary');
const hospitalList     = document.getElementById('hospital-list');

// ── State ───────────────────────────────────────────────────────────────────
let selectedFile = null;
let toastTimeout = null;
let latestPredictionLabel = null;

// ── Toast Notifications ─────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3000) {
  clearTimeout(toastTimeout);
  toast.textContent = message;
  toast.className = `toast show ${type === 'error' ? 'toast-error' : type === 'ok' ? 'toast-ok' : ''}`;
  toast.removeAttribute('hidden');
  toastTimeout = setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.classList.add('hidden'), 300);
  }, duration);
}

// ── Model Status Badge ──────────────────────────────────────────────────────
async function checkModelStatus() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(6000) });
    const data = await res.json();
    if (data.model_loaded) {
      setStatus('ok', '✓ Model Ready');
    } else {
      setStatus('error', '! Model Not Found');
      showToast('Model not found. Please run train.py first.', 'error', 6000);
    }
  } catch {
    setStatus('error', '✗ Server Offline');
    showToast('Backend server is offline. Please start it with: uvicorn main:app --reload', 'error', 6000);
  }
}

function setStatus(type, text) {
  modelStatusBadge.className = `status-badge status-${type}`;
  statusText.textContent = text;
}

// ── File Handling ───────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function isValidImageType(file) {
  return ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'].includes(file.type);
}

function setFile(file) {
  if (!file) return;

  if (!isValidImageType(file)) {
    showToast('Unsupported file type. Please upload JPEG, PNG, BMP, TIFF, or WebP.', 'error');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast('File too large. Maximum size is 20MB.', 'error');
    return;
  }

  selectedFile = file;

  // Build preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.alt = `Preview of ${file.name}`;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    dropZoneContent.classList.add('hidden');
    previewContent.classList.remove('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.setAttribute('aria-disabled', 'false');
    showToast('Image loaded — click Analyze to run AI detection', 'ok');
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  dropZoneContent.classList.remove('hidden');
  previewContent.classList.add('hidden');
  analyzeBtn.disabled = true;
  analyzeBtn.setAttribute('aria-disabled', 'true');
  latestPredictionLabel = null;
  hospitalStatus.textContent = '';
  hospitalStatus.classList.add('hidden');
  hospitalSummary.textContent = '';
  hospitalSummary.classList.add('hidden');
  hospitalList.innerHTML = '';
  hospitalList.classList.add('hidden');
}

// ── Drag & Drop ─────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
dropZone.addEventListener('click', (e) => {
  if (!e.target.closest('button')) fileInput.click();
});
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
changeBtn.addEventListener('click', (e) => { e.stopPropagation(); clearFile(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

// ── Results Panel State ─────────────────────────────────────────────────────
function showPanel(id) {
  [resultsEmpty, resultsLoading, resultsContent, resultsError].forEach(el => {
    el.classList.toggle('hidden', el.id !== id);
  });
}

// ── Animated Progress Bar ───────────────────────────────────────────────────
function animateBar(el, targetPct, delay = 0) {
  el.style.width = '0%';
  setTimeout(() => { el.style.width = `${(targetPct * 100).toFixed(1)}%`; }, delay);
}

// ── Display Results ─────────────────────────────────────────────────────────
function displayResults(data) {
  showPanel('results-content');
  latestPredictionLabel = data.prediction;

  const isMalignant   = data.is_malignant;
  const negProb       = data.idc_negative_prob;
  const posProb       = data.idc_positive_prob;
  const confidencePct = (data.confidence * 100).toFixed(1);

  // Badge
  const resultBadge = document.getElementById('result-badge');
  resultBadge.className = `result-badge ${isMalignant ? 'badge-malignant' : 'badge-benign'}`;
  resultBadge.textContent = isMalignant ? 'IDC Positive' : 'IDC Negative';

  // Prediction display
  const predictionDisplay  = document.getElementById('prediction-display');
  const predictionIconWrap = document.getElementById('prediction-icon-wrap');
  const predictionLabel    = document.getElementById('prediction-label');
  const predictionSub      = document.getElementById('prediction-sub');

  predictionDisplay.className = `prediction-display ${isMalignant ? 'malignant-display' : 'benign-display'}`;
  predictionIconWrap.textContent = isMalignant ? '⚠️' : '✅';
  predictionLabel.className = `prediction-label ${isMalignant ? 'label-malignant' : 'label-benign'}`;
  predictionLabel.textContent = data.prediction;
  predictionSub.textContent   = isMalignant
    ? 'Invasive Ductal Carcinoma (IDC+) detected'
    : 'No IDC detected — tissue appears benign';

  // Confidence bar
  const confidenceBar = document.getElementById('confidence-bar');
  document.getElementById('confidence-pct').textContent = `${confidencePct}%`;
  confidenceBar.className = `progress-bar ${isMalignant ? 'malignant-bar' : 'benign-bar'}`;
  animateBar(confidenceBar, data.confidence, 200);

  // Probability bars
  document.getElementById('prob-benign-val').textContent    = `${(negProb * 100).toFixed(1)}%`;
  document.getElementById('prob-malignant-val').textContent = `${(posProb * 100).toFixed(1)}%`;
  animateBar(document.getElementById('prob-benign-bar'),    negProb, 400);
  animateBar(document.getElementById('prob-malignant-bar'), posProb, 550);

  // Metadata
  document.getElementById('meta-model').textContent    = data.models_used?.join(', ') || 'DenseNet121';
  document.getElementById('meta-filename').textContent = data.filename || 'unknown';
  document.getElementById('meta-imgsize').textContent  = data.img_size_used || '224×224';

  // Explainability images (Grad-CAM)
  const explainabilitySection = document.getElementById('explainability-section');
  const gradcamImage = document.getElementById('gradcam-image');
  const gradcamHeatmap = document.getElementById('gradcam-heatmap');
  if (data.gradcam_overlay_base64) {
    gradcamImage.src = `data:image/png;base64,${data.gradcam_overlay_base64}`;
    gradcamHeatmap.src = data.gradcam_heatmap_base64
      ? `data:image/png;base64,${data.gradcam_heatmap_base64}`
      : '';
    gradcamHeatmap.parentElement.classList.toggle('hidden', !data.gradcam_heatmap_base64);
    explainabilitySection.classList.remove('hidden');
  } else {
    gradcamImage.src = '';
    gradcamHeatmap.src = '';
    explainabilitySection.classList.add('hidden');
  }

  // Scroll to results on mobile
  if (window.innerWidth < 860) {
    document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function renderHospitalCards(items) {
  hospitalList.innerHTML = '';
  items.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'hospital-card';
    card.innerHTML = `
      <div class="hospital-name">${idx + 1}. ${item.name}</div>
      <div class="hospital-meta">Distance: ${item.distance_km} km</div>
      <div class="hospital-meta">Address: ${item.address}</div>
      <div class="hospital-meta">Phone: ${item.phone}</div>
      <div class="hospital-meta">Website: ${item.website}</div>
    `;
    hospitalList.appendChild(card);
  });
  hospitalList.classList.remove('hidden');
}

async function fetchHospitalRecommendations() {
  const location = (locationInput.value || '').trim();
  if (!location) {
    showToast('Please enter your city or area first.', 'error');
    locationInput.focus();
    return;
  }

  const diagnosis = latestPredictionLabel || 'General Screening';
  hospitalStatus.textContent = 'Searching for nearby hospitals...';
  hospitalStatus.classList.remove('hidden');
  hospitalSummary.classList.add('hidden');
  hospitalList.innerHTML = '';
  hospitalList.classList.add('hidden');
  recommendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/recommend-hospitals`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location, diagnosis, radius_km: 35 }),
      signal: AbortSignal.timeout(30000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    hospitalStatus.textContent = `${data.hospitals.length} hospital${data.hospitals.length !== 1 ? 's' : ''} found near ${data.location?.resolved_name || location}`;
    hospitalSummary.textContent = data.summary || 'No recommendation summary available.';
    hospitalSummary.classList.remove('hidden');
    renderHospitalCards(data.hospitals || []);
    showToast(`Found ${data.hospitals.length} hospitals near your location.`, 'ok', 4000);
  } catch (err) {
    const msg = err.message || 'Failed to fetch recommendations.';
    hospitalStatus.textContent = `Error: ${msg}`;
    hospitalSummary.classList.add('hidden');
    hospitalList.classList.add('hidden');
    showToast(`Hospital search failed: ${msg}`, 'error', 5000);
  } finally {
    recommendBtn.disabled = false;
  }
}

// ── Analyze ─────────────────────────────────────────────────────────────────
async function analyze() {
  if (!selectedFile) { showToast('Please select an image first.', 'error'); return; }

  // Button loading state
  btnText.classList.add('hidden');
  btnSpinner.classList.remove('hidden');
  analyzeBtn.disabled = true;
  showPanel('results-loading');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch(`${API_BASE}/predict?include_heatmap=true`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(30000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    displayResults(data);
    showToast(
      data.is_malignant
        ? `⚠️ Malignant detected (${(data.confidence * 100).toFixed(1)}% confidence)`
        : `✅ Benign — no IDC detected (${(data.confidence * 100).toFixed(1)}% confidence)`,
      data.is_malignant ? 'error' : 'ok',
      5000
    );
  } catch (err) {
    let msg = err.message || 'An unexpected error occurred.';
    if (err.name === 'TimeoutError') msg = 'Request timed out. The model may still be loading.';
    showPanel('results-error');
    errorMessage.textContent = msg;
    showToast(`Error: ${msg}`, 'error', 5000);
  } finally {
    btnText.classList.remove('hidden');
    btnSpinner.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
}

// ── Button Listeners ────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', analyze);
reanalyzeBtn.addEventListener('click', () => {
  clearFile();
  showPanel('results-empty');
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
retryBtn.addEventListener('click', () => {
  if (selectedFile) {
    analyze();
  } else {
    showPanel('results-empty');
  }
});
recommendBtn.addEventListener('click', fetchHospitalRecommendations);
locationInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') fetchHospitalRecommendations();
});

// ── Init ────────────────────────────────────────────────────────────────────
(function init() {
  checkModelStatus();
  showPanel('results-empty');

  // Re-check every 30 seconds if server was offline
  setInterval(async () => {
    if (modelStatusBadge.classList.contains('status-error')) {
      await checkModelStatus();
    }
  }, 30000);
})();
