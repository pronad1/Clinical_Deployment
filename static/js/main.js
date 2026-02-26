// ===== Spinal Lesion Detection System - Main Application =====

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const loadingScreen = document.getElementById('loadingScreen');
const alertError = document.getElementById('alertError');
const resultsContainer = document.getElementById('resultsContainer');
const newScanBtn = document.getElementById('newScanBtn');

// State
let selectedFile = null;

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    initializeUploadZone();
    initializeEventListeners();
});

// ===== Upload Zone Initialization =====
function initializeUploadZone() {
    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
            fileInput.click();
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// ===== Event Listeners =====
function initializeEventListeners() {
    uploadBtn.addEventListener('click', handleUpload);
    newScanBtn.addEventListener('click', () => location.reload());
}

// ===== File Selection Handler =====
function handleFileSelect(file) {
    selectedFile = file;
    const fileName = file.name;
    const fileSize = (file.size / 1024 / 1024).toFixed(2);

    // Update upload zone UI
    document.querySelector('.upload-title').innerHTML =
        `<span style="color: var(--success);">✓ File Selected</span>`;
    document.querySelector('.upload-hint').innerHTML =
        `<strong>${fileName}</strong><br>Size: ${fileSize} MB`;

    uploadBtn.style.display = 'inline-flex';
    fileInput.value = '';
}

// ===== Upload and Analysis =====
async function handleUpload(e) {
    e.preventDefault();
    e.stopPropagation();

    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Show loading state
    showLoading();

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || data.error || 'Upload failed');
        }

        displayResults(data);

    } catch (error) {
        displayError(error.message);
    }
}

// ===== UI State Management =====
function showLoading() {
    uploadZone.parentElement.style.display = 'none';
    alertError.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingScreen.style.display = 'block';
}

function hideLoading() {
    loadingScreen.style.display = 'none';
}

function displayError(message) {
    hideLoading();
    alertError.style.display = 'block';

    const errorMsg = document.getElementById('errorMessage');
    if (message.includes('spine') || message.includes('SPINE') || message.includes('appears to be')) {
        errorMsg.innerHTML = `<strong>❌ Incorrect Image Type</strong><br><br>${message}<br><br><em style="color: var(--gray-600);">This system is specifically designed for spine X-ray and MRI analysis only.</em>`;
    } else {
        errorMsg.textContent = message;
    }

    // Scroll to error
    alertError.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetUpload() {
    selectedFile = null;
    alertError.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingScreen.style.display = 'none';
    uploadZone.parentElement.style.display = 'block';
    uploadBtn.style.display = 'none';

    document.querySelector('.upload-title').textContent = 'Drop your DICOM file here or click to browse';
    document.querySelector('.upload-hint').textContent = 'Supported format: .dcm, .dicom (Max 16MB)';
    fileInput.value = '';
}

// ===== Results Display =====
function displayResults(data) {
    hideLoading();
    resultsContainer.style.display = 'block';

    // Classification Results
    displayClassification(data.classification);

    // Detection Results
    if (data.detection && data.detection.num_detections > 0) {
        displayDetection(data.detection);
    }

    // Original Image Preview
    displayPreview(data);

    // Metadata
    displayMetadata(data.metadata);

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayClassification(classification) {
    const isAbnormal = classification.is_abnormal;
    const card = document.getElementById('classificationCard');

    card.className = `result-card ${isAbnormal ? 'abnormal' : 'normal'}`;

    const badge = document.getElementById('statusBadge');
    badge.className = `status-badge ${isAbnormal ? 'abnormal' : 'normal'}`;
    badge.textContent = isAbnormal ? '⚠️ ABNORMAL' : '✓ NORMAL';

    // Update metrics
    document.getElementById('ensembleScore').textContent =
        (classification.ensemble_score * 100).toFixed(2) + '%';
    document.getElementById('confidence').textContent =
        classification.confidence.toFixed(2) + '%';

    // Individual predictions
    const predictions = classification.individual_predictions;
    document.getElementById('densenet').textContent =
        predictions.densenet121 ? (predictions.densenet121 * 100).toFixed(2) + '%' : 'N/A';
    document.getElementById('resnet').textContent =
        predictions.resnet50 ? (predictions.resnet50 * 100).toFixed(2) + '%' : 'N/A';
    document.getElementById('efficientnet').textContent =
        predictions.efficientnet ? (predictions.efficientnet * 100).toFixed(2) + '%' : 'N/A';
}

function displayDetection(detection) {
    const card = document.getElementById('detectionCard');
    card.style.display = 'block';

    document.getElementById('numDetections').textContent = detection.num_detections;

    const detectionList = document.getElementById('detectionList');
    detectionList.innerHTML = '';

    detection.detections.forEach((det, idx) => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        item.innerHTML = `
            <strong>Lesion ${idx + 1}:</strong> ${det.class}<br>
            <strong>Confidence:</strong> ${(det.confidence * 100).toFixed(2)}%<br>
            <strong>Location:</strong> [${det.bbox.map(v => v.toFixed(0)).join(', ')}]
        `;
        detectionList.appendChild(item);
    });

    document.getElementById('annotatedImage').src =
        'data:image/png;base64,' + detection.annotated_image;
}

function displayPreview(data) {
    document.getElementById('previewImage').src =
        'data:image/png;base64,' + data.preview_image;
}

function displayMetadata(metadata) {
    document.getElementById('patientId').textContent = metadata.patient_id;
    document.getElementById('studyDate').textContent = metadata.study_date;
    document.getElementById('modality').textContent = metadata.modality;
    document.getElementById('imageSize').textContent = metadata.image_size;
}

// ===== Download Functions =====
function downloadImage(imageId, prefix) {
    const img = document.getElementById(imageId);
    const imgSrc = img.src;

    if (!imgSrc || imgSrc === '') {
        alert('No image available to download');
        return;
    }

    const link = document.createElement('a');
    link.href = imgSrc;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
    link.download = `${prefix}_${timestamp}.jpg`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

window.downloadAnnotatedImage = () => downloadImage('annotatedImage', 'spine_lesions_detected');
window.downloadOriginalImage = () => downloadImage('previewImage', 'spine_original');
window.downloadGradCAM = () => downloadImage('gradcamImage', 'gradcam_ensemble');
window.downloadSegmentation = () => downloadImage('segmentationImage', 'segmentation_analysis');

window.downloadLIME = () => {
    const img = document.getElementById('limeImage');
    if (!img.src) {
        alert('No LIME visualization to download');
        return;
    }

    const link = document.createElement('a');
    const base64Data = img.src.split(',')[1];
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });

    link.href = URL.createObjectURL(blob);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    link.download = `lime_explainability_${timestamp}.png`;
    link.click();
    URL.revokeObjectURL(link.href);
};

// ===== Visualization Generation Functions =====
async function generateVisualization(endpoint, btnId, loadingId, resultId, imageId) {
    if (!selectedFile) {
        alert('Please upload a file first');
        return;
    }

    const btn = document.getElementById(btnId);
    const loading = document.getElementById(loadingId);
    const result = document.getElementById(resultId);

    btn.disabled = true;
    result.style.display = 'none';
    loading.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        if (endpoint === '/generate-lime') {
            formData.append('style', 'grid');
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || data.error || 'Visualization generation failed');
        }

        // Display result
        const imageKey = endpoint.includes('gradcam') ? 'gradcam_image' :
            endpoint.includes('lime') ? 'lime_image' :
                'segmentation_image';
        document.getElementById(imageId).src = 'data:image/png;base64,' + data[imageKey];
        loading.style.display = 'none';
        result.style.display = 'block';

    } catch (error) {
        loading.style.display = 'none';
        alert('Error generating visualization: ' + error.message);
    } finally {
        btn.disabled = false;
    }
}

window.generateGradCAM = () => generateVisualization(
    '/generate-gradcam', 'gradcamBtn', 'gradcamLoading', 'gradcamResult', 'gradcamImage'
);

window.generateLIME = () => generateVisualization(
    '/generate-lime', 'limeBtn', 'limeLoading', 'limeResult', 'limeImage'
);

window.generateSegmentation = () => generateVisualization(
    '/generate-segmentation', 'segmentationBtn', 'segmentationLoading', 'segmentationResult', 'segmentationImage'
);

// ===== Error Handler for Retry =====
window.retryUpload = resetUpload;
