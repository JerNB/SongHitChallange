let currentAnalysis = null;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("resume-file");
const evaluateBtn = document.getElementById("evaluate-btn");
const spinner = document.getElementById("spinner");
const errorMessage = document.getElementById("error-message");
const resultsContainer = document.getElementById("results");
const downloadJsonBtn = document.getElementById("download-json");
const downloadPdfBtn = document.getElementById("download-pdf");
const tabButtons = document.querySelectorAll(".tab-btn");

const resumeTextInput = document.getElementById("resume-text");
const jobDescriptionInput = document.getElementById("job-description");

function setLoading(isLoading) {
  if (isLoading) {
    spinner.classList.remove("hidden");
    evaluateBtn.setAttribute("disabled", "disabled");
  } else {
    spinner.classList.add("hidden");
    evaluateBtn.removeAttribute("disabled");
  }
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove("hidden");
}

function clearError() {
  errorMessage.classList.add("hidden");
  errorMessage.textContent = "";
}

function activateTab(tab) {
  tabButtons.forEach((btn) => {
    if (btn.dataset.tab === tab) {
      btn.classList.add("active");
      btn.classList.add("text-indigo-600", "border-b-2", "border-indigo-600");
      btn.classList.remove("text-gray-500");
    } else {
      btn.classList.remove("active");
      btn.classList.remove("text-indigo-600", "border-b-2", "border-indigo-600");
      btn.classList.add("text-gray-500");
    }
  });
  if (!currentAnalysis) {
    resultsContainer.innerHTML = '<p class="text-gray-500">Run an evaluation to see results.</p>';
    return;
  }
  switch (tab) {
    case "overview":
      resultsContainer.innerHTML = renderOverview(currentAnalysis);
      break;
    case "sections":
      resultsContainer.innerHTML = renderSections(currentAnalysis);
      break;
    case "bullets":
      resultsContainer.innerHTML = renderBullets(currentAnalysis);
      break;
    case "keywords":
      resultsContainer.innerHTML = renderKeywords(currentAnalysis);
      break;
    case "report":
      resultsContainer.innerHTML = renderReport(currentAnalysis);
      break;
    default:
      resultsContainer.innerHTML = renderOverview(currentAnalysis);
  }
}

function renderOverview(analysis) {
  const categories = analysis.category_scores || {};
  const highImpact = analysis.high_impact_fixes || [];
  const categoryHtml = Object.entries(categories)
    .map(([key, value]) => {
      const width = Math.min(100, (value / 20) * 100);
      return `
        <div class="mb-3">
          <div class="flex justify-between text-sm font-medium">
            <span>${key}</span>
            <span>${value} / 20</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div class="bg-indigo-500 h-2 rounded-full" style="width: ${width}%;"></div>
          </div>
        </div>
      `;
    })
    .join("");
  const highImpactHtml = highImpact.length
    ? `<ol class="list-decimal list-inside space-y-1 text-sm">${highImpact
        .map((item) => `<li>${item}</li>`)
        .join("")}</ol>`
    : `<p class="text-gray-500 text-sm">No urgent fixes detected.</p>`;
  return `
    <div class="space-y-4">
      <div class="bg-indigo-50 border border-indigo-100 rounded-md p-4">
        <p class="text-xs uppercase text-indigo-500 font-semibold">Overall Score</p>
        <p class="text-3xl font-bold">${analysis.overall_score}</p>
      </div>
      <div>
        <h3 class="font-semibold mb-2">Category Breakdown</h3>
        ${categoryHtml}
      </div>
      <div>
        <h3 class="font-semibold mb-2">High-Impact Fixes</h3>
        ${highImpactHtml}
      </div>
    </div>
  `;
}

function renderSections(analysis) {
  const metadata = analysis.metadata || {};
  const findings = (analysis.findings || []).filter((f) => f.section !== "Bullets");
  const metaList = `
    <ul class="space-y-1 text-sm">
      <li><strong>Words:</strong> ${metadata.words ?? "-"}</li>
      <li><strong>Estimated pages:</strong> ${metadata.estimated_pages ?? "-"}</li>
      <li><strong>Sections detected:</strong> ${(metadata.detected_sections || []).join(", ") || "None"}</li>
      <li><strong>Bullet count:</strong> ${metadata.bullet_count ?? 0}</li>
      <li><strong>Average bullet length:</strong> ${metadata.average_bullet_length ?? 0} words</li>
    </ul>
  `;
  const findingsHtml = findings.length
    ? findings
        .map(
          (f) => `
          <div class="border border-gray-200 rounded-md p-3 mb-2">
            <p class="text-xs uppercase text-gray-500">${f.section} &middot; ${f.severity}</p>
            <p class="font-medium">${f.message}</p>
            <p class="text-gray-600 text-sm"><strong>Evidence:</strong> ${f.evidence || "-"}</p>
            <p class="text-gray-600 text-sm"><strong>How to fix:</strong> ${f.fix}</p>
          </div>
        `
        )
        .join("")
    : `<p class="text-gray-500">No structural concerns detected.</p>`;
  return `
    <div class="space-y-4">
      <div>
        <h3 class="font-semibold mb-2">Resume Metadata</h3>
        ${metaList}
      </div>
      <div>
        <h3 class="font-semibold mb-2">Findings</h3>
        ${findingsHtml}
      </div>
    </div>
  `;
}

function renderBullets(analysis) {
  const suggestions = analysis.bullet_suggestions || [];
  if (!suggestions.length) {
    return '<p class="text-gray-500">No bullet suggestions generated.</p>';
  }
  return `
    <div class="space-y-3">
      ${suggestions
        .map(
          (item) => `
            <div class="border border-gray-200 rounded-md p-3">
              <p class="text-xs uppercase text-gray-500">Original</p>
              <p class="text-gray-800">${item.original}</p>
              <p class="text-xs uppercase text-gray-500 mt-2">Improved</p>
              <p class="font-medium text-gray-900">${item.improved}</p>
              <p class="text-gray-600 text-sm mt-1">${item.rationale}</p>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderKeywords(analysis) {
  const coverage = analysis.keyword_coverage || {};
  const mode = coverage.mode === "jd" ? "Job Description" : "Recommended";
  const matched = coverage.matched || [];
  const missing = coverage.missing || [];
  return `
    <div class="space-y-4">
      <p class="text-sm text-gray-600">Mode: ${mode}</p>
      <div>
        <h3 class="font-semibold">Matched Keywords (${matched.length})</h3>
        <div class="flex flex-wrap gap-2 mt-2">
          ${matched
            .map((k) => `<span class="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">${k}</span>`)
            .join("") || '<span class="text-gray-500 text-sm">None yet.</span>'}
        </div>
      </div>
      <div>
        <h3 class="font-semibold">Missing Keywords (${missing.length})</h3>
        <div class="flex flex-wrap gap-2 mt-2">
          ${missing
            .map((k) => `<span class="px-2 py-1 bg-red-100 text-red-700 rounded-full text-xs">${k}</span>`)
            .join("") || '<span class="text-gray-500 text-sm">None missing.</span>'}
        </div>
      </div>
      <p class="text-sm text-gray-600">${coverage.notes || ""}</p>
    </div>
  `;
}

function renderReport(analysis) {
  const findings = analysis.findings || [];
  if (!findings.length) {
    return '<p class="text-gray-500">No findings reported.</p>';
  }
  return `
    <div class="space-y-3">
      ${findings
        .map(
          (f) => `
            <div class="border border-gray-200 rounded-md p-3">
              <p class="text-xs uppercase text-gray-500">${f.section} &middot; ${f.severity}</p>
              <p class="font-semibold">${f.message}</p>
              <p class="text-sm text-gray-600"><strong>Evidence:</strong> ${f.evidence || "-"}</p>
              <p class="text-sm text-gray-600"><strong>How to fix:</strong> ${f.fix}</p>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function resetResults() {
  currentAnalysis = null;
  resultsContainer.innerHTML = '<p class="text-gray-500">Results will appear here after evaluation.</p>';
  downloadJsonBtn.disabled = true;
  downloadPdfBtn.disabled = true;
}

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add("border-indigo-400", "bg-indigo-50");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-indigo-400", "bg-indigo-50");
  });
});

dropzone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files && files[0]) {
    fileInput.files = files;
  }
});

evaluateBtn.addEventListener("click", async () => {
  clearError();
  setLoading(true);
  try {
    const formData = new FormData();
    const file = fileInput.files[0];
    if (file) {
      formData.append("file", file);
    }
    if (resumeTextInput.value.trim()) {
      formData.append("resume_text", resumeTextInput.value.trim());
    }
    if (jobDescriptionInput.value.trim()) {
      formData.append("job_description", jobDescriptionInput.value.trim());
    }
    if (!file && !resumeTextInput.value.trim()) {
      setLoading(false);
      showError("Upload a resume file or paste the resume text.");
      return;
    }
    const response = await fetch("/api/evaluate", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Unable to evaluate resume.");
    }
    currentAnalysis = data;
    downloadJsonBtn.disabled = false;
    downloadPdfBtn.disabled = false;
    activateTab(document.querySelector(".tab-btn.active")?.dataset.tab || "overview");
  } catch (err) {
    console.error(err);
    showError(err.message || "Unexpected error.");
    resetResults();
  } finally {
    setLoading(false);
  }
});

downloadJsonBtn.addEventListener("click", async () => {
  if (!currentAnalysis) return;
  await downloadReport("json");
});

downloadPdfBtn.addEventListener("click", async () => {
  if (!currentAnalysis) return;
  await downloadReport("pdf");
});

async function downloadReport(kind) {
  const endpoint = kind === "pdf" ? "/api/report/pdf" : "/api/report/json";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis: currentAnalysis }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    showError(data.error || "Unable to generate report.");
    return;
  }
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = kind === "pdf" ? "resume_report.pdf" : "resume_report.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activateTab(btn.dataset.tab);
  });
});

resetResults();
activateTab("overview");
