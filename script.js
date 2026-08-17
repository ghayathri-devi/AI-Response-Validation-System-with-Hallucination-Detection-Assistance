const API_BASE = "";
const EVALUATE_URL = `${API_BASE}/evaluate`;
const HISTORY_URL = `${API_BASE}/history`;

const button = document.getElementById("evaluateBtn");
const pdfInput = document.getElementById("sourcePdf");
const clearPdfBtn = document.getElementById("clearPdfBtn");
const pdfFileName = document.getElementById("pdfFileName");

const batchButton = document.getElementById("batchEvaluateBtn");
const batchCsvInput = document.getElementById("batchCsv");
const clearBatchCsvBtn = document.getElementById("clearBatchCsvBtn");
const batchFileName = document.getElementById("batchFileName");

button.addEventListener("click", evaluateResponse);
batchButton.addEventListener("click", runBatchEvaluation);

//-------------------------------------
// Tab navigation
//-------------------------------------

const navLinks = document.querySelectorAll(".nav-link");
const tabPages = document.querySelectorAll(".tab-page");

navLinks.forEach((link) => {
    link.addEventListener("click", () => {
        const targetId = link.getAttribute("data-tab");

        navLinks.forEach((l) => l.classList.remove("active"));
        tabPages.forEach((p) => p.classList.remove("active"));

        link.classList.add("active");
        document.getElementById(targetId).classList.add("active");

        if (targetId === "analyticsTab") {
            loadAnalytics();
        }
    });
});

//-------------------------------------
// Analytics filters
//-------------------------------------

document.getElementById("applyFiltersBtn").addEventListener("click", () => {
    loadAnalytics();
});

document.getElementById("clearFiltersBtn").addEventListener("click", () => {
    document.getElementById("filterStartDate").value = "";
    document.getElementById("filterEndDate").value = "";
    document.getElementById("filterMode").value = "";
    loadAnalytics();
});

//-------------------------------------
// Show selected filename, or clear it (single-evaluation PDF)
//-------------------------------------

pdfInput.addEventListener("change", () => {
    if (pdfInput.files.length > 0) {
        pdfFileName.textContent = `Selected: ${pdfInput.files[0].name}`;
    } else {
        pdfFileName.textContent = "";
    }
});

clearPdfBtn.addEventListener("click", () => {
    pdfInput.value = "";
    pdfFileName.textContent = "";
});

//-------------------------------------
// Batch CSV file select / clear
//-------------------------------------

batchCsvInput.addEventListener("change", () => {
    if (batchCsvInput.files.length > 0) {
        batchFileName.textContent = `Selected: ${batchCsvInput.files[0].name}`;
    } else {
        batchFileName.textContent = "";
    }
});

clearBatchCsvBtn.addEventListener("click", () => {
    batchCsvInput.value = "";
    batchFileName.textContent = "";
});

//-------------------------------------
// Run a single evaluation
//-------------------------------------

async function evaluateResponse() {

    const question = document.getElementById("question").value.trim();
    const answer = document.getElementById("answer").value.trim();
    const reference = document.getElementById("reference").value.trim();
    const pdfFile = pdfInput.files.length > 0 ? pdfInput.files[0] : null;

    if (!question || !answer) {
        alert("Please fill in the question and candidate answer.");
        return;
    }

    button.innerHTML = "⏳ Evaluating...";
    button.disabled = true;

    try {

        const formData = new FormData();
        formData.append("question", question);
        formData.append("answer", answer);

        if (reference) {
            formData.append("reference_answer", reference);
        }

        if (pdfFile) {
            formData.append("source_pdf", pdfFile);
        }

        const response = await fetch(EVALUATE_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Server Error");
        }

        const data = await response.json();

        console.log(data);

        const evaluation = data.evaluation;

        //---------------------------------------
        // Verdict (aggregated result)
        //---------------------------------------

        document.getElementById("verdictLabel").innerHTML =
            evaluation.verdict_label || "--";

        document.getElementById("overall").innerHTML =
            formatScore(evaluation.overall_score);

        document.getElementById("verdictSummary").innerHTML =
            evaluation.verdict_summary || "";

        applyVerdictColor(evaluation.verdict_label);

        //---------------------------------------
        // Individual agent scores
        //---------------------------------------

        document.getElementById("accuracy").innerHTML =
            formatScore(evaluation.accuracy);

        document.getElementById("accuracyEvidence").innerHTML =
            evaluation.accuracy_evidence || "";

        document.getElementById("accuracyExcerpt").innerHTML =
            evaluation.accuracy_supporting_excerpt
                ? `“${evaluation.accuracy_supporting_excerpt}”`
                : "";

        document.getElementById("relevance").innerHTML =
            formatScore(evaluation.relevance);

        document.getElementById("relevanceReasoning").innerHTML =
            evaluation.relevance_reasoning || "";

        document.getElementById("completeness").innerHTML =
            formatScore(evaluation.completeness);

        document.getElementById("completenessReasoning").innerHTML =
            evaluation.completeness_reasoning || "";

        renderListItems("completenessMissing", evaluation.completeness_missing_aspects);

        document.getElementById("hallucination").innerHTML =
            evaluation.hallucination_risk;

        document.getElementById("hallucinationReasoning").innerHTML =
            evaluation.hallucination_reasoning || "";

        renderListItems("hallucinationFlags", evaluation.hallucination_flagged_statements);

        renderContexts(data.retrieved_context, data.used_source_pdf);

        loadHistory();

    }

    catch (error) {

        console.error(error);

        alert("Error connecting to backend.");

    }

    finally {

        button.innerHTML = " Evaluate Response";
        button.disabled = false;

    }

}

//-------------------------------------
// Run a batch evaluation from an uploaded CSV
//-------------------------------------

let lastBatchData = null;

async function runBatchEvaluation() {

    const file = batchCsvInput.files.length > 0 ? batchCsvInput.files[0] : null;

    if (!file) {
        alert("Please choose a CSV file first.");
        return;
    }

    batchButton.innerHTML = "⏳ Running batch...";
    batchButton.disabled = true;

    try {

        const formData = new FormData();
        formData.append("csv_file", file);

        const response = await fetch(`${API_BASE}/batch-evaluate`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Batch request failed");
        }

        renderBatchResults(data);
        lastBatchData = data;

    }

    catch (error) {

        console.error(error);
        alert(`Batch evaluation failed: ${error.message}`);

    }

    finally {

        batchButton.innerHTML = "RUN BATCH";
        batchButton.disabled = false;

    }

}

//-------------------------------------
// Render batch aggregate summary + per-row results table
//-------------------------------------

function renderBatchResults(data) {

    const aggregate = data.aggregate;
    const results = data.results;

    document.getElementById("batchEmptyState").hidden = true;
    document.getElementById("batchAggregateGrid").hidden = false;
    document.getElementById("batchResultsSection").hidden = false;

    document.getElementById("batchCount").innerHTML = aggregate.count;
    document.getElementById("batchAvgAccuracy").innerHTML = formatScore(aggregate.avg_accuracy);
    document.getElementById("batchAvgRelevance").innerHTML = formatScore(aggregate.avg_relevance);
    document.getElementById("batchAvgOverall").innerHTML = formatScore(aggregate.avg_overall_score);

    const distributionText = Object.entries(aggregate.verdict_distribution || {})
        .map(([label, count]) => `${count} ${label}`)
        .join(" · ");

    document.getElementById("batchVerdictDistribution").innerHTML =
        distributionText + (aggregate.total_flagged_hallucinations
            ? ` · ${aggregate.total_flagged_hallucinations} hallucinated claim(s) flagged across the batch`
            : "");

    const tbody = document.getElementById("batchResultsBody");
    tbody.innerHTML = "";

    results.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${r.question}</td>
            <td>${r.answer}</td>
            <td>${r.verdict_label}</td>
            <td>${formatScore(r.overall_score)}</td>
            <td>${formatScore(r.accuracy)}</td>
            <td>${formatScore(r.relevance)}</td>
            <td>${formatScore(r.completeness)}</td>
            <td>${r.hallucination_risk}</td>
        `;
        tbody.appendChild(tr);
    });

}

//-------------------------------------
// Export the current batch results as a PDF report
//-------------------------------------

document.getElementById("exportReportBtn").addEventListener("click", async () => {

    if (!lastBatchData) {
        alert("Run a batch evaluation first before exporting a report.");
        return;
    }

    const exportBtn = document.getElementById("exportReportBtn");
    exportBtn.textContent = "GENERATING...";
    exportBtn.disabled = true;

    try {

        const response = await fetch(`${API_BASE}/export-report`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                results: lastBatchData.results,
                aggregate: lastBatchData.aggregate,
            }),
        });

        if (!response.ok) {
            throw new Error("Failed to generate report");
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "evaluation_report.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

    } catch (error) {

        console.error(error);
        alert("Could not generate the PDF report. Is the server running?");

    } finally {

        exportBtn.textContent = "EXPORT PDF REPORT";
        exportBtn.disabled = false;

    }

});

//-------------------------------------
// Analytics dashboard — chart instances kept here so they
// can be destroyed and recreated cleanly on every tab visit
//-------------------------------------

let dimensionChartInstance = null;
let verdictChartInstance = null;
let trendChartInstance = null;

const CHART_COLORS = {
    accent: "#cdbdf0",
    accentDark: "#ad95e0",
    green: "#8fe0a8",
    amber: "#f0c46e",
    red: "#e07a7a",
    muted: "#8f8b80",
};

async function loadAnalytics() {

    try {

        const startDate = document.getElementById("filterStartDate").value;
        const endDate = document.getElementById("filterEndDate").value;
        const mode = document.getElementById("filterMode").value;

        const params = new URLSearchParams({ limit: 500 });
        if (startDate) params.append("start_date", startDate);
        if (endDate) params.append("end_date", endDate);
        if (mode) params.append("mode", mode);

        const response = await fetch(`${API_BASE}/analytics?${params.toString()}`);

        if (!response.ok) {
            throw new Error("Failed to load analytics");
        }

        const data = await response.json();

        if (!data.total_evaluations || data.total_evaluations === 0) {
            document.getElementById("analyticsEmptyState").hidden = false;
            document.getElementById("analyticsSummaryGrid").hidden = true;
            document.getElementById("analyticsVerdictCountsGrid").hidden = true;
            document.getElementById("analyticsChartsGrid").hidden = true;
            document.getElementById("analyticsEmptyState").querySelector("p").textContent =
                "No evaluations match the current filters.";
            return;
        }

        document.getElementById("analyticsEmptyState").hidden = true;
        document.getElementById("analyticsSummaryGrid").hidden = false;
        document.getElementById("analyticsVerdictCountsGrid").hidden = false;
        document.getElementById("analyticsChartsGrid").hidden = false;

        renderAnalyticsSummary(data);
        renderDimensionChart(data.avg_scores);
        renderVerdictChart(data.verdict_distribution);
        renderTrendChart(data.trend);

    } catch (error) {

        console.error(error);
        document.getElementById("analyticsEmptyState").hidden = false;
        document.getElementById("analyticsEmptyState").querySelector("p").textContent =
            "Could not load analytics. Is the server running?";

    }

}

function renderAnalyticsSummary(data) {

    document.getElementById("analyticsTotal").innerHTML = data.total_evaluations;
    document.getElementById("analyticsPassRate").innerHTML = formatScore(data.pass_rate);
    document.getElementById("analyticsAvgOverall").innerHTML = formatScore(data.avg_scores.overall);
    document.getElementById("analyticsHallucFreq").innerHTML = formatScore(data.hallucination_frequency);

    document.getElementById("analyticsPassCount").innerHTML = data.pass_count;
    document.getElementById("analyticsNeedsImprovementCount").innerHTML = data.needs_improvement_count;
    document.getElementById("analyticsFailCount").innerHTML = data.fail_count;

}

function renderDimensionChart(avgScores) {

    const ctx = document.getElementById("dimensionChart");

    if (dimensionChartInstance) {
        dimensionChartInstance.destroy();
    }

    dimensionChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Accuracy", "Relevance", "Completeness", "Overall"],
            datasets: [{
                label: "Average Score",
                data: [
                    (avgScores.accuracy * 100).toFixed(1),
                    (avgScores.relevance * 100).toFixed(1),
                    (avgScores.completeness * 100).toFixed(1),
                    (avgScores.overall * 100).toFixed(1),
                ],
                backgroundColor: [CHART_COLORS.accent, CHART_COLORS.accent, CHART_COLORS.accent, CHART_COLORS.accentDark],
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: CHART_COLORS.muted },
                    grid: { color: "rgba(255,255,255,0.06)" },
                },
                x: {
                    ticks: { color: CHART_COLORS.muted },
                    grid: { display: false },
                },
            },
        },
    });

}

function renderVerdictChart(verdictDistribution) {

    const ctx = document.getElementById("verdictChart");

    if (verdictChartInstance) {
        verdictChartInstance.destroy();
    }

    const labelColorMap = {
        "Excellent": CHART_COLORS.green,
        "Good": CHART_COLORS.accent,
        "Needs Improvement": CHART_COLORS.amber,
        "Poor": CHART_COLORS.red,
    };

    const labels = Object.keys(verdictDistribution);
    const values = Object.values(verdictDistribution);
    const colors = labels.map((l) => labelColorMap[l] || CHART_COLORS.muted);

    verdictChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: CHART_COLORS.muted },
                },
            },
        },
    });

}

function renderTrendChart(trend) {

    const ctx = document.getElementById("trendChart");

    if (trendChartInstance) {
        trendChartInstance.destroy();
    }

    trendChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: trend.map((t) => t.date),
            datasets: [{
                label: "Avg Overall Score",
                data: trend.map((t) => (t.avg_overall_score * 100).toFixed(1)),
                borderColor: CHART_COLORS.accent,
                backgroundColor: "rgba(205,189,240,0.15)",
                fill: true,
                tension: 0.3,
                pointBackgroundColor: CHART_COLORS.accentDark,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: CHART_COLORS.muted },
                    grid: { color: "rgba(255,255,255,0.06)" },
                },
                x: {
                    ticks: { color: CHART_COLORS.muted },
                    grid: { display: false },
                },
            },
        },
    });

}

//-------------------------------------
// Render a bulleted list into any <ul> by id (used for
// both hallucination flags and completeness missing aspects)
//-------------------------------------

function renderListItems(elementId, items) {

    const list = document.getElementById(elementId);
    list.innerHTML = "";

    if (!items || items.length === 0) {
        return;
    }

    items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
    });

}

//-------------------------------------
// Color-code the verdict card border by label
//-------------------------------------

function applyVerdictColor(label) {

    const card = document.querySelector(".verdict-card");
    card.classList.remove("verdict-excellent", "verdict-good", "verdict-needs-improvement", "verdict-poor");

    const map = {
        "Excellent": "verdict-excellent",
        "Good": "verdict-good",
        "Needs Improvement": "verdict-needs-improvement",
        "Poor": "verdict-poor",
    };

    if (map[label]) {
        card.classList.add(map[label]);
    }

}

//-------------------------------------
// Render retrieved context chunks (on the Retrieval tab)
//-------------------------------------

function renderContexts(chunks, usedSourcePdf) {

    const contextDiv = document.getElementById("contexts");
    contextDiv.innerHTML = "";

    if (chunks && chunks.length > 0) {

        chunks.forEach((chunk, index) => {

            const div = document.createElement("div");
            div.className = "context-item";

            const isPdfChunk = usedSourcePdf && index === chunks.length - 1;
            const label = isPdfChunk ? "Uploaded Source PDF" : `Chunk ${index + 1}`;

            div.innerHTML = `
                <h4>${label}</h4>
                <p>${chunk}</p>
            `;

            contextDiv.appendChild(div);

        });

    } else {

        contextDiv.innerHTML = `
            <div class="context-item">
                <p>No context retrieved.</p>
            </div>
        `;

    }

}

//-------------------------------------
// Load and render evaluation history (on the History tab)
//-------------------------------------

async function loadHistory() {

    const historyList = document.getElementById("historyList");

    try {

        const response = await fetch(`${HISTORY_URL}?limit=20`);

        if (!response.ok) {
            throw new Error("Failed to load history");
        }

        const data = await response.json();
        const records = data.history || [];

        if (records.length === 0) {
            historyList.innerHTML = `
                <div class="context-item">
                    <p>No evaluations yet.</p>
                </div>
            `;
            return;
        }

        historyList.innerHTML = "";

        records.forEach((record) => {

            const div = document.createElement("div");
            div.className = "context-item";

            const refTag = record.reference_answer ? "" : " (no reference)";
            const pdfTag = record.used_source_pdf ? " · PDF used" : "";

            const flags = record.hallucination_flagged_statements || [];
            const flagsHtml = flags.length > 0
                ? `<ul class="flag-list">${flags.map(f => `<li>${f}</li>`).join("")}</ul>`
                : "";

            const missing = record.completeness_missing_aspects || [];
            const missingHtml = missing.length > 0
                ? `<ul class="flag-list">${missing.map(m => `<li>${m}</li>`).join("")}</ul>`
                : "";

            div.innerHTML = `
                <h4>${formatDate(record.created_at)} · ${record.verdict_label || ""}${refTag}${pdfTag}</h4>
                <p><strong>Q:</strong> ${record.question}</p>
                <p><strong>A:</strong> ${record.answer}</p>
                <p>
                    Accuracy: ${formatScore(record.accuracy)} ·
                    Relevance: ${formatScore(record.relevance)} ·
                    Completeness: ${formatScore(record.completeness)} ·
                    Hallucination: ${record.hallucination_risk} ·
                    Overall: ${formatScore(record.overall_score)}
                </p>
                <p class="metric-reasoning">${record.verdict_summary || ""}</p>
                <p class="metric-reasoning">${record.relevance_reasoning || ""}</p>
                <p class="metric-reasoning">${record.accuracy_evidence || ""}</p>
                <p class="metric-reasoning">${record.completeness_reasoning || ""}</p>
                ${missingHtml}
                <p class="metric-reasoning">${record.hallucination_reasoning || ""}</p>
                ${flagsHtml}
            `;

            historyList.appendChild(div);

        });

    } catch (error) {

        console.error(error);
        historyList.innerHTML = `
            <div class="context-item">
                <p>Could not load history.</p>
            </div>
        `;

    }

}

document.addEventListener("DOMContentLoaded", loadHistory);

//-------------------------------------
// Convert decimal score to percentage
//-------------------------------------

function formatScore(value) {

    if (value === undefined || value === null)
        return "--";

    if (typeof value === "string")
        return value;

    return (value * 100).toFixed(1) + "%";

}

//-------------------------------------
// Format an ISO timestamp for display
//-------------------------------------

function formatDate(isoString) {

    if (!isoString) return "Unknown time";

    const date = new Date(isoString);

    return date.toLocaleString();

}