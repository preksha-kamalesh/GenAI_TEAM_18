const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");
const answerEl = document.getElementById("answer");
const claimsEl = document.getElementById("claims");
const docsEl = document.getElementById("docs");

let elapsedTimer = null;

function badgeClass(label) {
  const v = (label || "").toLowerCase();
  if (v.includes("entail")) return "badge-entailment";
  if (v.includes("contra")) return "badge-contradiction";
  return "badge-neutral";
}

function startElapsedTimer() {
  const start = Date.now();
  statusEl.innerHTML = `<span class="spinner"></span> Running pipeline… <span class="elapsed">0.0s</span>`;
  elapsedTimer = setInterval(() => {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const elapsedSpan = statusEl.querySelector(".elapsed");
    if (elapsedSpan) elapsedSpan.textContent = `${elapsed}s`;
  }, 100);
}

function stopElapsedTimer() {
  if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

function renderMetrics(metrics) {
  const verificationMode = metrics.verification_mode || "unknown";
  const modeLabel = verificationMode === "heuristic" ? "Lexical Heuristic" : "NLI Model";
  const modeClass = verificationMode === "heuristic" ? "mode-heuristic" : "mode-nli";

  const entries = [
    ["FactScore", `${metrics.fact_score}%`],
    ["Hallucination Rate", `${metrics.hallucination_rate}%`],
    ["Supported Claims", `${metrics.num_supported}/${metrics.num_claims}`],
    ["Contradictions", `${metrics.num_contradictions}`],
  ];

  metricsEl.innerHTML =
    `<div class="verification-mode ${modeClass}">Verification: ${modeLabel}</div>` +
    entries
      .map(
        ([k, v]) => `
      <div class="metric">
        <div class="metric-label">${k}</div>
        <div class="metric-value">${v}</div>
      </div>
    `,
      )
      .join("");
}

function renderTimings(timings) {
  if (!timings) return;
  const timingsEl = document.getElementById("timings");
  if (!timingsEl) return;

  const entries = Object.entries(timings).map(
    ([k, v]) => `<span class="timing-item"><strong>${k.replace(/_/g, " ")}:</strong> ${v}ms</span>`,
  );
  timingsEl.innerHTML = entries.join(" · ");
}

function renderClaims(claims) {
  if (!claims || claims.length === 0) {
    claimsEl.innerHTML = `<p class="empty-state">No claims extracted.</p>`;
    return;
  }
  claimsEl.innerHTML = claims
    .map(
      (c) => `
      <div class="card">
        <span class="badge ${badgeClass(c.label)}">${c.label}</span>
        <span class="doc-score">confidence ${Number(c.confidence).toFixed(3)}</span>
        <p>${c.claim}</p>
      </div>
    `,
    )
    .join("");
}

function renderDocs(docs) {
  if (!docs || docs.length === 0) {
    docsEl.innerHTML = `<p class="empty-state">No documents retrieved.</p>`;
    return;
  }
  docsEl.innerHTML = docs
    .map(
      (d) => `
      <div class="card">
        <div class="doc-score">rank ${d.rank} | score ${Number(d.score).toFixed(4)} | ${d.source}</div>
        <p>${d.text}</p>
      </div>
    `,
    )
    .join("");
}

function renderGuard(guard) {
  if (!guard || !guard.triggered) return;
  const guardEl = document.getElementById("guard");
  if (!guardEl) return;
  guardEl.innerHTML = `
    <div class="guard-warning">
      <strong>⚠ Retrieval Guard Triggered:</strong> ${guard.reason || "unknown"}
      ${guard.top_distance ? `<br>Top distance: ${guard.top_distance} (threshold: ${guard.distance_threshold})` : ""}
      ${guard.intent_coverage !== undefined ? `<br>Intent coverage: ${(guard.intent_coverage * 100).toFixed(1)}% (threshold: ${(guard.intent_threshold * 100).toFixed(1)}%)` : ""}
      ${guard.missing_terms && guard.missing_terms.length ? `<br>Missing terms: ${guard.missing_terms.join(", ")}` : ""}
    </div>
  `;
  guardEl.style.display = "block";
}

async function runPipeline() {
  const question = questionInput.value.trim();
  if (!question) {
    statusEl.textContent = "Enter a question first.";
    return;
  }

  startElapsedTimer();
  askBtn.disabled = true;

  // Hide guard
  const guardEl = document.getElementById("guard");
  if (guardEl) {
    guardEl.style.display = "none";
    guardEl.innerHTML = "";
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 3 }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const data = await response.json();
    stopElapsedTimer();

    renderMetrics(data.metrics);
    renderTimings(data.timings);
    answerEl.textContent = data.final_verified_answer;
    renderClaims(data.claims);
    renderDocs(data.retrieved_documents);
    renderGuard(data.retrieval_guard);

    const totalMs = data.timings?.total_ms;
    statusEl.textContent = totalMs
      ? `Done in ${(totalMs / 1000).toFixed(2)}s`
      : "Done.";
  } catch (err) {
    stopElapsedTimer();
    if (err.name === "AbortError") {
      statusEl.textContent = "Request timed out (60s). The server may still be starting up — try again.";
    } else {
      statusEl.textContent = `Error: ${err.message}`;
    }
  } finally {
    askBtn.disabled = false;
  }
}

askBtn.addEventListener("click", runPipeline);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runPipeline();
});

questionInput.value = "Does hypoglycaemia increase cardiovascular risk?";
