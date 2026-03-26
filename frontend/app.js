const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");
const answerEl = document.getElementById("answer");
const claimsEl = document.getElementById("claims");
const docsEl = document.getElementById("docs");

function badgeClass(label) {
  const v = (label || "").toLowerCase();
  if (v.includes("entail")) return "badge-entailment";
  if (v.includes("contra")) return "badge-contradiction";
  return "badge-neutral";
}

function renderMetrics(metrics) {
  const entries = [
    ["FactScore", `${metrics.fact_score}%`],
    ["Hallucination Rate", `${metrics.hallucination_rate}%`],
    ["Supported Claims", `${metrics.num_supported}/${metrics.num_claims}`],
    ["Contradictions", `${metrics.num_contradictions}`],
  ];

  metricsEl.innerHTML = entries
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

function renderClaims(claims) {
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

async function runPipeline() {
  const question = questionInput.value.trim();
  if (!question) {
    statusEl.textContent = "Enter a question first.";
    return;
  }

  statusEl.textContent = "Running pipeline...";
  askBtn.disabled = true;

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 3 }),
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const data = await response.json();
    renderMetrics(data.metrics);
    answerEl.textContent = data.final_verified_answer;
    renderClaims(data.claims);
    renderDocs(data.retrieved_documents);
    statusEl.textContent = "Done.";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    askBtn.disabled = false;
  }
}

askBtn.addEventListener("click", runPipeline);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runPipeline();
});

questionInput.value = "Does hypoglycaemia increase cardiovascular risk?";
