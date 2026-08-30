// NOTE: Update this URL to your Render deployment URL once live
const API_URL = "https://network-attack-forecasting-7p0h.onrender.com/upload-csv";

document.getElementById("run-btn").addEventListener("click", async () => {
    const runBtn = document.getElementById("run-btn");
    const trajectoryDiv = document.getElementById("trajectory-output");
    const attentionDiv = document.getElementById("attention-bar-container");
    const kSteps = parseInt(document.getElementById("k-steps").value, 10);
    const fileInput = document.getElementById("csv-upload");

    if (fileInput.files.length === 0) {
        alert("Please upload a CSV file containing telemetry data.");
        return;
    }

    runBtn.innerText = "SIMULATING WORLD MODEL DYNAMICS...";
    runBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch(`${API_URL}?k_steps=${kSteps}`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText);
        }
        
        const data = await response.json();

        trajectoryDiv.innerHTML = "";
        data.trajectories.forEach((traj) => {
            const card = document.createElement("div");
            card.className = `traj-card stage-${traj.stage_id}`;
            card.innerHTML = `
                <div class="traj-time">${traj.time_offset}</div>
                <div class="traj-stage">${traj.predicted_stage}</div>
                <div class="traj-conf">${(traj.confidence * 100).toFixed(1)}% Confidence</div>
            `;
            trajectoryDiv.appendChild(card);
        });

        attentionDiv.innerHTML = "";
        const weights = data.explainability.temporal_attention_weights || [];
        weights.forEach((w, idx) => {
            const barWrapper = document.createElement("div");
            barWrapper.className = "bar-wrapper";
            barWrapper.innerHTML = `
                <div class="bar-fill" style="height: ${Math.max(w * 100, 5)}%;"></div>
                <span class="bar-label">t-${12 - idx}</span>
            `;
            attentionDiv.appendChild(barWrapper);
        });

        const latest = data.latest_features || Array(6).fill("N/A");
        document.getElementById("val-flow").innerText = typeof latest[0] === 'number' ? latest[0].toFixed(2) : latest[0];
        document.getElementById("val-byte").innerText = typeof latest[1] === 'number' ? latest[1].toFixed(2) : latest[1];
        document.getElementById("val-syn").innerText = typeof latest[2] === 'number' ? latest[2].toFixed(4) : latest[2];
        document.getElementById("val-rst").innerText = typeof latest[3] === 'number' ? latest[3].toFixed(4) : latest[3];
        document.getElementById("val-entropy").innerText = typeof latest[4] === 'number' ? latest[4].toFixed(4) : latest[4];
        document.getElementById("val-iat").innerText = typeof latest[5] === 'number' ? latest[5].toFixed(2) : latest[5];

    } catch (err) {
        trajectoryDiv.innerHTML = `<div class="error-msg">ERROR: Processing failed. Verify backend is running and CSV has at least 12 rows and 6 columns.</div>`;
        console.error(err);
    } finally {
        runBtn.innerText = "EXECUTE TRAJECTORY ROLLOUT";
        runBtn.disabled = false;
    }
});
