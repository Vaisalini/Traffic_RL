const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const cors = require("cors");

const app = express();
const PORT = 5000;

app.use(express.json());
app.use(cors());

app.use("/plots", express.static(path.join(__dirname, "..", "model", "plots")));
// Path to the model directory
const MODEL_DIR = path.join(__dirname, "..", "model");

// Function to safely parse numeric values from Python output
function parseMetrics(output) {
    try {
        const lines = output.trim().split('\n');
        const metrics = {};
        for (const line of lines) {
            const [key, value] = line.split(':').map(item => item.trim());
            if (key && value && !isNaN(parseFloat(value))) {
                metrics[key] = parseFloat(value);
            }
        }
        return metrics;
    } catch (error) {
        console.error("Error parsing metrics:", error);
        return {};
    }
}

app.post("/fixed", (req, res) => {
    const { epochs: receivedEpochs, steps, gui } = req.body; // Rename the received epochs
    const epochs = 1; // Force epochs to be 1 for fixed simulation

    if (!receivedEpochs || !steps) { // Keep the original check for required parameters
        return res.status(400).json({ error: "Missing required parameters" });
    }

    const fixedProcess = spawn("python", [
        path.join(MODEL_DIR, "fixed.py"),
        "-e", epochs,
        "-s", steps,
        ...(gui ? ["--gui"] : [])
    ], { cwd: MODEL_DIR });

    let fixedOutput = "";
    let fixedError = "";

    fixedProcess.stdout.on("data", (data) => {
        const outputChunk = data.toString();
        fixedOutput += outputChunk;
        console.log(`Fixed Time Simulation Output: ${outputChunk}`);
    });

    fixedProcess.stderr.on("data", (data) => {
        const errorChunk = data.toString();
        fixedError += errorChunk;
        console.error(`Fixed Time Simulation Error: ${errorChunk}`);
    });

    fixedProcess.on("close", (code) => {
        console.log(`Fixed Time Simulation process exited with code ${code}`);
        const metrics = parseMetrics(fixedOutput);
        res.json({
            message: "Fixed Time Simulation process completed",
            exitCode: code,
            output: fixedOutput,
            errorOutput: fixedError,
            metrics: metrics // Include the parsed metrics
        });
    });
});


// Train RL Model
app.post("/train", (req, res) => {
    const { modelName, epochs, steps } = req.body;

    console.log(`[INFO] Training started for ${modelName} with ${epochs} epochs and ${steps} steps.`);

    const trainProcess = spawn("python", [
        path.join(MODEL_DIR, "train_RL.py"),
        "--train",
        "-m", modelName,
        "-e", epochs,
        "-s", steps
    ], { cwd: MODEL_DIR });

    trainProcess.on("close", (code) => {
        console.log(`[INFO] Training completed with exit code ${code}`);
        res.json({ message: `Training completed with exit code ${code}` });
    });

    trainProcess.on("error", (error) => {
        console.error("[ERROR] Training process failed:", error);
        res.status(500).json({ message: "Training failed" });
    });
});

app.get("/train-stream", (req, res) => {
    const modelName = req.query.modelName || "default_model";
    const epochs = req.query.epochs || "5";
    const steps = req.query.steps || "5";

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const trainProcess = spawn("python", [
        path.join(MODEL_DIR, "train_RL.py"),
        "--train",
        "-m", modelName,
        "-e", epochs,
        "-s", steps
    ], { cwd: MODEL_DIR });

    trainProcess.stdout.on("data", (data) => {
        res.write(`data: ${data.toString().trim()}\n\n`);
    });

    trainProcess.stderr.on("data", (data) => {
        // Send the raw stderr data without adding "[ERROR]" prefix
        res.write(`data: ${data.toString()}\n\n`);
        console.error(`[TRAIN STDERR] ${data.toString()}`); // Keep logging to the server console
    });

    trainProcess.on("close", (code) => {
        res.write(`data: [DONE] Training completed with exit code ${code}\n\n`);
        res.end();
    });

    req.on("close", () => {
        trainProcess.kill();
        res.end();
    });
});


// Run RL-based SUMO Simulation
app.post("/simulate", (req, res) => {
    const { modelName, epochs, steps, gui } = req.body; // Receive the 'gui' parameter
    if (!modelName || !epochs || !steps) {
        return res.status(400).json({ error: "Missing required parameters" });
    }

    const pythonArgs = [
        path.join(MODEL_DIR, "train_RL.py"),
        "-m", modelName,
        "-e", epochs,
        "-s", steps,
        ...(gui ? ["--gui"] : []) // Conditionally add the --gui flag
    ];

    const simProcess = spawn("python", pythonArgs, { cwd: MODEL_DIR });

    let simulationOutput = "";
    let simulationError = "";

    simProcess.stdout.on("data", (data) => {
        const outputChunk = data.toString();
        simulationOutput += outputChunk;
        console.log(`[PYTHON STDOUT] ${outputChunk}`); // Log each chunk immediately
    });

    simProcess.stderr.on("data", (data) => {
        const errorChunk = data.toString();
        simulationError += errorChunk;
        console.error(`[PYTHON STDERR] ${errorChunk}`); // Log each chunk immediately
    });

    simProcess.on("close", (code) => {
        console.log(`Simulation process exited with code ${code}`);
        const metrics = parseMetrics(simulationOutput);
        res.json({
            message: "Simulation process completed",
            exitCode: code,
            output: simulationOutput,
            errorOutput: simulationError,
            metrics: metrics // Include the parsed metrics
        });
    });

    simProcess.on("error", (err) => {
        console.error("[SPAWN ERROR]", err);
        res.status(500).json({ message: "Failed to spawn simulation process", error: err.message });
    });
});
// Start Server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});