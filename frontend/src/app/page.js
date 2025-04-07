"use client";

import { useState } from "react";
import TrainRL from "@/components/TrainRL";
import SimulateRL from "@/components/SimulateRL";
import SimulateFixed from "@/components/SimulateFixed";
import LiveLogs from "@/components/LiveLogs";
import TrainingGraph from "@/components/TrainingGraph";

export default function Home() {
  const [modelName, setModelName] = useState("traffic_model1");
  const [epochs, setEpochs] = useState(50);
  const [steps, setSteps] = useState(500);
  const [logs, setLogs] = useState("");
  const [guiRL, setGuiRL] = useState(false);
  const [guiFixed, setGuiFixed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [trainingCompleted, setTrainingCompleted] = useState(false);
  const [eventSource, setEventSource] = useState(null); // To manage the EventSource

  // --- Training Function (Modified) ---
  const handleTrain = async () => {
    setLogs("Starting RL model training and fetching live logs...\n");
    setLoading(true);
    setTrainingCompleted(false); // Reset training completed state

    // Start live logs first
    const params = new URLSearchParams({ modelName, epochs, steps });
    const newEventSource = new EventSource(`http://localhost:5000/train-stream?${params}`);
    setEventSource(newEventSource);

    newEventSource.onmessage = (event) => {
      const newLog = event.data;
      if (newLog.includes("[DONE]")) {
        newEventSource.close();
        setEventSource(null);
        setLoading(false);
        setTrainingCompleted(true);
      }
      setLogs((prevLogs) => prevLogs + newLog + "\n");
    };

    newEventSource.onerror = (err) => {
      console.error("Error receiving training logs:", err);
      newEventSource.close();
      setEventSource(null);
      setLoading(false);
      setLogs((prevLogs) => prevLogs + "\nError receiving training logs.");
    };

    try {
      const response = await fetch("http://localhost:5000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelName, epochs, steps }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Training request failed: ${errorData?.message || response.statusText}`);
      }
      const result = await response.json();
      setLogs((prev) => prev + "\nTraining initiated successfully. Monitoring live logs...");
    } catch (error) {
      console.error("Training error:", error);
      setLogs((prev) => prev + `\nError initiating training: ${error.message}`);
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
      setLoading(false);
    }
  };

  const handleSimulateRL = async () => {
    setLogs("Running SUMO RL simulation using the trained model...\n");
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelName, epochs: 1, steps, gui: guiRL }), // Reduced epochs to 1 for a single simulation run
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`RL simulation failed: ${errorData?.message || response.statusText}`);
      }
      const result = await response.json();
      const message = result?.output || result?.message || result?.errorOutput || "RL simulation completed.";
      setLogs((prev) => prev + "\n" + message);
    } catch (error) {
      console.error("RL Simulation error:", error);
      setLogs((prev) => prev + `\nError running RL simulation: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFixedSimulation = async () => {
    setLogs("Running fixed-time simulation...\n");
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/fixed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: 1, steps, gui: guiFixed }), // Ensured epochs is 1 for a single simulation
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Fixed-time simulation failed: ${errorData?.message || response.statusText}`);
      }
      const result = await response.json();
      const message = result?.output || result?.message || result?.errorOutput || "Fixed simulation completed.";
      setLogs((prev) => prev + "\n" + message);
    } catch (error) {
      console.error("Fixed-time error:", error);
      setLogs((prev) => prev + `\nError running fixed-time simulation: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-100 to-blue-50 text-gray-800">
      <div className="max-w-7xl mx-auto px-4 py-10 sm:px-6 lg:px-8">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-extrabold tracking-tight text-blue-700 sm:text-5xl">
            🚦 Intelligent Traffic Control
          </h1>
          <p className="mt-3 text-lg sm:text-xl text-gray-600">
            Reinforcement Learning Powered Smart Traffic Signals
          </p>
        </header>
  
        <main className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card title="🧠 Train RL Model" color="from-purple-200 to-purple-100">
            <TrainRL
              modelName={modelName}
              setModelName={setModelName}
              epochs={epochs}
              setEpochs={setEpochs}
              steps={steps}
              setSteps={setSteps}
              loading={loading}
              handleTrain={handleTrain}
            />
          </Card>
  
          <Card title="🤖 RL Simulation" color="from-green-200 to-green-100">
            <SimulateRL
              modelName={modelName}
              epochs={1}
              steps={steps}
              loading={loading}
              guiRL={guiRL}
              setGuiRL={setGuiRL}
              handleSimulateRL={handleSimulateRL}
            />
          </Card>
  
          <Card title="⏱️ Fixed-Time Simulation" color="from-yellow-200 to-yellow-100">
            <SimulateFixed
              loading={loading}
              guiFixed={guiFixed}
              setGuiFixed={setGuiFixed}
              handleFixedSimulation={handleFixedSimulation}
            />
          </Card>
        </main>
  
        <section className="mt-10">
          <Card title="📜 Live Logs" color="from-sky-200 to-sky-100">
            <LiveLogs logs={logs} />
          </Card>
        </section>
  
        {trainingCompleted && (
          <section className="mt-10">
            <Card title="📊 Training Graph" color="from-indigo-200 to-indigo-100">
              <TrainingGraph modelName={modelName} />
            </Card>
          </section>
        )}
  
        <footer className="mt-16 text-center text-sm text-gray-500">
          &copy; {new Date().getFullYear()} Intelligent Traffic Control. Built with 💡 and 🚦.
        </footer>
      </div>
    </div>
  );
  
}

function Card({ title, children, color = "from-white to-white" }) {
  return (
    <div
      className={`bg-gradient-to-br ${color} rounded-2xl shadow-md hover:shadow-xl transition-shadow duration-300 border border-gray-100`}
    >
      <div className="p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-3">{title}</h2>
        <div>{children}</div>
      </div>
    </div>
  );
}
