import React from "react";
import { BrainCircuit } from "lucide-react";

const TrainRL = ({
  modelName,
  setModelName,
  epochs,
  setEpochs,
  steps,
  setSteps,
  loading,
  handleTrain,
}) => {
  return (
    <div className="w-full max-w-2xl mx-auto bg-white rounded-2xl shadow-md p-6 hover:shadow-lg transition">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
        <BrainCircuit size={24} /> Train RL Model
      </h2>

      <div className="mb-4">
        <label className="block text-sm font-semibold text-gray-700 mb-1">Model Name:</label>
        <input
          type="text"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
          placeholder="Enter model name"
        />
      </div>

      <div className="mb-4">
        <label className="block text-sm font-semibold text-gray-700 mb-1">Epochs:</label>
        <input
          type="number"
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
          placeholder="e.g. 100"
        />
      </div>

      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-1">Steps:</label>
        <input
          type="number"
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
          placeholder="e.g. 500"
        />
      </div>

      <button
        onClick={handleTrain}
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg
              className="animate-spin h-5 w-5 text-white"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8z"
              />
            </svg>
            Training...
          </span>
        ) : (
          "Start RL Training"
        )}
      </button>
    </div>
  );
};

export default TrainRL;
