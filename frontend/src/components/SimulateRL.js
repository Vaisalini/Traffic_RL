import React from "react";
import { BrainCircuit, PlayCircle } from "lucide-react";

const SimulateRL = ({
  modelName,
  epochs,
  steps,
  loading,
  guiRL,
  setGuiRL,
  handleSimulateRL,
}) => {
  return (
    <div className="bg-white border border-gray-200 p-6 rounded-2xl shadow-md hover:shadow-lg transition">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
        <BrainCircuit size={24} /> RL Simulation
      </h2>

      {/* Model Name */}
      <div className="mb-4">
        <label
          htmlFor="rlModelName"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Model Name
        </label>
        <input
          type="text"
          id="rlModelName"
          value={modelName}
          readOnly
          className="w-full px-4 py-2 rounded-lg border bg-gray-100 text-gray-600 cursor-not-allowed shadow-sm"
        />
      </div>

      {/* Epochs */}
      <div className="mb-4">
        <label
          htmlFor="rlEpochs"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Epochs
        </label>
        <input
          type="number"
          id="rlEpochs"
          value={epochs}
          readOnly
          className="w-full px-4 py-2 rounded-lg border bg-gray-100 text-gray-600 cursor-not-allowed shadow-sm"
        />
      </div>

      {/* Steps */}
      <div className="mb-6">
        <label
          htmlFor="rlSteps"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Steps
        </label>
        <input
          type="number"
          id="rlSteps"
          value={steps}
          readOnly
          className="w-full px-4 py-2 rounded-lg border bg-gray-100 text-gray-600 cursor-not-allowed shadow-sm"
        />
      </div>

      {/* GUI Toggle */}
      <div className="flex items-center justify-between mb-6">
        <span className="text-gray-700 font-medium">Enable SUMO GUI</span>
        <label className="inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={guiRL}
            onChange={(e) => setGuiRL(e.target.checked)}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-checked:bg-green-500 transition-all duration-300 relative">
            <span className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-all duration-300 peer-checked:translate-x-5"></span>
          </div>
        </label>
      </div>

      {/* Start Button */}
      <button
        onClick={handleSimulateRL}
        disabled={loading}
        className={`w-full flex items-center justify-center gap-2 py-2 px-4 text-white font-semibold rounded-xl transition-all duration-300 ${
          loading
            ? "bg-green-300 cursor-not-allowed"
            : "bg-green-500 hover:bg-green-600"
        }`}
      >
        {loading ? (
          <>
            <span className="animate-spin">⏳</span> Running RL Simulation...
          </>
        ) : (
          <>
            <PlayCircle size={20} /> Start RL Simulation
          </>
        )}
      </button>
    </div>
  );
};

export default SimulateRL;
