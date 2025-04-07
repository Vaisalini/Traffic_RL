import React from "react";
import { PlayIcon } from "lucide-react";

const SimulateFixed = ({ loading, guiFixed, setGuiFixed, handleFixedSimulation }) => {
  return (
    <div className="bg-white border border-gray-200 rounded-2xl shadow-md p-6 transition hover:shadow-lg">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center gap-2">
        ⏱️ Fixed-Time Simulation
      </h2>

      {/* Toggle switch for SUMO GUI */}
      <div className="flex items-center justify-between mb-6">
        <span className="text-gray-700 font-medium">Enable SUMO GUI</span>
        <label className="inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={guiFixed}
            onChange={(e) => setGuiFixed(e.target.checked)}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-checked:bg-orange-500 transition-all duration-300 relative">
            <span className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-all duration-300 peer-checked:translate-x-5"></span>
          </div>
        </label>
      </div>

      {/* Simulation Button */}
      <button
        onClick={handleFixedSimulation}
        disabled={loading}
        className={`w-full flex items-center justify-center gap-2 py-2 px-4 text-white font-semibold rounded-xl transition-all duration-300 ${
          loading
            ? "bg-orange-300 cursor-not-allowed"
            : "bg-orange-500 hover:bg-orange-600"
        }`}
      >
        {loading ? (
          <>
            <span className="animate-spin">⏳</span>
            Running Simulation...
          </>
        ) : (
          <>
            <PlayIcon size={18} />
            Start Fixed Simulation
          </>
        )}
      </button>
    </div>
  );
};

export default SimulateFixed;
