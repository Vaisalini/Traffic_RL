import React, { useState } from "react";
import { LineChart } from "lucide-react";

const TrainingGraph = ({ modelName }) => {
  const [isLoaded, setIsLoaded] = useState(false);

  const imageUrl = `http://localhost:5000/plots/time_vs_epoch_${modelName}.png?ts=${Date.now()}`;

  return (
    <div className="mt-8 w-full max-w-2xl mx-auto">
      <div className="bg-white p-6 rounded-2xl shadow-md hover:shadow-lg transition duration-300">
        <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <LineChart size={24} /> Training Progress
        </h2>

        {!isLoaded && (
          <div className="w-full h-64 bg-gray-100 rounded-lg flex items-center justify-center text-gray-500">
            Loading graph...
          </div>
        )}

        <img
          src={imageUrl}
          alt="Training Graph"
          onLoad={() => setIsLoaded(true)}
          className={`w-full max-h-[500px] border border-gray-300 rounded-xl shadow-sm transition-opacity duration-500 ${
            isLoaded ? "opacity-100" : "opacity-0"
          }`}
        />
      </div>
    </div>
  );
};

export default TrainingGraph;
