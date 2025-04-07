// components/LiveLogs.js
import React from "react";

const LiveLogs = ({ logs }) => {
  return (
    <div className="mt-6 w-full max-w-3xl mx-auto">
      <div className="bg-gray-900 border border-green-500 rounded-2xl shadow-xl overflow-hidden">
        <div className="bg-gradient-to-r from-green-400 to-green-600 p-3 sticky top-0 z-10">
          <h2 className="text-white font-bold text-lg">📡 Live Logs</h2>
        </div>
        <pre className="p-4 text-green-300 font-mono text-sm overflow-y-auto h-72 scrollbar-thin scrollbar-thumb-green-500 scrollbar-track-gray-800 whitespace-pre-wrap">
          {logs.length > 0 ? logs : "Waiting for logs..."}
        </pre>
      </div>
    </div>
  );
};

export default LiveLogs;
