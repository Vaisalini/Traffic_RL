const API_URL = "http://localhost:5000"; // Your backend URL

export const startFixedSimulation = async (epochs, steps, gui) => {
  const response = await fetch(`${API_URL}/fixed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ epochs, steps, gui }),
  });
  return response.json();
};

export const startTraining = async (modelName, epochs, steps) => {
  const response = await fetch(`${API_URL}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelName, epochs, steps }),
  });
  return response.json();
};

export const startSimulation = async (modelName, epochs, steps) => {
  const response = await fetch(`${API_URL}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelName, epochs, steps }),
  });
  return response.json();
};
