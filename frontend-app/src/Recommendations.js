import React from "react";

function Recommendations({ recommendations }) {
  if (recommendations.length === 0) return null;

  return (
    <div>
      <h2>Recommendations:</h2>
      <ul>
        {recommendations.map((title, idx) => (
          <li key={idx}>{title}</li>
        ))}
      </ul>
    </div>
  );
}

export default Recommendations;