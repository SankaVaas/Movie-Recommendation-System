import React, { useState } from "react";

function MovieForm({ onSubmit }) {
  const [movieTitle, setMovieTitle] = useState("");
  const [userId, setUserId] = useState(1); // Default user ID

  const handleSubmit = (e) => {
    e.preventDefault();
    if (movieTitle.trim()) {
        console.log("movie : "+ movieTitle);
      onSubmit(movieTitle, userId);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginBottom: "1.5rem" }}>
      <div>
        <label>
          Movie Title:
          <input
            type="text"
            value={movieTitle}
            onChange={(e) => setMovieTitle(e.target.value)}
            required
            style={{ marginLeft: 8 }}
          />
        </label>
      </div>
      <div style={{ marginTop: 8 }}>
        <label>
          User ID:
          <input
            type="number"
            value={userId}
            onChange={(e) => setUserId(Number(e.target.value))}
            min={1}
            style={{ marginLeft: 8, width: 60 }}
          />
        </label>
      </div>
      <button type="submit" style={{ marginTop: 12 }}>Get Recommendations</button>
    </form>
  );
}

export default MovieForm;