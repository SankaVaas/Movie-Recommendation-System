import React, { useState } from "react";
import MovieForm from "./MovieForm";
import Recommendations from "./Recommendations";
import MovieRecommendationForm from './components/UserForm';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import background from "./images/background.jpg";


const theme = createTheme({
    palette: {
      primary: {
        main: '#e50914', // Netflix red
      },
      secondary: {
        main: '#ffca28', // Golden color for stars/ratings
      },
      background: {
        default: '#141414', // Dark background like streaming services
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h4: {
        fontWeight: 700,
      },
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            textTransform: 'none',
            fontWeight: 600,
            padding: '10px 0',
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
            },
          },
        },
      },
    },
  });

  
function App() {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // This function gets called when the form is submitted
  const fetchRecommendations = async (movie_title, user_id) => {
    setLoading(true);
    setError(null);
    
    try {
      // Create JSON payload instead of FormData
      const requestData = {
        movie_title: movie_title,
        user_id: parseInt(user_id)
      };
      
      console.log("Sending request:", requestData);
      
      const response = await fetch("http://localhost:8000/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestData)
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Response data:", data);
      
      setRecommendations(data.recommendations || []);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  

  return (

      <ThemeProvider theme={theme}>
        <CssBaseline /> {/* This provides a baseline CSS reset */}
        <MovieRecommendationForm onSubmit={fetchRecommendations} />

        {/* <MovieForm onSubmit={fetchRecommendations} /> */}
      
      {loading && <p>Loading recommendations...</p>}
      {error && <p style={{ color: "red" }}>Error: {error}</p>}
      
      <Recommendations recommendations={recommendations} />

      </ThemeProvider>



 
  );
}

export default App;