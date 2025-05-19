import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  Container,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  CircularProgress,
  Alert
} from '@mui/material';
import MovieIcon from '@mui/icons-material/Movie';
import SendIcon from '@mui/icons-material/Send';
import backgroundImage from '../images/background2.jpg'; // Adjust the path as needed


function MovieRecommendationForm() {
  const [movieTitle, setMovieTitle] = useState('');
  const [userId, setUserId] = useState(1);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const requestData = {
        movie_title: movieTitle,
        user_id: parseInt(userId)
      };
      
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
      setRecommendations(data.recommendations || []);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
         sx={{
        minHeight: '100vh',
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        display: 'flex',
        alignItems: 'center',
        py: 4,
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          zIndex: 1,
        }
      }}
    >
      <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 2 }}>
        <Paper 
          elevation={6} 
          sx={{ 
            p: 4, 
            backdropFilter: 'blur(10px)',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            borderRadius: 2 
          }}
        >
          <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
            Movie Recommendations
          </Typography>
          
          <Typography variant="body1" color="text.secondary" paragraph align="center">
            Find your next favorite movie based on your preferences
          </Typography>
          
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 3 }}>
            <TextField
              fullWidth
              label="Movie Title"
              variant="outlined"
              value={movieTitle}
              onChange={(e) => setMovieTitle(e.target.value)}
              required
              margin="normal"
              placeholder="Enter a movie you enjoyed"
            />
            
            <TextField
              fullWidth
              label="User ID"
              variant="outlined"
              type="number"
              value={userId}
              onChange={(e) => setUserId(Number(e.target.value))}
              inputProps={{ min: 1 }}
              margin="normal"
            />
            
            <Button 
              type="submit" 
              variant="contained" 
              color="primary" 
              fullWidth 
              size="large"
              disabled={loading}
              endIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
              sx={{ mt: 3, mb: 2 }}
            >
              {loading ? 'Finding Recommendations...' : 'Get Recommendations'}
            </Button>
          </Box>
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Paper>
        
        {recommendations.length > 0 && (
          <Paper 
            elevation={6} 
            sx={{ 
              p: 3, 
              mt: 3, 
              backdropFilter: 'blur(10px)',
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              borderRadius: 2 
            }}
          >
            <Typography variant="h5" component="h2" gutterBottom sx={{ color: 'primary.main' }}>
              Your Recommendations
            </Typography>
            
            <List>
              {recommendations.map((movie, index) => (
                <ListItem key={index} divider={index < recommendations.length - 1}>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'primary.main' }}>
                      <MovieIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText 
                    primary={movie} 
                    primaryTypographyProps={{ fontWeight: 'medium' }}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        )}
      </Container>
    </Box>
  );
}

export default MovieRecommendationForm;