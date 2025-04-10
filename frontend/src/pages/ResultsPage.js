import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Divider from '@mui/material/Divider';
import Grid from '@mui/material/Grid';
import LinearProgress from '@mui/material/LinearProgress';
import Stack from '@mui/material/Stack';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import ReactMarkdown from 'react-markdown';
import ReplayIcon from '@mui/icons-material/Replay';
import SaveIcon from '@mui/icons-material/Save';
import HomeIcon from '@mui/icons-material/Home';
import ShareIcon from '@mui/icons-material/Share';
import Chip from '@mui/material/Chip';

// Progress bar with label component
const LinearProgressWithLabel = ({ value, label }) => {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
      <Box sx={{ width: '40%', mr: 1 }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
      </Box>
      <Box sx={{ width: '45%', mr: 1 }}>
        <LinearProgress 
          variant="determinate" 
          value={value * 100} 
          sx={{ 
            height: 10, 
            borderRadius: 5,
            backgroundColor: '#e0e0e0',
            '& .MuiLinearProgress-bar': {
              backgroundColor: getColorForScore(value),
            },
          }} 
        />
      </Box>
      <Box sx={{ width: '15%' }}>
        <Typography variant="body2" color="text.secondary">
          {Math.round(value * 100)}%
        </Typography>
      </Box>
    </Box>
  );
};

// Helper function to get color based on score
const getColorForScore = (score) => {
  if (score >= 0.8) return '#4caf50'; // green
  if (score >= 0.6) return '#8bc34a'; // light green
  if (score >= 0.4) return '#ffc107'; // amber
  if (score >= 0.2) return '#ff9800'; // orange
  return '#f44336'; // red
};

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get the question, answer, and feedback from location state
  const { question, answer, feedback } = location.state || {};
  
  // Move useEffect outside the conditional
  useEffect(() => {
    // Redirect to interview page if no data
    if (!question || !feedback) {
      navigate('/interview');
    }
  }, [navigate, question, feedback]);
  
  // Return early if no data, after the useEffect
  if (!question || !feedback) {
    return null;
  }
  
  // Handle starting a new interview
  const handleNewInterview = () => {
    navigate('/interview');
  };
  
  // Handle saving results (dummy function for demo)
  const handleSaveResults = () => {
    const resultsJson = JSON.stringify({ question, answer, feedback }, null, 2);
    const blob = new Blob([resultsJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `interview-results-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };
  
  // Handle sharing results (dummy function for demo)
  const handleShareResults = () => {
    alert('Share functionality would be implemented here in a production app.');
  };
  
  return (
    <Box sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Interview Results
      </Typography>
      
      <Grid container spacing={4}>
        {/* Question Section */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Question:
              </Typography>
              <Typography variant="body1" paragraph>
                {question.content}
              </Typography>
              
              <Typography variant="subtitle2" color="text.secondary">
                Role: {question.role}
              </Typography>
              <Typography variant="subtitle2" color="text.secondary">
                Type: {question.type} | Difficulty: {question.difficulty}
              </Typography>
              
              <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {question.expected_skills.map((skill, index) => (
                  <Chip key={index} label={skill} size="small" color="primary" variant="outlined" />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Answer Section */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Your Answer:
              </Typography>
              <Typography variant="body1" paragraph sx={{ minHeight: 100 }}>
                {answer.content || "Audio response recorded"}
              </Typography>
              
              {answer.audio_url && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Audio Recording:
                  </Typography>
                  <audio src={answer.audio_url} controls></audio>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Feedback Section */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3, mt: 2 }}>
            <Typography variant="h5" gutterBottom>
              Feedback & Evaluation
            </Typography>
            
            <Grid container spacing={4}>
              <Grid item xs={12} md={7}>
                <Box sx={{ mt: 2 }}>
                  <ReactMarkdown>
                    {feedback.content}
                  </ReactMarkdown>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={5}>
                <Box sx={{ p: 3, bgcolor: '#f5f5f5', borderRadius: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Performance Metrics
                  </Typography>
                  
                  <Box sx={{ mt: 3 }}>
                    <LinearProgressWithLabel 
                      value={feedback.metrics.technical_accuracy} 
                      label="Technical Accuracy" 
                    />
                    <LinearProgressWithLabel 
                      value={feedback.metrics.completeness} 
                      label="Completeness" 
                    />
                    <LinearProgressWithLabel 
                      value={feedback.metrics.clarity} 
                      label="Clarity" 
                    />
                    <LinearProgressWithLabel 
                      value={feedback.metrics.relevance} 
                      label="Relevance" 
                    />
                    
                    <Divider sx={{ my: 2 }} />
                    
                    <LinearProgressWithLabel 
                      value={feedback.metrics.overall_score} 
                      label="Overall Score" 
                    />
                  </Box>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
      
      {/* Action Buttons */}
      <Stack 
        direction={{ xs: 'column', sm: 'row' }} 
        spacing={2} 
        justifyContent="center" 
        sx={{ mt: 4 }}
      >
        <Button 
          variant="contained" 
          color="primary" 
          startIcon={<ReplayIcon />}
          onClick={handleNewInterview}
        >
          Practice Another Question
        </Button>
        <Button 
          variant="outlined" 
          startIcon={<SaveIcon />}
          onClick={handleSaveResults}
        >
          Save Results
        </Button>
        <Button 
          variant="outlined" 
          startIcon={<ShareIcon />}
          onClick={handleShareResults}
        >
          Share Results
        </Button>
        <Button 
          variant="outlined" 
          startIcon={<HomeIcon />}
          onClick={() => navigate('/')}
        >
          Back to Home
        </Button>
      </Stack>
    </Box>
  );
};

export default ResultsPage;
