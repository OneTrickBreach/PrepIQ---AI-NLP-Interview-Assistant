import axios from 'axios';

// Create base axios instance
const api = axios.create({
  baseURL: '/api', // This will use the proxy setup in package.json
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
});

// API functions
export const ApiClient = {
  // Get list of available roles
  getRoles: async () => {
    try {
      const response = await api.get('/roles');
      return response.data;
    } catch (error) {
      console.error('Error getting roles:', error);
      throw error;
    }
  },
  
  // Generate an interview question
  generateQuestion: async (role, difficulty = 'medium', type = 'technical') => {
    try {
      // Corrected endpoint path
      // Removed explicit /api prefix, baseURL handles it
      const response = await api.post('/questions', {
        role,
        difficulty,
        type
      });
      return response.data;
    } catch (error) {
      console.error('Error generating question:', error);
      throw error;
    }
  },
  
  // Generate an answer for a question
  generateAnswer: async (questionId, content) => {
    try {
      const response = await api.post('/answers', {
        question_id: questionId,
        content
      });
      return response.data;
    } catch (error) {
      console.error('Error generating answer:', error);
      throw error;
    }
  },
  
  // Evaluate an answer
  // Added expected_skills parameter
  evaluateAnswer: async (question, answer, expected_skills) => {
    try {
      // Corrected endpoint path and added expected_skills to payload
      // Removed explicit /api prefix, baseURL handles it
      const response = await api.post('/evaluate', {
        question,
        answer,
        expected_skills
      });
      return response.data;
    } catch (error) {
      console.error('Error evaluating answer:', error);
      throw error;
    }
  },
  
  // Generate feedback for an answer
  generateFeedback: async (question, answer, metrics) => {
    try {
      // Ensure endpoint path is correct
      // Removed explicit /api prefix, baseURL handles it
      const response = await api.post('/generate-feedback', {
        question,
        answer,
        metrics
      });
      return response.data;
    } catch (error) {
      console.error('Error generating feedback:', error);
      throw error;
    }
  },
  
  // Process a complete interview question
  processQuestion: async (question, answerText, generateModelAnswer = false) => {
    try {
      const response = await api.post('/process', {
        question,
        answer_text: answerText,
        generate_model_answer: generateModelAnswer
      });
      return response.data;
    } catch (error) {
      console.error('Error processing question:', error);
      throw error;
    }
  },

  // Transcribe audio blob
  transcribeAudio: async (audioBlob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob); // Reads blob as base64 data URL
      reader.onloadend = async () => {
        try {
          // Extract base64 data (remove prefix 'data:audio/wav;base64,')
          const base64Audio = reader.result.split(',')[1];
          
          // Corrected endpoint path
          // Removed explicit /api prefix, baseURL handles it
          const response = await api.post('/transcribe', {
            audio_content: base64Audio,
            // domain: 'interview'
          });
          
          if (response.data && response.data.transcript) {
            resolve(response.data.transcript);
          } else {
            reject(new Error('Transcription failed or returned invalid format.'));
          }
        } catch (error) {
          console.error('Error transcribing audio:', error);
          reject(error);
        }
      };
      reader.onerror = (error) => {
        console.error('Error reading audio blob:', error);
        reject(error);
      };
    });
  }
  // Removed old uploadAudio function
};

export default ApiClient;
