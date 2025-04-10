from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import base64 # Import base64 module
from src.speech_to_text import SpeechToText
from src.schemas import Question, Answer, EvaluationMetrics, Feedback # Removed RoleDetails import
from config.config import settings
from src.models.custom_evaluator import CustomEvaluatorModel
from src.feedback.generator import FeedbackGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
# Provide the GCS bucket name to SpeechToText
GCS_BUCKET_NAME = "nlp-project-interview-audio-v2"
stt = SpeechToText(bucket_name=GCS_BUCKET_NAME)
# Initialize model (will automatically load latest from models/ if available)
custom_model = CustomEvaluatorModel()
feedback_generator = FeedbackGenerator()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-NLP Interview Assistant API"
)

class TranscriptionRequest(BaseModel):
    audio_content: str
    domain: str = "general"

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    expected_skills: List[str]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    metrics: Dict[str, float]

# --- Request Schemas ---
class QuestionRequest(BaseModel):
    role: str
    difficulty: str = 'medium'
    type: str = 'technical'

@app.post("/api/transcribe") # Added /api prefix
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcribe audio content to text.
    """
    try:
        # Decode base64 audio content to bytes
        try:
            # Ensure padding is correct if needed, though standard base64 usually includes it
            # Remove potential data URL prefix if frontend sends it (safer)
            if "," in request.audio_content:
                base64_audio = request.audio_content.split(',')[1]
            else:
                base64_audio = request.audio_content
            audio_bytes = base64.b64decode(base64_audio)
        except Exception as decode_error:
            logger.error(f"Base64 decoding error: {decode_error}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Transcribe audio
        transcript = stt.transcribe_audio(audio_bytes, request.domain)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Transcription failed")
            
        return {"transcript": transcript}
        
    except HTTPException as http_exc: # Keep specific HTTP exceptions
        raise http_exc
    except Exception as e:
        # Log the specific exception type and message from transcribe_audio
        logger.exception(f"Transcription failed in main handler: {type(e).__name__}: {e}")
        # Return a more specific error message if possible
        detail_msg = f"Transcription failed: {e}" if str(e) else "Transcription failed due to an internal error."
        # Consider returning 400 if it was a client-side issue like bad audio/GCS error
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        raise HTTPException(status_code=status_code, detail=detail_msg)

@app.post("/api/evaluate") # Added /api prefix
async def evaluate_answer(request: EvaluationRequest):
    """
    Evaluate an answer and return metrics.
    """
    logger.info(f"Received evaluation request for question: {request.question[:50]}...") # Log first 50 chars
    try:
        logger.info("Calling custom_model.evaluate_answer...")
        # Create Question/Answer objects with placeholders for missing required fields
        temp_question_id = "temp_q_id" # Placeholder
        temp_answer_id = "temp_a_id"   # Placeholder
        
        question_obj = Question(
            id=temp_question_id,
            role="unknown", # Placeholder
            type="unknown", # Placeholder
            difficulty="unknown", # Placeholder
            content=request.question
        )
        answer_obj = Answer(
            id=temp_answer_id,
            question_id=temp_question_id,
            content=request.answer
        )
        
        metrics_obj = custom_model.evaluate_answer(question_obj, answer_obj)
        # Removed stray parenthesis from previous line
        logger.info(f"Evaluation successful. Metrics object type: {type(metrics_obj)}")

        # Convert EvaluationMetrics object to dict if needed
        if isinstance(metrics_obj, dict):
            metrics = metrics_obj
        elif hasattr(metrics_obj, 'dict'):
             metrics = metrics_obj.dict()
        elif hasattr(metrics_obj, '__dict__'):
             metrics = metrics_obj.__dict__ # Fallback if no .dict()
        else:
             logger.warning("Metrics object is not a dict and has no dict() method. Trying direct conversion.")
             metrics = dict(metrics_obj) # Attempt direct conversion

        logger.info(f"Returning metrics: {metrics}")
        return {"metrics": metrics}

    except Exception as e:
        logger.exception(f"Evaluation error occurred: {str(e)}") # Use logger.exception for traceback
        raise HTTPException(status_code=500, detail=f"Failed to evaluate answer: {str(e)}")

# --- Mock Question Data (Placeholder) ---
MOCK_QUESTIONS = {
    "technical": {
        "Software Engineer": [
            "Explain the difference between depth-first search and breadth-first search.",
            "What are the principles of object-oriented programming?",
            "Explain time and space complexity."
        ],
        "Data Scientist": [
            "Describe how you would handle imbalanced datasets.",
            "Explain the difference between supervised and unsupervised learning.",
            "How would you deal with missing data?"
        ],
         "Default": [
            "What are key technical skills for this role?",
            "Describe a challenging technical problem you solved.",
            "How do you stay updated with technologies?"
        ]
    },
    "behavioral": {
         "Default": [
            "Describe a challenging project and how you overcame obstacles.",
            "Tell me about working under pressure.",
            "Give an example of learning a new technology quickly."
         ]
    }
}

MOCK_SKILLS = {
    "Software Engineer": ["Algorithms", "Data Structures", "Problem Solving", "OOP"],
    "Data Scientist": ["Machine Learning", "Statistics", "Data Analysis", "Python"],
    "Default": ["Technical Knowledge", "Problem Solving", "Communication"]
}

import random
import datetime

@app.post("/api/questions", response_model=Question) # Added /api prefix
async def generate_question_endpoint(request: QuestionRequest):
    """
    Generate an interview question based on role, difficulty, and type.
    (Currently uses placeholder logic)
    """
    logger.info(f"Generating question for role: {request.role}, type: {request.type}")
    try:
        q_type = request.type if request.type in MOCK_QUESTIONS else "technical"
        role_key = request.role if request.role in MOCK_QUESTIONS[q_type] else "Default"
        
        # Handle case where role might exist in technical but not behavioral
        if role_key == "Default" and request.role in MOCK_QUESTIONS.get("technical", {}):
             role_key = request.role # Use specific role if available in technical
        elif role_key == "Default" and q_type == "behavioral":
             role_key = "Default" # Behavioral often uses default

        question_list = MOCK_QUESTIONS[q_type].get(role_key, MOCK_QUESTIONS[q_type]["Default"])
        content = random.choice(question_list)
        
        skills_role_key = request.role if request.role in MOCK_SKILLS else "Default"
        skills = random.sample(MOCK_SKILLS.get(skills_role_key, MOCK_SKILLS["Default"]), k=min(3, len(MOCK_SKILLS.get(skills_role_key, MOCK_SKILLS["Default"]))))

        question_obj = Question(
            id=f"q_{random.randint(1000, 9999)}", # Generate random ID
            role=request.role,
            type=request.type,
            difficulty=request.difficulty,
            content=content,
            expected_skills=skills,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return question_obj
    except Exception as e:
        logger.error(f"Question generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate question")


@app.post("/api/generate-feedback") # Added /api prefix
async def generate_feedback(request: FeedbackRequest):
    """
    Generate feedback based on evaluation metrics.
    """
    logger.info(f"Received feedback request for question: {request.question[:50]}...") # Log first 50 chars
    logger.info(f"Metrics received: {request.metrics}")
    try:
        logger.info("Calling feedback_generator.generate_feedback...")
        # Ensure metrics are passed correctly, might need conversion if not already dict
        try:
            metrics_for_feedback = EvaluationMetrics(**request.metrics)
        except Exception as pydantic_error:
             logger.error(f"Error creating EvaluationMetrics from request data: {pydantic_error}")
             raise HTTPException(status_code=400, detail="Invalid metrics format provided.")

        # Create Question/Answer objects with placeholders for missing required fields
        temp_question_id = "temp_q_id" # Placeholder
        temp_answer_id = "temp_a_id"   # Placeholder

        question_obj = Question(
            id=temp_question_id,
            role="unknown", # Placeholder
            type="unknown", # Placeholder
            difficulty="unknown", # Placeholder
            content=request.question
        )
        answer_obj = Answer(
            id=temp_answer_id,
            question_id=temp_question_id,
            content=request.answer
        )

        feedback_obj = feedback_generator.generate_feedback(
            question_obj,
            answer_obj,
            metrics_for_feedback
        )
        logger.info(f"Feedback generation successful. Feedback object type: {type(feedback_obj)}")

        # Convert Feedback object to dict if needed
        if isinstance(feedback_obj, dict):
            feedback = feedback_obj
        elif hasattr(feedback_obj, 'dict'):
             feedback = feedback_obj.dict()
        elif hasattr(feedback_obj, '__dict__'):
             feedback = feedback_obj.__dict__ # Fallback if no .dict()
        else:
             logger.warning("Feedback object is not a dict and has no dict() method. Trying direct conversion.")
             feedback = dict(feedback_obj) # Attempt direct conversion

        logger.info(f"Returning feedback: {feedback}")
        return {"feedback": feedback}

    except Exception as e:
        logger.exception(f"Feedback generation error occurred: {str(e)}") # Use logger.exception for traceback
        raise HTTPException(status_code=500, detail=f"Failed to generate feedback: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "version": settings.VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
