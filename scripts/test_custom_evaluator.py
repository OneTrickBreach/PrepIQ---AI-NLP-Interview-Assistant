"""
Script to test the custom evaluator model on a small sample of data.
"""
import os
import sys
import json
import torch
import argparse
import logging
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.custom_evaluator import CustomEvaluator, CustomEvaluatorModel
from src.schemas import Question, Answer, EvaluationMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_sample_data(data_dir, sample_size=5):
    """
    Load a small sample of questions and answers for testing.
    
    Args:
        data_dir: Directory containing the organized data
        sample_size: Number of question-answer pairs to sample
        
    Returns:
        List of (question, answer, metrics) tuples
    """
    logger.info(f"Loading sample data from {data_dir}")
    
    samples = []
    roles_dir = os.path.join(data_dir, "roles")
    
    # Get a few roles
    roles = []
    for role_file in os.listdir(roles_dir)[:3]:  # Limit to 3 roles
        if role_file.endswith(".json"):
            role = role_file.replace(".json", "")
            roles.append(role)
    
    # For each role, get some questions and answers
    for role in roles:
        questions_dir = os.path.join(data_dir, "questions", role)
        if not os.path.exists(questions_dir):
            continue
            
        # Get a few questions
        question_files = os.listdir(questions_dir)[:min(3, len(os.listdir(questions_dir)))]
        
        for question_file in question_files:
            question_path = os.path.join(questions_dir, question_file)
            with open(question_path, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
                question = Question(**question_data)
            
            # Get answers for this question
            question_id = question_file.replace(".json", "")
            answers_dir = os.path.join(data_dir, "answers", role, question_id)
            
            if not os.path.exists(answers_dir):
                continue
                
            # Get answer files (not metrics files)
            answer_files = [f for f in os.listdir(answers_dir) if f.endswith(".json") and not f.endswith("_metrics.json")]
            
            for answer_file in answer_files:
                answer_path = os.path.join(answers_dir, answer_file)
                metrics_path = os.path.join(answers_dir, answer_file.replace(".json", "_metrics.json"))
                
                # Load answer
                with open(answer_path, 'r', encoding='utf-8') as f:
                    answer_data = json.load(f)
                    answer = Answer(**answer_data)
                
                # Load metrics if available
                metrics = None
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics_data = json.load(f)
                        metrics = EvaluationMetrics(**metrics_data)
                
                samples.append((question, answer, metrics))
                
                # Stop if we have enough samples
                if len(samples) >= sample_size:
                    break
            
            # Stop if we have enough samples
            if len(samples) >= sample_size:
                break
        
        # Stop if we have enough samples
        if len(samples) >= sample_size:
            break
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples

def test_model(model, samples):
    """
    Test the model on sample data.
    
    Args:
        model: The model to test
        samples: List of (question, answer, metrics) tuples
    """
    logger.info("Testing model on sample data")
    
    results = []
    
    for question, answer, ground_truth in tqdm(samples):
        # Get model predictions
        predicted_metrics = model.evaluate_answer(question, answer)
        
        # Store results
        result = {
            "question": question.content,
            "answer": answer.content,
            "predicted": {
                "technical_accuracy": predicted_metrics.technical_accuracy,
                "completeness": predicted_metrics.completeness,
                "clarity": predicted_metrics.clarity,
                "relevance": predicted_metrics.relevance,
                "overall_score": predicted_metrics.overall_score
            }
        }
        
        # Add ground truth if available
        if ground_truth:
            result["ground_truth"] = {
                "technical_accuracy": ground_truth.technical_accuracy,
                "completeness": ground_truth.completeness,
                "clarity": ground_truth.clarity,
                "relevance": ground_truth.relevance,
                "overall_score": ground_truth.overall_score
            }
        
        results.append(result)
    
    # Print sample results
    print("\nSample results:")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        print(f"\nSample {i+1}:")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Predicted metrics: {result['predicted']}")
        if "ground_truth" in result:
            print(f"Ground truth metrics: {result['ground_truth']}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom evaluator model")
    
    parser.add_argument("--data_dir", type=str, default="organized_data", help="Directory containing the organized data")
    parser.add_argument("--sample_size", type=int, default=5, help="Number of samples to test")
    
    args = parser.parse_args()
    
    try:
        # Load sample data
        samples = load_sample_data(args.data_dir, args.sample_size)
        
        # Create model
        logger.info("Creating model")
        model = CustomEvaluatorModel()
        
        # Test model
        results = test_model(model, samples)
        
        # Save results
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, "custom_evaluator_test_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
