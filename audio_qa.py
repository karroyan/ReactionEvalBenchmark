# write a benchmark for to evaluate whether a model can answer the question based on the audio
# the model can call this code to get the score

import os
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class QAItem:
    """Represents a single question-answer item in the benchmark."""
    audio_path: str
    question: str
    options: List[str]
    answer: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Stores the results of model evaluation."""
    accuracy: float
    total_questions: int
    correct_answers: int
    detailed_results: List[Dict[str, Any]]
    average_response_time: float
    confidence_scores: Optional[List[float]] = None

class AudioQABenchmark:
    """
    A comprehensive benchmark for evaluating audio question-answering models.
    
    This benchmark can be used with any audio model that follows the specified interface.
    """
    
    def __init__(self, data_path: str, audio_base_path: Optional[str] = None):
        """
        Initialize the benchmark with data.
        
        Args:
            data_path: Path to JSONL file containing questions and answers
            audio_base_path: Base path for audio files (if different from data_path directory)
        """
        self.data_path = data_path
        self.audio_base_path = audio_base_path
        self.qa_items = self._load_data()
    
    def _load_data(self) -> List[QAItem]:
        """Load QA data from JSONL file."""
        qa_items = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Handle relative and absolute paths
                    audio_path = data['audio_path']
                    # if not os.path.isabs(audio_path):
                    #     audio_path = self.audio_base_path / audio_path
                    
                    qa_item = QAItem(
                        audio_path=str(audio_path),
                        question=data['question'],
                        options=data['options'],
                        answer=data['answer'],
                        metadata=data.get('metadata', {})
                    )
                    qa_items.append(qa_item)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(qa_items)} QA items from {self.data_path}")
        return qa_items
    
    def evaluate_model(
        self, 
        model_function: Callable[[str, str, List[str]], str],
        include_confidence: bool = False,
        confidence_function: Optional[Callable[[str, str, List[str]], Tuple[str, float]]] = None
    ) -> EvaluationResult:
        """
        Evaluate a model on the benchmark.
        
        Args:
            model_function: Function that takes (audio_path, question, options) and returns answer
            include_confidence: Whether to collect confidence scores
            confidence_function: Function that returns (answer, confidence) tuple
        
        Returns:
            EvaluationResult containing all evaluation metrics
        """
        if include_confidence and confidence_function is None:
            raise ValueError("confidence_function must be provided when include_confidence=True")
        
        correct_answers = 0
        detailed_results = []
        response_times = []
        confidence_scores = []
        
        print(f"Evaluating model on {len(self.qa_items)} questions...")
        
        for i, qa_item in enumerate(self.qa_items, 1):
            print(f"Processing question {i}/{len(self.qa_items)}")
            
            # Check if audio file exists
            if not os.path.exists(qa_item.audio_path):
                print(f"Warning: Audio file not found: {qa_item.audio_path}")
                continue
            
            try:
                start_time = time.time()
                
                if include_confidence and confidence_function:
                    predicted_answer, confidence = confidence_function(
                        qa_item.audio_path, qa_item.question, qa_item.options
                    )
                    confidence_scores.append(confidence)
                else:
                    predicted_answer = model_function(
                        qa_item.audio_path, qa_item.question, qa_item.options
                    )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Normalize answers for comparison
                predicted_normalized = self._normalize_answer(predicted_answer)
                correct_normalized = self._normalize_answer(qa_item.answer)
                
                is_correct = predicted_normalized == correct_normalized
                if is_correct:
                    correct_answers += 1
                
                # Store detailed results
                result_detail = {
                    'question_id': i - 1,
                    'audio_path': qa_item.audio_path,
                    'question': qa_item.question,
                    'options': qa_item.options,
                    'correct_answer': qa_item.answer,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response_time': response_time,
                    'metadata': qa_item.metadata
                }
                
                if include_confidence:
                    result_detail['confidence'] = confidence_scores[-1] if confidence_scores else None
                
                detailed_results.append(result_detail)
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                # Add failed result
                detailed_results.append({
                    'question_id': i - 1,
                    'audio_path': qa_item.audio_path,
                    'question': qa_item.question,
                    'error': str(e),
                    'is_correct': False,
                    'response_time': 0
                })
        
        # Calculate metrics
        total_questions = len([r for r in detailed_results if 'error' not in r])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        result = EvaluationResult(
            accuracy=accuracy,
            total_questions=total_questions,
            correct_answers=correct_answers,
            detailed_results=detailed_results,
            average_response_time=avg_response_time,
            confidence_scores=confidence_scores if confidence_scores else None
        )
        
        self._print_summary(result)
        return result
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison (handles A/B/C/D and option text)."""
        if not isinstance(answer, str):
            return str(answer).strip().upper()
        
        answer = answer.strip()
        
        # First, try to extract A/B/C/D from the response using regex
        # Look for patterns like "A", "option A", "choice A", "(A)", "A.", "A)", etc.
        patterns = [
            r'\b([ABCD])\b',  # Single letter A, B, C, or D
            r'(?:option|choice|answer|select)\s*([ABCD])\b',  # "option A", "choice B", etc.
            r'\(([ABCD])\)',  # "(A)", "(B)", etc.
            r'\b([ABCD])[.)]\s',  # "A. ", "B) ", etc.
            r'^([ABCD])[.)]\s',  # Starting with "A. " or "B) "
        ]
        
        # Try each pattern to extract the letter
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            if matches:
                # Return the first valid match (A, B, C, or D)
                first_match = matches[0].upper()
                if first_match in ['A', 'B', 'C', 'D']:
                    return first_match
        
        # If no pattern matches, try to find A/B/C/D anywhere in the string
        answer_upper = answer.upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in answer_upper:
                # Check if it's likely to be the answer choice (not part of another word)
                # Look for the letter with word boundaries or common separators
                if re.search(rf'\b{letter}\b', answer_upper):
                    return letter
        
        # As a fallback, check if the answer matches any option text directly
        # This is handled by returning the original answer for text matching
        answer_clean = answer.upper().strip()
        if answer_clean in ['A', 'B', 'C', 'D']:
            return answer_clean
            
        # If still no match, return the cleaned original answer
        # This allows for text-based matching if needed
        return answer_clean
    
    def _print_summary(self, result: EvaluationResult):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Questions: {result.total_questions}")
        print(f"Correct Answers: {result.correct_answers}")
        print(f"Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
        print(f"Average Response Time: {result.average_response_time:.2f} seconds")
        
        if result.confidence_scores:
            avg_confidence = np.mean(result.confidence_scores)
            print(f"Average Confidence: {avg_confidence:.4f}")
        
        print("="*50)
    
    def save_results(self, result: EvaluationResult, output_path: str):
        """Save detailed results to JSON file."""
        output_data = {
            'summary': {
                'accuracy': result.accuracy,
                'total_questions': result.total_questions,
                'correct_answers': result.correct_answers,
                'average_response_time': result.average_response_time
            },
            'detailed_results': result.detailed_results
        }
        
        if result.confidence_scores:
            output_data['summary']['average_confidence'] = float(np.mean(result.confidence_scores))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
    
    def get_statistics(self, result: EvaluationResult) -> Dict[str, Any]:
        """Get additional statistics from the evaluation results."""
        correct_results = [r for r in result.detailed_results if r.get('is_correct', False)]
        incorrect_results = [r for r in result.detailed_results if not r.get('is_correct', True)]
        
        stats = {
            'accuracy': result.accuracy,
            'total_questions': result.total_questions,
            'correct_answers': result.correct_answers,
            'incorrect_answers': len(incorrect_results),
            'average_response_time': result.average_response_time,
            'response_time_std': float(np.std([r.get('response_time', 0) for r in result.detailed_results])),
            'min_response_time': float(np.min([r.get('response_time', 0) for r in result.detailed_results])),
            'max_response_time': float(np.max([r.get('response_time', 0) for r in result.detailed_results]))
        }
        
        if result.confidence_scores:
            stats.update({
                'average_confidence': float(np.mean(result.confidence_scores)),
                'confidence_std': float(np.std(result.confidence_scores)),
                'min_confidence': float(np.min(result.confidence_scores)),
                'max_confidence': float(np.max(result.confidence_scores))
            })
        
        return stats


# Example usage and model interface
def example_model_function(audio_path: str, question: str, options: List[str]) -> str:
    """
    Example model function interface that any audio model should implement.
    
    Args:
        audio_path: Path to the audio file
        question: The question to answer
        options: List of possible answer options
    
    Returns:
        The predicted answer (should be one of the options or A/B/C/D)
    """
    # This is just a dummy implementation - replace with your actual model
    import random
    return random.choice(['A', 'B', 'C', 'D'])


def example_confidence_model_function(audio_path: str, question: str, options: List[str]) -> Tuple[str, float]:
    """
    Example model function that also returns confidence scores.
    
    Args:
        audio_path: Path to the audio file
        question: The question to answer
        options: List of possible answer options
    
    Returns:
        Tuple of (predicted_answer, confidence_score)
    """
    # This is just a dummy implementation - replace with your actual model
    import random
    answer = random.choice(['A', 'B', 'C', 'D'])
    confidence = random.uniform(0.1, 1.0)
    return answer, confidence


# Main evaluation function for easy usage
def run_benchmark(
    model_function: Callable[[str, str, List[str]], str],
    data_path: str = "data/option_qa.jsonl",
    output_path: Optional[str] = None,
    include_confidence: bool = False,
    confidence_function: Optional[Callable] = None
) -> EvaluationResult:
    """
    Run the benchmark evaluation with a model.
    
    Args:
        model_function: Your model function
        data_path: Path to the benchmark data
        output_path: Path to save results (optional)
        include_confidence: Whether to include confidence scoring
        confidence_function: Function that returns (answer, confidence)
    
    Returns:
        EvaluationResult with all metrics
    """
    benchmark = AudioQABenchmark(data_path)
    result = benchmark.evaluate_model(
        model_function, 
        include_confidence=include_confidence,
        confidence_function=confidence_function
    )
    
    if output_path:
        benchmark.save_results(result, output_path)
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Running example benchmark evaluation...")
    
    # Run with basic model
    result = run_benchmark(
        model_function=example_model_function,
        data_path="/fs-computility/niuyazhe/lixueyan/acapella/eval_benchmark/data/option_qa.jsonl",
        output_path="eval_results.json"
    )
    
    # Example with confidence scores
    # result_with_confidence = run_benchmark(
    #     model_function=example_model_function,
    #     include_confidence=True,
    #     confidence_function=example_confidence_model_function,
    #     output_path="eval_results_with_confidence.json"
    # ) 