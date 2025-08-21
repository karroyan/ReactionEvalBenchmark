#!/usr/bin/env python3
"""
LLM-based evaluator for music appraisal completeness and quality.

This module uses an LLM to evaluate whether a music appraisal contains
all the required elements according to the structured scoring criteria.
"""

import json
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from openai import AzureOpenAI


@dataclass
class LLMEvaluationResult:
    """Results from LLM evaluation of music appraisal."""
    total_score: float
    max_score: float
    reasoning: str
    overall_assessment: str


class AudioLLMEvaluator:
    """
    LLM-based evaluator for music appraisal completeness.
    
    Uses multiple LLMs with structured prompts to evaluate whether a music appraisal
    contains all required elements and provides detailed scoring.
    """
    
    def __init__(self, clients: Dict[str, Any], models: List[str] = None, model_weights: Dict[str, float] = None):
        """
        Initialize the LLM evaluator with multiple models and their clients.
        
        Args:
            clients: Dictionary mapping model names to their API clients (OpenAI, AzureOpenAI, Ark, etc.)
            models: List of model names to use. If None, uses all clients' keys
            model_weights: Dictionary mapping model names to their weights for aggregation.
                         If None, uses equal weights for all models.
        """
        self.clients = clients
        self.models = models if models else list(clients.keys())
        self.model_weights = model_weights if model_weights else {model: 1.0 for model in self.models}
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.model_weights.values())
        self.model_weights = {k: v/weight_sum for k, v in self.model_weights.items()}
        
        self.scoring_prompt = self._load_scoring_prompt()
        
        # Conservative sampling parameters for stable evaluation
        self.sampling_params = {
            "temperature": 0,  # Very low temperature for consistency
            "top_p": 0.5,       # More focused sampling
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 4096,
            "stream": False,
            "logprobs": False
        }

    def _load_scoring_prompt(self) -> str:
        """Load the scoring prompt from file."""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompt", "prompt_llm_score.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def call_api(self, user_message: str, system_prompt: str, model: str, is_json: bool = True) -> Any:
        """
        Call a specific LLM API.
        
        Args:
            user_message: User message content
            system_prompt: System prompt
            model: Model name to use
            is_json: Whether to request JSON format
            
        Returns:
            API response content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        try:
            client = self.clients[model]
            # All clients should have a similar chat.completions.create interface
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **{
                    **self.sampling_params,
                    "response_format": {"type": "json_object"} if is_json else None
                }
            )
            content = response.choices[0].message.content
            
            if is_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON response from model {model}: {content}")
                    return content
            else:
                return content
                
        except Exception as e:
            print(f"Error calling API for model {model}: {e}")
            return None

    def evaluate_appraisal(self, appraisal_text: str) -> LLMEvaluationResult:
        """
        Evaluate a music appraisal using multiple LLMs.
        
        Args:
            appraisal_text: The music appraisal text to evaluate
            
        Returns:
            Aggregated LLMEvaluationResult from multiple models
        """
        # Prepare the user message
        user_message = f"请根据评分标准对以下歌曲评价内容进行打分：\n\n{appraisal_text}"
        
        # Collect results from all models
        results = []
        for model in self.models:
            response = self.call_api(user_message, self.scoring_prompt, model, is_json=False)
            if response is not None:
                try:
                    result = self._parse_llm_response(response)
                    results.append(result)
                except Exception as e:
                    print(f"Error parsing response from model {model}: {e}")
        
        # If no valid results, return default
        if not results:
            return self._get_default_result("All model calls failed")
        
        # Aggregate results
        return self._aggregate_results(results)

    def _get_default_result(self, error_message: str) -> LLMEvaluationResult:
        """Return default result with error message."""
        return LLMEvaluationResult(
            total_score=0.0,
            max_score=16.0,
            reasoning=error_message,
            overall_assessment="无法评估"
        )

    def _aggregate_results(self, results: List[LLMEvaluationResult]) -> LLMEvaluationResult:
        """
        Aggregate results from multiple models using weighted average.
        Models with higher weights have more influence on the final score.
        """
        if not results:
            return self._get_default_result("No valid results to aggregate")

        # Calculate weighted average score
        weighted_score = 0.0
        total_weight = 0.0
        
        for model, result in zip(self.models[:len(results)], results):
            weight = self.model_weights.get(model, 1.0)
            weighted_score += result.total_score * weight
            total_weight += weight

        avg_total_score = weighted_score / total_weight if total_weight > 0 else 0.0
        max_score = results[0].max_score  # Should be the same for all results
        
        # Combine reasoning and assessments with model weights
        combined_reasoning = "Aggregated analysis (weighted by model confidence):\n\n"
        for model, result in zip(self.models[:len(results)], results):
            weight = self.model_weights.get(model, 1.0)
            combined_reasoning += f"Model {model} (weight: {weight:.2f}):\n{result.reasoning}\n\n---\n\n"
        
        # Combine overall assessments with weights
        combined_assessment = "Weighted assessments:\n" + "\n".join([
            f"Model {model} (weight: {self.model_weights.get(model, 1.0):.2f}): {result.overall_assessment}"
            for model, result in zip(self.models[:len(results)], results)
        ])
        
        return LLMEvaluationResult(
            total_score=avg_total_score,
            max_score=max_score,
            reasoning=combined_reasoning,
            overall_assessment=combined_assessment
        )

    def _parse_llm_response(self, response: str) -> LLMEvaluationResult:
        """Parse the LLM evaluation response."""
        try:
            total_score = 0.0
            overall_assessment = ""
            
            # Extract total score
            score_match = re.search(r'\*\*总分\*\*?\s*[：:]*\s*(\d+\.?\d*)\s*(?:/\s*\d+\.?\d*)?\s*分?', response)
            if score_match:
                total_score = float(score_match.group(1))
            
            # Extract overall assessment
            if "**总体评价**" in response:
                assessment_section = response.split("**总体评价**")[1]
                # Take text until next ** marker or end
                assessment_match = re.search(r'[：:]\s*([^*]+?)(?:\*\*|$)', assessment_section)
                if assessment_match:
                    overall_assessment = assessment_match.group(1).strip()
                else:
                    # Fallback: take first line after 总体评价
                    lines = assessment_section.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('**'):
                            overall_assessment = line.strip()
                            break
            
            return LLMEvaluationResult(
                total_score=total_score,
                max_score=16.0,
                reasoning=response,  # Store full response as reasoning
                overall_assessment=overall_assessment
            )
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._get_default_result(f"Parse error: {str(e)}. Response: {response}")
    
    def calculate_category_scores(self, detailed_scores: Dict) -> Dict[str, float]:
        """
        Calculate scores for each main category.
        
        This method is deprecated since detailed scores are no longer used.
        Returns empty dict for compatibility.
        """
        return {}
    
    def get_completeness_score(self, detailed_scores: Dict) -> float:
        """
        Calculate a completeness score (0-1) based on how many elements are present.
        
        This method is deprecated since detailed scores are no longer used.
        Returns 0.0 for compatibility.
        """
        return 0.0 