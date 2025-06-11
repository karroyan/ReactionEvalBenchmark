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
    
    Uses structured prompts to evaluate whether a music appraisal
    contains all required elements and provides detailed scoring.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-v3"):
        """
        Initialize the LLM evaluator.
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL
            model: Model name to use
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.scoring_prompt = self._load_scoring_prompt()

    
    def _load_scoring_prompt(self) -> str:
        """Load the scoring prompt from file."""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompt", "prompt_llm_score.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def call_api(self, user_message: str, system_prompt: str, 
                 temperature: float = 0.6, is_json: bool = True) -> Any:
        """
        Call the LLM API.
        
        Args:
            user_message: User message content
            system_prompt: System prompt
            temperature: Sampling temperature
            is_json: Whether to request JSON format
            
        Returns:
            API response content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
                stream=False,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=0.95,
                logprobs=False,
                response_format={"type": "json_object"} if is_json else None
            )
            
            content = response.choices[0].message.content
            
            if is_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON response: {content}")
                    return content
            else:
                return content
                
        except Exception as e:
            print(f"Error calling API: {e}")
            return None
    
    def evaluate_appraisal(self, appraisal_text: str) -> LLMEvaluationResult:
        """
        Evaluate a music appraisal for completeness and quality.
        
        Args:
            appraisal_text: The music appraisal text to evaluate
            
        Returns:
            LLMEvaluationResult with detailed scoring
        """
        # Prepare the user message
        user_message = f"请根据评分标准对以下歌曲评价内容进行打分：\n\n{appraisal_text}"
        
        # Call the LLM (no JSON format needed since prompt returns structured text)
        response = self.call_api(user_message, self.scoring_prompt, is_json=False)
        
        if response is None:
            # Return default result if API call failed
            return LLMEvaluationResult(
                total_score=0.0,
                max_score=16.0,
                reasoning="API call failed",
                overall_assessment="无法评估"
            )
        
        # Parse the response - response is structured text like "### **总分** \n**15.5/16分**"
        try:
            total_score = 0.0
            overall_assessment = ""
            
            # Extract total score
            if "**总分**" in response:
                score_section = response.split("**总分**")[1]
                # Look for pattern like "15.5/16分" or "15.5分"
                score_match = re.search(r'(\d+\.?\d*)', score_section)
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
            return LLMEvaluationResult(
                total_score=0.0,
                max_score=16.0,
                reasoning=f"Parse error: {str(e)}. Response: {response}",
                overall_assessment="解析失败"
            )
    
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