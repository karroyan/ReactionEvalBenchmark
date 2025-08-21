#!/usr/bin/env python3
"""
Precision evaluator for music appraisal factual accuracy.

This module uses an LLM to compare model outputs with ground truth song details
and evaluate factual accuracy.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from openai import AzureOpenAI


@dataclass
class PrecisionEvaluationResult:
    """Results from precision evaluation of music appraisal."""
    precision_score: float  # 0-1 score for factual accuracy
    total_claims: int  # Total number of factual claims made
    correct_claims: int  # Number of correct factual claims
    incorrect_claims: int  # Number of incorrect factual claims
    detailed_analysis: str  # LLM's detailed analysis
    claim_breakdown: List[Dict[str, Any]]  # List of individual claim evaluations


class AudioPrecisionEvaluator:
    """
    Precision evaluator for music appraisal factual accuracy.
    
    Uses multiple LLMs to compare model outputs with ground truth and evaluate
    the accuracy of factual claims made in the appraisal.
    """
    
    def __init__(self, clients: Dict[str, Any], models: List[str] = None, model_weights: Dict[str, float] = None):
        """
        Initialize the precision evaluator with multiple models and their clients.
        
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
        
        self.evaluation_prompt = self._get_precision_prompt()
        
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

    def _get_precision_prompt(self) -> str:
        """Get the precision evaluation prompt."""
        return """### **System Prompt**  
**角色**：你是一位专业的音乐事实核查员，负责验证音乐评价内容中的事实准确性。  
**任务**：对比音乐评价文本与歌曲真实信息，判断评价中提到的具体事实是否正确。重点关注准确性，而非完整性。

### **评估标准**  
- **只评估明确提及的事实信息**，不要求评价包含所有信息
- **重点关注准确性**：提到的信息是否与真实情况一致
- **包括但不限于**：音乐风格、歌曲描述、歌曲主题、创作背景、音乐风格细分、嗓音特点、MV概念、风格/氛围、编曲/细节、作曲/结构、嗓音描述、情感表达、演唱技巧、歌手背景关联、歌曲背景/文化关联、流行趋势/亚文化洞察等信息
- **忽略主观感受**：如"好听"、"感动"等个人观点不算事实错误

### **评分方式**
1. **识别所有事实性陈述**：从评价文本中提取具体的事实声明，只评估明确提及的部分
2. **逐项核实**：对比每个事实与真实信息是否一致
3. **计算准确率**：正确事实数量 / 总事实数量

### **输出格式要求**  
请按以下格式输出评估结果：

**事实核查分析**：
[逐项列出发现的事实声明，标注是否正确]

**准确性统计**：
- 总计事实声明：X条
- 正确事实：X条  
- 错误事实：X条
- 准确率：X%

**总体评价**：
[一句话总结事实准确性表现]"""

    def call_api(self, user_message: str, system_prompt: str, model: str) -> str:
        """
        Call a specific LLM API for precision evaluation.
        
        Args:
            user_message: User message content
            system_prompt: System prompt
            model: Model name to use
            
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
                **self.sampling_params
            )
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error calling API for model {model}: {e}")
            return None

    def evaluate_precision(self, appraisal_text: str, 
                         ground_truth: Dict[str, Any]) -> PrecisionEvaluationResult:
        """
        Evaluate the factual precision using multiple LLMs.
        
        Args:
            appraisal_text: The music appraisal text to evaluate
            ground_truth: Ground truth information about the song
            
        Returns:
            Aggregated PrecisionEvaluationResult from multiple models
        """
        # Format ground truth information for comparison
        truth_info = self._format_ground_truth(ground_truth)
        
        # Prepare the user message
        user_message = f"""请核查以下音乐评价的事实准确性：

**评价文本**：
{appraisal_text}

**真实歌曲信息**：
{truth_info}

请按照评估标准进行事实核查分析。"""
        
        # Collect results from all models
        results = []
        for model in self.models:
            response = self.call_api(user_message, self.evaluation_prompt, model)
            if response is not None:
                try:
                    result = self._parse_precision_response(response)
                    results.append(result)
                except Exception as e:
                    print(f"Error parsing response from model {model}: {e}")
        
        # If no valid results, return default
        if not results:
            return self._get_default_result("All model calls failed")
        
        # Aggregate results
        return self._aggregate_results(results)

    def _get_default_result(self, error_message: str) -> PrecisionEvaluationResult:
        """Return default result with error message."""
        return PrecisionEvaluationResult(
            precision_score=0.0,
            total_claims=0,
            correct_claims=0,
            incorrect_claims=0,
            detailed_analysis=error_message,
            claim_breakdown=[]
        )

    def _aggregate_results(self, results: List[PrecisionEvaluationResult]) -> PrecisionEvaluationResult:
        """
        Aggregate results from multiple models using weighted average and voting.
        For factual claims, uses a voting mechanism where models with higher weights have more voting power.
        """
        if not results:
            return self._get_default_result("No valid results to aggregate")

        # Calculate weighted average scores
        weighted_precision = 0.0
        total_weight = 0.0
        
        for model, result in zip(self.models[:len(results)], results):
            weight = self.model_weights.get(model, 1.0)
            weighted_precision += result.precision_score * weight
            total_weight += weight

        avg_precision = weighted_precision / total_weight if total_weight > 0 else 0.0
        
        # Use weighted voting for claims
        claim_votes = {}  # {claim_text: {True: total_weight_for_true, False: total_weight_for_false}}
        
        for model, result in zip(self.models[:len(results)], results):
            weight = self.model_weights.get(model, 1.0)
            for claim in result.claim_breakdown:
                claim_text = claim['claim']
                is_correct = claim['is_correct']
                
                if claim_text not in claim_votes:
                    claim_votes[claim_text] = {True: 0.0, False: 0.0}
                claim_votes[claim_text][is_correct] += weight
        
        # Determine final claim correctness based on weighted votes
        final_claims = []
        for claim_text, votes in claim_votes.items():
            is_correct = votes[True] > votes[False]
            final_claims.append({
                'claim': claim_text,
                'is_correct': is_correct,
                'confidence': max(votes[True], votes[False]) / (votes[True] + votes[False])
            })
        
        # Count total claims
        total_claims = len(final_claims)
        correct_claims = sum(1 for claim in final_claims if claim['is_correct'])
        incorrect_claims = total_claims - correct_claims
        
        # Combine detailed analysis with weights
        detailed_analysis = "Aggregated analysis (weighted by model confidence):\n\n"
        for model, result in zip(self.models[:len(results)], results):
            weight = self.model_weights.get(model, 1.0)
            detailed_analysis += f"Model {model} (weight: {weight:.2f}):\n{result.detailed_analysis}\n\n---\n\n"
        
        return PrecisionEvaluationResult(
            precision_score=avg_precision,
            total_claims=total_claims,
            correct_claims=correct_claims,
            incorrect_claims=incorrect_claims,
            detailed_analysis=detailed_analysis,
            claim_breakdown=final_claims
        )
    
    def _format_ground_truth(self, ground_truth: Dict[str, Any]) -> str:
        """Format ground truth information for LLM comparison."""
        formatted_info = []
        
        field_map = {
            "genre": "音乐风格",
            "language": "语言",
            "description": "歌曲描述",
            "theme": "歌曲主题",
            "music_style": "音乐风格细分",
            "vocal_characteristics": "嗓音特点",
            "style_or_atmosphere": "风格/氛围",
            "arrangement_or_details": "编曲/细节",
            "composition_or_structure": "作曲/结构",
            "vocal_tone_description": "嗓音描述",
            "emotional_expression": "情感表达",
            "vocal_technique_awareness": "演唱技巧",
            "singer_background_association": "歌手背景关联",
            "song_background_or_cultural_association": "歌曲背景/文化关联",
            "trend_or_subcultural_insight": "流行趋势/亚文化洞察"
        }

        for key in field_map:
            if key in ground_truth:
                formatted_info.append(f"{field_map[key]}:{ground_truth[key]}")

        # 处理未在映射表中的其他字段
        for key, value in ground_truth.items():
            if key not in field_map and key != "audio_path":
                formatted_info.append(f"{key}:{value}")

        return "\n".join(formatted_info) if formatted_info else "无详细信息"
    
    def _parse_precision_response(self, response: str) -> PrecisionEvaluationResult:
        """Parse the LLM precision evaluation response."""
        import re
        
        # Initialize default values
        total_claims = 0
        correct_claims = 0
        incorrect_claims = 0
        precision_score = 0.0
        detailed_analysis = response
        claim_breakdown = []
        
        # Extract statistics from response
        # Look for patterns like "总计事实声明：5条"
        total_match = re.search(r'总计事实声明[：:]\s*(\d+)', response)
        if total_match:
            total_claims = int(total_match.group(1))
        
        # Look for "正确事实：3条"
        correct_match = re.search(r'正确事实[：:]\s*(\d+)', response)
        if correct_match:
            correct_claims = int(correct_match.group(1))
        
        # Look for "错误事实：2条"
        incorrect_match = re.search(r'错误事实[：:]\s*(\d+)', response)
        if incorrect_match:
            incorrect_claims = int(incorrect_match.group(1))
        
        # Look for "准确率：60%"
        accuracy_match = re.search(r'准确率[：:]\s*(\d+\.?\d*)%', response)
        if accuracy_match:
            precision_score = float(accuracy_match.group(1)) / 100.0
        elif total_claims > 0:
            # Calculate from counts if percentage not found
            precision_score = correct_claims / total_claims
        
        # Extract individual claims analysis
        if "**事实核查分析**" in response:
            analysis_section = response.split("**事实核查分析**")[1]
            if "**准确性统计**" in analysis_section:
                analysis_section = analysis_section.split("**准确性统计**")[0]
            
            # Parse individual claims (simplified parsing)
            lines = analysis_section.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('**'):
                    # Try to identify if it's a claim evaluation
                    if '✓' in line or '✗' in line or '正确' in line or '错误' in line:
                        is_correct = '✓' in line or '正确' in line
                        claim_text = re.sub(r'[✓✗]', '', line).strip()
                        claim_breakdown.append({
                            'claim': claim_text,
                            'is_correct': is_correct
                        })
        
        return PrecisionEvaluationResult(
            precision_score=precision_score,
            total_claims=total_claims,
            correct_claims=correct_claims,
            incorrect_claims=incorrect_claims,
            detailed_analysis=detailed_analysis,
            claim_breakdown=claim_breakdown
        )
    
    def calculate_average_precision(self, results: List[PrecisionEvaluationResult]) -> float:
        """Calculate average precision score across multiple evaluations."""
        if not results:
            return 0.0
        
        valid_results = [r for r in results if r.total_claims > 0]
        if not valid_results:
            return 0.0
        
        return sum(r.precision_score for r in valid_results) / len(valid_results) 