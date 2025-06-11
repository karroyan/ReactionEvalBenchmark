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
    
    Uses LLM to compare model outputs with ground truth and evaluate
    the accuracy of factual claims made in the appraisal.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-v3"):
        """
        Initialize the precision evaluator.
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL  
            model: Model name to use
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.evaluation_prompt = self._get_precision_prompt()
    
    def _get_precision_prompt(self) -> str:
        """Get the precision evaluation prompt."""
        return """### **System Prompt**  
**角色**：你是一位专业的音乐事实核查员，负责验证音乐评价内容中的事实准确性。  
**任务**：对比音乐评价文本与歌曲真实信息，判断评价中提到的具体事实是否正确。重点关注准确性，而非完整性。

### **评估标准**  
- **只评估明确提及的事实信息**，不要求评价包含所有信息
- **重点关注准确性**：提到的信息是否与真实情况一致
- **包括但不限于**：歌手信息、发布时间、音乐风格、创作背景、歌曲主题等
- **忽略主观感受**：如"好听"、"感动"等个人观点不算事实错误

### **评分方式**
1. **识别所有事实性陈述**：从评价文本中提取具体的事实声明
2. **逐项核实**：对比每个事实与真实信息
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

    def call_api(self, user_message: str, system_prompt: str, 
                 temperature: float = 0.3) -> str:
        """
        Call the LLM API for precision evaluation.
        
        Args:
            user_message: User message content
            system_prompt: System prompt
            temperature: Sampling temperature (lower for more consistent factual evaluation)
            
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
                logprobs=False
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error calling API: {e}")
            return None
    
    def evaluate_precision(self, appraisal_text: str, 
                          ground_truth: Dict[str, Any]) -> PrecisionEvaluationResult:
        """
        Evaluate the factual precision of a music appraisal.
        
        Args:
            appraisal_text: The music appraisal text to evaluate
            ground_truth: Ground truth information about the song
            
        Returns:
            PrecisionEvaluationResult with detailed factual accuracy analysis
        """
        # Prepare ground truth information for comparison
        truth_info = self._format_ground_truth(ground_truth)
        
        # Prepare the user message
        user_message = f"""请核查以下音乐评价的事实准确性：

**评价文本**：
{appraisal_text}

**真实歌曲信息**：
{truth_info}

请按照评估标准进行事实核查分析。"""
        
        # Call the LLM
        response = self.call_api(user_message, self.evaluation_prompt)
        
        if response is None:
            # Return default result if API call failed
            return PrecisionEvaluationResult(
                precision_score=0.0,
                total_claims=0,
                correct_claims=0,
                incorrect_claims=0,
                detailed_analysis="API call failed",
                claim_breakdown=[]
            )
        
        # Parse the response
        try:
            return self._parse_precision_response(response)
        except Exception as e:
            print(f"Error parsing precision response: {e}")
            return PrecisionEvaluationResult(
                precision_score=0.0,
                total_claims=0,
                correct_claims=0,
                incorrect_claims=0,
                detailed_analysis=f"Parse error: {str(e)}. Response: {response}",
                claim_breakdown=[]
            )
    
    def _format_ground_truth(self, ground_truth: Dict[str, Any]) -> str:
        """Format ground truth information for LLM comparison."""
        formatted_info = []
        
        # Core song information
        if 'artist' in ground_truth:
            formatted_info.append(f"歌手：{ground_truth['artist']}")
        if 'title' in ground_truth:
            formatted_info.append(f"歌曲名：{ground_truth['title']}")
        if 'release_year' in ground_truth:
            formatted_info.append(f"发布年份：{ground_truth['release_year']}")
        if 'genre' in ground_truth:
            formatted_info.append(f"音乐风格：{ground_truth['genre']}")
        if 'album' in ground_truth:
            formatted_info.append(f"专辑：{ground_truth['album']}")
        
        # Song details
        if 'description' in ground_truth:
            formatted_info.append(f"歌曲描述：{ground_truth['description']}")
        if 'theme' in ground_truth:
            formatted_info.append(f"歌曲主题：{ground_truth['theme']}")
        if 'background' in ground_truth:
            formatted_info.append(f"创作背景：{ground_truth['background']}")
        
        # Additional metadata
        for key, value in ground_truth.items():
            if key not in ['artist', 'title', 'release_year', 'genre', 'album', 
                          'description', 'theme', 'background', 'audio_path']:
                formatted_info.append(f"{key}：{value}")
        
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