#!/usr/bin/env python3
"""
Novelty evaluator for music appraisal depth and personal insights.

This module evaluates whether a music appraisal provides new, detailed, and 
personal information beyond the basic ground truth facts, focusing on 
music-related novelty and depth of insight.
"""

import json
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class NoveltyEvaluationResult:
    """Results from novelty evaluation of music appraisal."""
    novelty_score: float  # 0-1 overall novelty score
    music_relevance_score: float  # 0-1 how music-related the novel content is
    depth_score: float  # 0-1 depth and insight of novel content
    personal_insight_score: float  # 0-1 personal/subjective insights
    
    # Detailed breakdown
    novel_content_categories: Dict[str, List[str]]  # Categorized novel content
    total_novel_statements: int
    music_related_novel_statements: int
    detailed_analysis: str
    
    # Specific insight types found
    has_personal_reactions: bool
    has_technical_analysis: bool
    has_creative_interpretations: bool
    has_cultural_context: bool
    has_comparative_analysis: bool


class AudioNoveltyEvaluator:
    """
    Novelty evaluator for music appraisal depth and personal insights.
    
    Evaluates whether appraisals provide meaningful new information beyond
    ground truth, focusing on music-related novelty and personal insights.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-v3"):
        """
        Initialize the novelty evaluator.
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL  
            model: Model name to use
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.evaluation_prompt = self._get_novelty_prompt()
    
    def _get_novelty_prompt(self) -> str:
        """Get the novelty evaluation prompt."""
        return """### **System Prompt**  
**角色**：你是一位专业的音乐评论分析师，专门评估音乐评价的深度、新颖性和个人洞察力。  
**任务**：识别评价文本中超越基础事实的新颖内容，评估其音乐相关性和洞察深度。

### **评估维度**  

#### 1. **新颖性识别** (重点关注超越基础信息的内容)
- **个人情感反应**: "让我想起...", "给我的感觉是...", "听到这首歌时我..."
- **深度技术分析**: 超越基本风格的具体音乐制作细节、乐器使用、编曲技巧
- **创意解读**: 隐喻、比喻、艺术性描述 ("声音如丝绸般", "鼓点像心跳")
- **文化背景**: 时代背景、社会影响、文化意义分析
- **对比分析**: 与其他歌曲、艺术家的比较和关联

#### 2. **音乐相关性** (确保新颖内容与音乐相关)
- 必须与音乐本身、表演、制作、或音乐体验相关
- 排除完全无关的个人生活分享或离题内容

#### 3. **洞察深度** (评估分析的深入程度)
- **表面级**: 简单的好听/不好听
- **分析级**: 具体的音乐元素分析
- **洞察级**: 深入的音乐理解和独特视角

### **评分标准** (10分制)
- **新颖性得分** (0-4分): 超越基础事实的新信息量
- **音乐相关性** (0-3分): 新颖内容的音乐相关程度  
- **洞察深度** (0-3分): 分析的深入程度和独特性

### **输出格式要求**  
请按以下格式输出评估结果：

**新颖内容识别**：
[按类别列出发现的新颖内容]
- 个人情感反应: [内容列表]
- 深度技术分析: [内容列表]  
- 创意解读: [内容列表]
- 文化背景: [内容列表]
- 对比分析: [内容列表]

**音乐相关性评估**：
- 音乐相关的新颖内容: X条
- 音乐无关或离题内容: X条
- 音乐相关性得分: X/3分

**洞察深度评估**：
- 表面级评价: X条
- 分析级评价: X条  
- 洞察级评价: X条
- 洞察深度得分: X/3分

**新颖性统计**：
- 总计新颖内容: X条
- 新颖性得分: X/4分
- 综合得分: X/10分

**总体评价**：
[一句话总结新颖性和洞察力表现]"""

    def call_api(self, user_message: str, system_prompt: str, 
                 temperature: float = 0.4) -> str:
        """
        Call the LLM API for novelty evaluation.
        
        Args:
            user_message: User message content
            system_prompt: System prompt
            temperature: Sampling temperature (moderate for creativity assessment)
            
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
    
    def evaluate_novelty(self, appraisal_text: str, 
                        ground_truth: Dict[str, Any]) -> NoveltyEvaluationResult:
        """
        Evaluate the novelty and personal insights in a music appraisal.
        
        Args:
            appraisal_text: The music appraisal text to evaluate
            ground_truth: Ground truth information about the song
            
        Returns:
            NoveltyEvaluationResult with detailed novelty analysis
        """
        # Format ground truth for comparison
        truth_info = self._format_ground_truth(ground_truth)
        
        # Prepare the user message
        user_message = f"""请评估以下音乐评价的新颖性和洞察力：

**评价文本**：
{appraisal_text}

**已知基础信息**（评估时排除这些基础事实）：
{truth_info}

请识别超越这些基础信息的新颖内容，并按评估标准进行分析。"""
        
        # Call the LLM
        response = self.call_api(user_message, self.evaluation_prompt)
        
        if response is None:
            # Return default result if API call failed
            return NoveltyEvaluationResult(
                novelty_score=0.0,
                music_relevance_score=0.0,
                depth_score=0.0,
                personal_insight_score=0.0,
                novel_content_categories={},
                total_novel_statements=0,
                music_related_novel_statements=0,
                detailed_analysis="API call failed",
                has_personal_reactions=False,
                has_technical_analysis=False,
                has_creative_interpretations=False,
                has_cultural_context=False,
                has_comparative_analysis=False
            )
        
        # Parse the response
        try:
            return self._parse_novelty_response(response)
        except Exception as e:
            print(f"Error parsing novelty response: {e}")
            return NoveltyEvaluationResult(
                novelty_score=0.0,
                music_relevance_score=0.0,
                depth_score=0.0,
                personal_insight_score=0.0,
                novel_content_categories={},
                total_novel_statements=0,
                music_related_novel_statements=0,
                detailed_analysis=f"Parse error: {str(e)}. Response: {response}",
                has_personal_reactions=False,
                has_technical_analysis=False,
                has_creative_interpretations=False,
                has_cultural_context=False,
                has_comparative_analysis=False
            )
    
    def _format_ground_truth(self, ground_truth: Dict[str, Any]) -> str:
        """Format ground truth information for comparison."""
        formatted_info = []
        
        # List all known basic facts
        fact_fields = ['artist', 'title', 'release_year', 'album', 'genre', 
                      'theme', 'background', 'music_style', 'vocal_characteristics',
                      'description', 'chart_performance', 'awards']
        
        for field in fact_fields:
            if field in ground_truth:
                formatted_info.append(f"{field}: {ground_truth[field]}")
        
        return "\n".join(formatted_info) if formatted_info else "无基础信息"
    
    def _parse_novelty_response(self, response: str) -> NoveltyEvaluationResult:
        """Parse the LLM novelty evaluation response."""
        # Initialize default values
        novelty_score = 0.0
        music_relevance_score = 0.0
        depth_score = 0.0
        personal_insight_score = 0.0
        
        novel_content_categories = {
            'personal_reactions': [],
            'technical_analysis': [],
            'creative_interpretations': [],
            'cultural_context': [],
            'comparative_analysis': []
        }
        
        total_novel_statements = 0
        music_related_novel_statements = 0
        detailed_analysis = response
        
        # Extract category content
        categories_map = {
            '个人情感反应': 'personal_reactions',
            '深度技术分析': 'technical_analysis',
            '创意解读': 'creative_interpretations',
            '文化背景': 'cultural_context',
            '对比分析': 'comparative_analysis'
        }
        
        # Parse novel content categories
        if "**新颖内容识别**" in response:
            content_section = response.split("**新颖内容识别**")[1]
            if "**音乐相关性评估**" in content_section:
                content_section = content_section.split("**音乐相关性评估**")[0]
            
            for chinese_name, english_key in categories_map.items():
                if chinese_name in content_section:
                    # Extract content for this category
                    pattern = rf'{chinese_name}:\s*\[(.*?)\]'
                    match = re.search(pattern, content_section, re.DOTALL)
                    if match:
                        content_text = match.group(1).strip()
                        if content_text and content_text != '无' and content_text != 'None':
                            # Split by commas or line breaks
                            items = [item.strip() for item in re.split(r'[,，\n]', content_text) 
                                   if item.strip() and item.strip() not in ['无', 'None', '暂无']]
                            novel_content_categories[english_key] = items
        
        # Extract scores
        # Look for "新颖性得分: X/4分"
        novelty_match = re.search(r'新颖性得分[：:]\s*(\d+\.?\d*)/4', response)
        if novelty_match:
            novelty_score = float(novelty_match.group(1)) / 4.0
        
        # Look for "音乐相关性得分: X/3分"
        relevance_match = re.search(r'音乐相关性得分[：:]\s*(\d+\.?\d*)/3', response)
        if relevance_match:
            music_relevance_score = float(relevance_match.group(1)) / 3.0
        
        # Look for "洞察深度得分: X/3分"
        depth_match = re.search(r'洞察深度得分[：:]\s*(\d+\.?\d*)/3', response)
        if depth_match:
            depth_score = float(depth_match.group(1)) / 3.0
        
        # Extract statement counts
        total_match = re.search(r'总计新颖内容[：:]\s*(\d+)', response)
        if total_match:
            total_novel_statements = int(total_match.group(1))
        
        related_match = re.search(r'音乐相关的新颖内容[：:]\s*(\d+)', response)
        if related_match:
            music_related_novel_statements = int(related_match.group(1))
        
        # Calculate personal insight score (combination of personal reactions and depth)
        personal_reactions_count = len(novel_content_categories['personal_reactions'])
        creative_interpretations_count = len(novel_content_categories['creative_interpretations'])
        personal_insight_score = min(1.0, (personal_reactions_count + creative_interpretations_count) / 5.0)
        
        # Determine what types of insights are present
        has_personal_reactions = len(novel_content_categories['personal_reactions']) > 0
        has_technical_analysis = len(novel_content_categories['technical_analysis']) > 0
        has_creative_interpretations = len(novel_content_categories['creative_interpretations']) > 0
        has_cultural_context = len(novel_content_categories['cultural_context']) > 0
        has_comparative_analysis = len(novel_content_categories['comparative_analysis']) > 0
        
        return NoveltyEvaluationResult(
            novelty_score=novelty_score,
            music_relevance_score=music_relevance_score,
            depth_score=depth_score,
            personal_insight_score=personal_insight_score,
            novel_content_categories=novel_content_categories,
            total_novel_statements=total_novel_statements,
            music_related_novel_statements=music_related_novel_statements,
            detailed_analysis=detailed_analysis,
            has_personal_reactions=has_personal_reactions,
            has_technical_analysis=has_technical_analysis,
            has_creative_interpretations=has_creative_interpretations,
            has_cultural_context=has_cultural_context,
            has_comparative_analysis=has_comparative_analysis
        )
    
    def calculate_average_novelty(self, results: List[NoveltyEvaluationResult]) -> Dict[str, float]:
        """Calculate average novelty scores across multiple evaluations."""
        if not results:
            return {
                'avg_novelty_score': 0.0,
                'avg_music_relevance': 0.0,
                'avg_depth_score': 0.0,
                'avg_personal_insight': 0.0
            }
        
        return {
            'avg_novelty_score': sum(r.novelty_score for r in results) / len(results),
            'avg_music_relevance': sum(r.music_relevance_score for r in results) / len(results),
            'avg_depth_score': sum(r.depth_score for r in results) / len(results),
            'avg_personal_insight': sum(r.personal_insight_score for r in results) / len(results)
        } 