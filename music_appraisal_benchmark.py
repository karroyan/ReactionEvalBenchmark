#!/usr/bin/env python3
"""
Comprehensive Music Appraisal Benchmark System.

This module integrates multiple evaluation components:
1. Option-based question answering
2. LLM-based completeness scoring  
3. Precision evaluation against ground truth
4. Novelty/detail assessment

Provides a unified interface for evaluating music appraisal models.
"""

import json
import os
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path

from audio_qa import AudioQABenchmark, run_benchmark, EvaluationResult
from llm_evaluator import AudioLLMEvaluator, LLMEvaluationResult
from precision_evaluator import AudioPrecisionEvaluator, PrecisionEvaluationResult
from novelty_evaluator import AudioNoveltyEvaluator, NoveltyEvaluationResult


@dataclass
class ComprehensiveEvaluationResult:
    """Results from comprehensive music appraisal evaluation."""
    
    # Option-based QA results
    qa_accuracy: float
    qa_total_questions: int
    qa_correct_answers: int
    qa_detailed_results: List[Dict[str, Any]]
    
    # LLM completeness scoring results
    completeness_score: float
    completeness_max_score: float
    completeness_details: Dict[str, Any]
    completeness_assessment: str
    
    # Precision results
    precision_score: Optional[float] = None
    precision_details: Optional[Dict[str, Any]] = None
    
    # Novelty results ✨ NEW
    novelty_score: Optional[float] = None
    novelty_details: Optional[Dict[str, Any]] = None
    
    # Overall metrics
    overall_score: Optional[float] = None
    summary: Optional[str] = None


class MusicAppraisalBenchmark:
    """
    Comprehensive benchmark for evaluating music appraisal models.
    
    Integrates multiple evaluation components to provide a holistic
    assessment of model performance across different aspects.
    """
    
    def __init__(self, 
                 qa_data_path: str,
                 llm_api_key: str,
                 llm_base_url: str,
                 song_details_path: Optional[str] = None,
                 llm_model: str = "deepseek-v3"):
        """
        Initialize the comprehensive benchmark.
        
        Args:
            qa_data_path: Path to option-based QA data (JSONL)
            llm_api_key: API key for LLM evaluator
            llm_base_url: Base URL for LLM API
            song_details_path: Path to song details data (JSONL)
            llm_model: LLM model name to use
        """
        self.qa_data_path = qa_data_path
        self.song_details_path = song_details_path
        
        # Initialize QA benchmark
        self.qa_benchmark = AudioQABenchmark(qa_data_path)
        
        # Initialize LLM evaluator
        self.llm_evaluator = AudioLLMEvaluator(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model
        )
        
        # Initialize precision evaluator
        self.precision_evaluator = AudioPrecisionEvaluator(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model
        )
        
        # Initialize novelty evaluator
        self.novelty_evaluator = AudioNoveltyEvaluator(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model
        )
        
        # Load song details if provided
        self.song_details = self._load_song_details() if song_details_path else {}
    
    def _load_song_details(self) -> Dict[str, Dict]:
        """Load song details from JSONL file."""
        song_details = {}
        
        if not self.song_details_path or not os.path.exists(self.song_details_path):
            return song_details
        
        try:
            with open(self.song_details_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    audio_path = data.get('audio_path', '')
                    song_details[audio_path] = data
        except Exception as e:
            print(f"Warning: Failed to load song details: {e}")
        
        return song_details
    
    def evaluate_model(self,
                      qa_model_function: Callable[[str, str, List[str]], str],
                      appraisal_model_function: Callable[[str], str],
                      include_qa_confidence: bool = False,
                      qa_confidence_function: Optional[Callable] = None,
                      enable_precision_eval: bool = True,
                      enable_novelty_eval: bool = True) -> ComprehensiveEvaluationResult:
        """
        Evaluate a model comprehensively across all components.
        
        Args:
            qa_model_function: Function for answering option-based questions
            appraisal_model_function: Function for generating music appraisal text
            include_qa_confidence: Whether to include confidence scores for QA
            qa_confidence_function: Function for QA confidence scoring
            enable_precision_eval: Whether to run precision evaluation
            enable_novelty_eval: Whether to run novelty evaluation
            
        Returns:
            ComprehensiveEvaluationResult with all evaluation metrics
        """
        print("Starting comprehensive music appraisal evaluation...")
        
        # 1. Evaluate option-based QA
        print("\n1. Evaluating option-based question answering...")
        qa_result = self.qa_benchmark.evaluate_model(
            qa_model_function,
            include_confidence=include_qa_confidence,
            confidence_function=qa_confidence_function
        )
        
        # 2. Evaluate completeness using LLM
        print("\n2. Evaluating appraisal completeness...")
        completeness_results = []
        precision_results = []
        novelty_results = []
        
        for qa_item in self.qa_benchmark.qa_items:
            try:
                # Generate appraisal for each audio file
                appraisal_text = appraisal_model_function(qa_item.audio_path)
                
                # Get song details if available
                song_info = self.song_details.get(qa_item.audio_path)
                
                # Evaluate completeness
                completeness_result = self.llm_evaluator.evaluate_appraisal(
                    appraisal_text
                )
                
                completeness_results.append({
                    'audio_path': qa_item.audio_path,
                    'appraisal_text': appraisal_text,
                    'completeness_result': completeness_result,
                    'song_info': song_info
                })
                
                # 3. Evaluate precision if enabled and ground truth available
                if enable_precision_eval and song_info:
                    print(f"   Evaluating precision for {os.path.basename(qa_item.audio_path)}...")
                    precision_result = self.precision_evaluator.evaluate_precision(
                        appraisal_text, song_info
                    )
                    
                    precision_results.append({
                        'audio_path': qa_item.audio_path,
                        'precision_result': precision_result,
                        'song_info': song_info
                    })
                
                # 4. Evaluate novelty if enabled and ground truth available
                if enable_novelty_eval and song_info:
                    print(f"   Evaluating novelty for {os.path.basename(qa_item.audio_path)}...")
                    novelty_result = self.novelty_evaluator.evaluate_novelty(
                        appraisal_text, song_info
                    )
                    
                    novelty_results.append({
                        'audio_path': qa_item.audio_path,
                        'novelty_result': novelty_result,
                        'song_info': song_info
                    })
                
            except Exception as e:
                print(f"Error evaluating {qa_item.audio_path}: {e}")
                continue
        
        # Calculate average completeness scores
        if completeness_results:
            avg_completeness_score = sum(r['completeness_result'].total_score 
                                       for r in completeness_results) / len(completeness_results)
            avg_completeness_max = sum(r['completeness_result'].max_score 
                                     for r in completeness_results) / len(completeness_results)
            
            # Aggregate detailed scores
            aggregated_details = self._aggregate_completeness_details(completeness_results)
        else:
            avg_completeness_score = 0.0
            avg_completeness_max = 16.0
            aggregated_details = {}
        
        # Calculate precision scores
        if precision_results:
            avg_precision_score = sum(r['precision_result'].precision_score 
                                    for r in precision_results) / len(precision_results)
            precision_details = self._aggregate_precision_details(precision_results)
            print(f"\n3. Precision evaluation completed: {avg_precision_score:.2%}")
        else:
            avg_precision_score = None
            precision_details = None
            if enable_precision_eval:
                print("\n3. Precision evaluation skipped (no ground truth data)")
        
        # Calculate novelty scores
        if novelty_results:
            avg_novelty_score = sum(r['novelty_result'].novelty_score 
                                  for r in novelty_results) / len(novelty_results)
            novelty_details = self._aggregate_novelty_details(novelty_results)
            print(f"\n4. Novelty evaluation completed: {avg_novelty_score:.2%}")
        else:
            avg_novelty_score = None
            novelty_details = None
            if enable_novelty_eval:
                print("\n4. Novelty evaluation skipped (no ground truth data)")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            qa_result.accuracy,
            avg_completeness_score / avg_completeness_max if avg_completeness_max > 0 else 0,
            avg_precision_score,
            avg_novelty_score
        )
        
        # Generate summary
        summary = self._generate_summary(qa_result, avg_completeness_score, avg_precision_score, avg_novelty_score, overall_score)
        
        return ComprehensiveEvaluationResult(
            qa_accuracy=qa_result.accuracy,
            qa_total_questions=qa_result.total_questions,
            qa_correct_answers=qa_result.correct_answers,
            qa_detailed_results=qa_result.detailed_results,
            completeness_score=avg_completeness_score,
            completeness_max_score=avg_completeness_max,
            completeness_details=aggregated_details,
            completeness_assessment=completeness_results[0]['completeness_result'].overall_assessment if completeness_results else "",
            precision_score=avg_precision_score,
            precision_details=precision_details,
            novelty_score=avg_novelty_score,
            novelty_details=novelty_details,
            overall_score=overall_score,
            summary=summary
        )
    
    def _aggregate_completeness_details(self, completeness_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate detailed completeness scores across all evaluations."""
        if not completeness_results:
            return {}
        
        # Since detailed scores are no longer available, return basic aggregation
        total_scores = [r['completeness_result'].total_score for r in completeness_results]
        assessments = [r['completeness_result'].overall_assessment for r in completeness_results]
        
        return {
            'average_total_score': sum(total_scores) / len(total_scores) if total_scores else 0,
            'individual_scores': total_scores,
            'assessments': assessments
        }
    
    def _aggregate_precision_details(self, precision_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate detailed precision scores across all evaluations."""
        if not precision_results:
            return {}
        
        total_claims = sum(r['precision_result'].total_claims for r in precision_results)
        total_correct = sum(r['precision_result'].correct_claims for r in precision_results)
        total_incorrect = sum(r['precision_result'].incorrect_claims for r in precision_results)
        
        individual_scores = [r['precision_result'].precision_score for r in precision_results]
        
        # Collect all claim breakdowns
        all_claims = []
        for result in precision_results:
            for claim in result['precision_result'].claim_breakdown:
                all_claims.append({
                    'audio_path': result['audio_path'],
                    'claim': claim['claim'],
                    'is_correct': claim['is_correct']
                })
        
        return {
            'total_claims': total_claims,
            'total_correct_claims': total_correct,
            'total_incorrect_claims': total_incorrect,
            'overall_precision': total_correct / total_claims if total_claims > 0 else 0,
            'individual_scores': individual_scores,
            'average_score': sum(individual_scores) / len(individual_scores) if individual_scores else 0,
            'claim_breakdown': all_claims,
            'files_evaluated': len(precision_results)
        }
    
    def _aggregate_novelty_details(self, novelty_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate detailed novelty scores across all evaluations."""
        if not novelty_results:
            return {}
        
        # Calculate averages
        avg_music_relevance = sum(r['novelty_result'].music_relevance_score for r in novelty_results) / len(novelty_results)
        avg_depth_score = sum(r['novelty_result'].depth_score for r in novelty_results) / len(novelty_results)
        avg_personal_insight = sum(r['novelty_result'].personal_insight_score for r in novelty_results) / len(novelty_results)
        
        # Aggregate content categories
        all_categories = {
            'personal_reactions': [],
            'technical_analysis': [],
            'creative_interpretations': [],
            'cultural_context': [],
            'comparative_analysis': []
        }
        
        # Count insight types
        insight_counts = {
            'has_personal_reactions': 0,
            'has_technical_analysis': 0,
            'has_creative_interpretations': 0,
            'has_cultural_context': 0,
            'has_comparative_analysis': 0
        }
        
        total_novel_statements = sum(r['novelty_result'].total_novel_statements for r in novelty_results)
        total_music_related = sum(r['novelty_result'].music_related_novel_statements for r in novelty_results)
        
        for result in novelty_results:
            novelty_result = result['novelty_result']
            
            # Aggregate categories
            for category, items in novelty_result.novel_content_categories.items():
                if category in all_categories:
                    all_categories[category].extend([
                        {'audio_path': result['audio_path'], 'content': item} for item in items
                    ])
            
            # Count insight types
            if novelty_result.has_personal_reactions:
                insight_counts['has_personal_reactions'] += 1
            if novelty_result.has_technical_analysis:
                insight_counts['has_technical_analysis'] += 1
            if novelty_result.has_creative_interpretations:
                insight_counts['has_creative_interpretations'] += 1
            if novelty_result.has_cultural_context:
                insight_counts['has_cultural_context'] += 1
            if novelty_result.has_comparative_analysis:
                insight_counts['has_comparative_analysis'] += 1
        
        return {
            'avg_music_relevance_score': avg_music_relevance,
            'avg_depth_score': avg_depth_score,
            'avg_personal_insight_score': avg_personal_insight,
            'total_novel_statements': total_novel_statements,
            'total_music_related_statements': total_music_related,
            'music_relevance_ratio': total_music_related / total_novel_statements if total_novel_statements > 0 else 0,
            'novel_content_categories': all_categories,
            'insight_type_counts': insight_counts,
            'files_evaluated': len(novelty_results),
            'individual_scores': [r['novelty_result'].novelty_score for r in novelty_results]
        }
    
    def _calculate_overall_score(self, qa_accuracy: float, completeness_ratio: float,
                               precision_score: Optional[float], novelty_score: Optional[float]) -> float:
        """Calculate overall score combining all evaluation components."""
        scores = [qa_accuracy, completeness_ratio]
        weights = [0.3, 0.4]  # QA: 30%, Completeness: 40%
        
        if precision_score is not None:
            scores.append(precision_score)
            weights.append(0.2)  # Precision: 20%
        
        if novelty_score is not None:
            scores.append(novelty_score)
            weights.append(0.1)  # Novelty: 10%
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return sum(score * weight for score, weight in zip(scores, weights))
    
    def _generate_summary(self, qa_result: EvaluationResult, completeness_score: float,
                         precision_score: Optional[float], novelty_score: Optional[float], overall_score: float) -> str:
        """Generate a summary of the evaluation results."""
        precision_text = f"\n- 事实准确率: {precision_score:.2%}" if precision_score is not None else ""
        novelty_text = f"\n- 新颖性得分: {novelty_score:.2%}" if novelty_score is not None else ""
        
        return f"""模型综合评估结果：
- 选择题准确率: {qa_result.accuracy:.2%} ({qa_result.correct_answers}/{qa_result.total_questions})
- 完整性评分: {completeness_score:.1f}/16.0 ({completeness_score/16*100:.1f}%){precision_text}{novelty_text}
- 综合得分: {overall_score:.2%}
- 平均响应时间: {qa_result.average_response_time:.2f}秒"""
    
    def save_results(self, result: ComprehensiveEvaluationResult, output_path: str):
        """Save comprehensive evaluation results to JSON file."""
        output_data = {
            'summary': {
                'qa_accuracy': result.qa_accuracy,
                'qa_total_questions': result.qa_total_questions,
                'qa_correct_answers': result.qa_correct_answers,
                'completeness_score': result.completeness_score,
                'completeness_max_score': result.completeness_max_score,
                'precision_score': result.precision_score,
                'novelty_score': result.novelty_score,
                'overall_score': result.overall_score,
                'summary': result.summary
            },
            'detailed_results': {
                'qa_results': result.qa_detailed_results,
                'completeness_details': result.completeness_details,
                'precision_details': result.precision_details,
                'novelty_details': result.novelty_details
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive results saved to: {output_path}")
    
    def print_detailed_report(self, result: ComprehensiveEvaluationResult):
        """Print a detailed evaluation report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MUSIC APPRAISAL EVALUATION REPORT")
        print("="*80)
        
        print(f"\n1. OPTION-BASED QUESTION ANSWERING")
        print(f"   Accuracy: {result.qa_accuracy:.4f} ({result.qa_accuracy*100:.2f}%)")
        print(f"   Correct: {result.qa_correct_answers}/{result.qa_total_questions}")
        
        print(f"\n2. COMPLETENESS EVALUATION")
        print(f"   Score: {result.completeness_score:.2f}/{result.completeness_max_score:.2f}")
        print(f"   Percentage: {result.completeness_score/result.completeness_max_score*100:.1f}%")
        print(f"   Assessment: {result.completeness_assessment}")
        
        if result.completeness_details:
            print(f"\n   Additional Details:")
            avg_score = result.completeness_details.get('average_total_score', 0)
            print(f"     Average Score: {avg_score:.2f}")
            individual_scores = result.completeness_details.get('individual_scores', [])
            if individual_scores:
                print(f"     Score Range: {min(individual_scores):.1f} - {max(individual_scores):.1f}")
                print(f"     Number of Evaluations: {len(individual_scores)}")
        
        if result.precision_score is not None:
            print(f"\n3. PRECISION EVALUATION")
            print(f"   Overall Accuracy: {result.precision_score:.4f} ({result.precision_score*100:.2f}%)")
            
            if result.precision_details:
                details = result.precision_details
                print(f"   Total Claims: {details.get('total_claims', 0)}")
                print(f"   Correct Claims: {details.get('total_correct_claims', 0)}")
                print(f"   Incorrect Claims: {details.get('total_incorrect_claims', 0)}")
                print(f"   Files Evaluated: {details.get('files_evaluated', 0)}")
                
                individual_scores = details.get('individual_scores', [])
                if individual_scores:
                    print(f"   Score Range: {min(individual_scores):.1%} - {max(individual_scores):.1%}")
        
        if result.novelty_score is not None:
            print(f"\n4. NOVELTY EVALUATION")
            print(f"   Overall Novelty: {result.novelty_score:.4f} ({result.novelty_score*100:.2f}%)")
            
            if result.novelty_details:
                details = result.novelty_details
                print(f"   Music Relevance: {details.get('avg_music_relevance_score', 0):.2f}")
                print(f"   Depth Score: {details.get('avg_depth_score', 0):.2f}")
                print(f"   Personal Insight: {details.get('avg_personal_insight_score', 0):.2f}")
                print(f"   Novel Statements: {details.get('total_novel_statements', 0)}")
                print(f"   Music-Related Ratio: {details.get('music_relevance_ratio', 0):.1%}")
                
                # Show insight type breakdown
                insight_counts = details.get('insight_type_counts', {})
                files_count = details.get('files_evaluated', 1)
                print(f"\n   Insight Types Found:")
                print(f"     Personal Reactions: {insight_counts.get('has_personal_reactions', 0)}/{files_count}")
                print(f"     Technical Analysis: {insight_counts.get('has_technical_analysis', 0)}/{files_count}")
                print(f"     Creative Interpretations: {insight_counts.get('has_creative_interpretations', 0)}/{files_count}")
                print(f"     Cultural Context: {insight_counts.get('has_cultural_context', 0)}/{files_count}")
                print(f"     Comparative Analysis: {insight_counts.get('has_comparative_analysis', 0)}/{files_count}")
        
        print(f"\n5. OVERALL ASSESSMENT")
        print(f"   Overall Score: {result.overall_score:.4f} ({result.overall_score*100:.2f}%)")
        print(f"\n{result.summary}")
        
        print("="*80)


# Convenience function for easy usage
def run_comprehensive_benchmark(
    qa_model_function: Callable[[str, str, List[str]], str],
    appraisal_model_function: Callable[[str], str],
    qa_data_path: str = "eval_benchmark/option_qa.jsonl",
    song_details_path: str = "eval_benchmark/data/song_details.jsonl",
    llm_api_key: str = "sk-dar2q3sf4dqtzikq",
    llm_base_url: str = "https://cloud.infini-ai.com/maas/v1",
    output_path: Optional[str] = None,
    include_qa_confidence: bool = False,
    qa_confidence_function: Optional[Callable] = None,
    enable_precision_eval: bool = True,
    enable_novelty_eval: bool = True
) -> ComprehensiveEvaluationResult:
    """
    Run comprehensive music appraisal benchmark evaluation.
    
    Args:
        qa_model_function: Function for option-based QA
        appraisal_model_function: Function for generating appraisals
        qa_data_path: Path to QA data
        song_details_path: Path to song details
        llm_api_key: LLM API key
        llm_base_url: LLM API base URL
        output_path: Optional output file path
        include_qa_confidence: Whether to include QA confidence
        qa_confidence_function: QA confidence function
        enable_precision_eval: Whether to run precision evaluation
        enable_novelty_eval: Whether to run novelty evaluation
        
    Returns:
        ComprehensiveEvaluationResult
    """
    benchmark = MusicAppraisalBenchmark(
        qa_data_path=qa_data_path,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        song_details_path=song_details_path
    )
    
    result = benchmark.evaluate_model(
        qa_model_function=qa_model_function,
        appraisal_model_function=appraisal_model_function,
        include_qa_confidence=include_qa_confidence,
        qa_confidence_function=qa_confidence_function,
        enable_precision_eval=enable_precision_eval,
        enable_novelty_eval=enable_novelty_eval
    )
    
    benchmark.print_detailed_report(result)
    
    if output_path:
        benchmark.save_results(result, output_path)
    
    return result 