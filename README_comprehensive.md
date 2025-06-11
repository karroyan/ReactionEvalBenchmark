# Comprehensive Music Appraisal Benchmark

A comprehensive evaluation framework for audio models that generate music appraisals and answer music-related questions. This benchmark evaluates models across multiple dimensions to provide a holistic assessment of their music understanding and appraisal capabilities.

## 🎯 Evaluation Components

### 1. Option-based Question Answering (30% weight)
- **Purpose**: Tests specific music knowledge and comprehension
- **Method**: Multiple-choice questions about song structure, style, and content
- **Scoring**: Accuracy percentage based on correct answers
- **Features**: Flexible answer extraction from verbose model responses

### 2. LLM-based Completeness Scoring (40% weight) 
- **Purpose**: Evaluates whether appraisals contain all required elements
- **Method**: Uses LLM to score content across 4 dimensions (16-point scale):
  - 音乐理解 (Music Understanding): 7 points
  - 背景及情景理解 (Background Understanding): 4 points  
  - 语言表达 (Language Expression): 2 points
  - 人设类型 (Persona Consistency): 3 points
- **Scoring**: Total score out of 16 points with detailed breakdown

### 3. Precision Evaluation (20% weight)
- **Purpose**: Checks factual accuracy of claims made in appraisals
- **Method**: Uses LLM to compare model outputs with ground truth song details
- **Focus**: Only evaluates accuracy of mentioned facts (not completeness)
- **Features**: 
  - Identifies all factual claims in appraisal text
  - Compares each claim against verified ground truth
  - Calculates precision score: correct claims / total claims
  - Provides detailed breakdown of correct vs incorrect facts

### 4. Novelty Assessment (10% weight) ✨ **NEW**
- **Purpose**: Rewards detailed, personal, and creative insights beyond basic facts
- **Method**: Uses LLM to identify and evaluate novel content in 5 categories
- **Focus**: Music-related novelty, depth of insight, and personal perspectives
- **Categories**:
  - **Personal Reactions**: Emotional responses, memories, personal connections
  - **Technical Analysis**: Detailed production insights, instrumentation, arrangement
  - **Creative Interpretations**: Metaphors, artistic descriptions, sensory language
  - **Cultural Context**: Historical significance, social impact, era analysis
  - **Comparative Analysis**: Connections to other artists, songs, or genres

## 🚀 Quick Start

### Basic Usage

```python
from music_appraisal_benchmark import run_comprehensive_benchmark

def my_qa_model(audio_path: str, question: str, options: List[str]) -> str:
    # Your QA model implementation
    return "I think the answer is A"

def my_appraisal_model(audio_path: str) -> str:
    # Your appraisal generation model
    return "This song by 蔡依林 is a beautiful ballad..."

# Run comprehensive evaluation
result = run_comprehensive_benchmark(
    qa_model_function=my_qa_model,
    appraisal_model_function=my_appraisal_model,
    qa_data_path="data/option_qa.jsonl",
    song_details_path="data/song_details.jsonl",  # Required for precision & novelty eval
    enable_precision_eval=True,
    enable_novelty_eval=True
)

print(f"Overall Score: {result.overall_score:.2%}")
print(f"QA Accuracy: {result.qa_accuracy:.2%}")
print(f"Completeness: {result.completeness_score:.1f}/16")
print(f"Precision: {result.precision_score:.2%}")
print(f"Novelty: {result.novelty_score:.2%}")
```

### Advanced Usage

```python
from music_appraisal_benchmark import MusicAppraisalBenchmark

# Initialize benchmark with custom settings
benchmark = MusicAppraisalBenchmark(
    qa_data_path="data/option_qa.jsonl",
    song_details_path="data/song_details.jsonl",
    llm_api_key="your-api-key",
    llm_base_url="your-llm-endpoint"
)

# Run evaluation with custom options
result = benchmark.evaluate_model(
    qa_model_function=my_qa_model,
    appraisal_model_function=my_appraisal_model,
    enable_precision_eval=True,
    enable_novelty_eval=True,
    include_qa_confidence=True,
    qa_confidence_function=my_confidence_model
)

# Get detailed report
benchmark.print_detailed_report(result)
benchmark.save_results(result, "results.json")
```

## 📊 Data Formats

### QA Data Format (`option_qa.jsonl`)
```json
{
  "audio_path": "/path/to/audio.mp3",
  "question": "这首歌的段落类型是什么？",
  "options": ["verse", "chorus", "bridge", "outro"],
  "correct_answer": "A"
}
```

### Song Details Format (`song_details.jsonl`) - For Precision & Novelty Evaluation
```json
{
  "audio_path": "/path/to/audio.mp3",
  "artist": "蔡依林",
  "title": "倒带", 
  "release_year": 2004,
  "album": "城堡",
  "genre": "流行",
  "theme": "回忆与时光倒流",
  "background": "表达对过往恋情的怀念",
  "music_style": "融合流行和电子元素",
  "vocal_characteristics": "清甜中带有成熟韵味"
}
```

## 🎯 Precision Evaluation Details

The precision evaluation component focuses on **factual accuracy** rather than completeness. It:

1. **Extracts Factual Claims**: Identifies specific factual statements in the appraisal
2. **Compares with Ground Truth**: Checks each claim against verified song information  
3. **Ignores Subjective Content**: Skips personal opinions like "sounds good" or "very moving"
4. **Calculates Precision**: Reports the percentage of factual claims that are correct

### Example Precision Analysis

**Model Output:**
> "这首《倒带》是蔡依林在2004年发布的经典作品，收录在专辑《城堡》中。歌曲采用了R&B风格，表达了对回忆的怀念。"

**Ground Truth:**
- Release year: 2004 ✓
- Album: 城堡 ✓  
- Genre: 流行 (not R&B) ✗
- Theme: 回忆 ✓

**Precision Score:** 3/4 = 75%

## ✨ Novelty Evaluation Details

The novelty evaluation component focuses on **creative depth and personal insights** beyond basic facts. It:

1. **Identifies Novel Content**: Finds content that goes beyond ground truth information
2. **Categorizes by Type**: Classifies novel content into 5 insight categories
3. **Evaluates Music Relevance**: Ensures novel content is actually about music
4. **Scores Depth**: Assesses the insight level from surface to deep analysis

### Novel Content Categories

#### 1. Personal Reactions 🎭
- Emotional responses: "这首歌让我想起..."
- Memory connections: "听到时我仿佛回到..."
- Personal interpretations: "对我来说这首歌代表..."

#### 2. Technical Analysis 🔧
- Production details: "鼓点的处理特别巧妙"
- Instrumentation insights: "弦乐的运用画龙点睛"
- Arrangement specifics: "副歌部分的和声层次"

#### 3. Creative Interpretations 🎨
- Metaphors: "声音如丝绸般顺滑"
- Artistic descriptions: "鼓点像心跳一样规律"
- Sensory language: "音色仿佛在耳边轻抚"

#### 4. Cultural Context 🌍
- Historical significance: "华语流行音乐黄金时代的尾声"
- Social impact: "代表了那个时代的情感表达"
- Era analysis: "承上启下的经典之作"

#### 5. Comparative Analysis 📊
- Artist comparisons: "和王菲的《红豆》有异曲同工之妙"
- Genre connections: "让我想起了日本的City Pop"
- Style evolution: "比她之前的作品更有深度"

### Example Novelty Analysis

**Model Output:**
> "听到这首歌的第一秒，我就仿佛被带回到了那个青涩的年代。蔡依林的声音就像丝绸划过水面一样顺滑，鼓点的处理特别巧妙，每一下击打都像心跳一样规律。我觉得这首歌和王菲的《红豆》有异曲同工之妙。"

**Novelty Analysis:**
- Personal Reactions: "仿佛被带回到那个青涩的年代" ✓
- Creative Interpretations: "像丝绸划过水面", "像心跳一样规律" ✓✓  
- Technical Analysis: "鼓点的处理特别巧妙" ✓
- Comparative Analysis: "和王菲的《红豆》有异曲同工之妙" ✓

**Novelty Score:** High (5 novel statements, all music-related)

### Scoring Criteria (10-point scale)
- **Novelty Score** (0-4 points): Amount of new information beyond basic facts
- **Music Relevance** (0-3 points): How music-related the novel content is
- **Insight Depth** (0-3 points): Depth and uniqueness of analysis

## 📈 Evaluation Results

### Result Structure
```python
@dataclass
class ComprehensiveEvaluationResult:
    # QA metrics
    qa_accuracy: float
    qa_total_questions: int
    qa_correct_answers: int
    
    # Completeness metrics  
    completeness_score: float
    completeness_max_score: float
    completeness_assessment: str
    
    # Precision metrics
    precision_score: float
    precision_details: Dict[str, Any]
    
    # Novelty metrics ✨ NEW
    novelty_score: float
    novelty_details: Dict[str, Any]
    
    # Overall assessment
    overall_score: float
    summary: str
```

### Novelty Details Structure
```python
{
    "avg_music_relevance_score": 0.85,
    "avg_depth_score": 0.72,
    "avg_personal_insight_score": 0.68,
    "total_novel_statements": 25,
    "total_music_related_statements": 23,
    "music_relevance_ratio": 0.92,
    "novel_content_categories": {
        "personal_reactions": [...],
        "technical_analysis": [...],
        "creative_interpretations": [...],
        "cultural_context": [...],
        "comparative_analysis": [...]
    },
    "insight_type_counts": {
        "has_personal_reactions": 2,
        "has_technical_analysis": 1,
        "has_creative_interpretations": 2,
        "has_cultural_context": 1,
        "has_comparative_analysis": 1
    }
}
```

## ⚙️ Configuration

### Scoring Weights
- Option-based QA: 30%
- Completeness: 40%  
- Precision: 20%
- Novelty: 10%

### LLM Settings
- Default model: `deepseek-v3`
- Temperature: 0.6 (completeness), 0.3 (precision), 0.4 (novelty)
- Max tokens: 4096

## 🔧 Requirements

```bash
pip install openai pandas numpy
```

## 📝 Example Output

```
COMPREHENSIVE MUSIC APPRAISAL EVALUATION REPORT
================================================================================

1. OPTION-BASED QUESTION ANSWERING
   Accuracy: 0.7500 (75.00%)
   Correct: 3/4

2. COMPLETENESS EVALUATION  
   Score: 13.50/16.00
   Percentage: 84.4%
   Assessment: 内容较为完整，涵盖了主要评价维度

3. PRECISION EVALUATION
   Overall Accuracy: 0.8000 (80.00%)
   Total Claims: 15
   Correct Claims: 12
   Incorrect Claims: 3
   Files Evaluated: 2

4. NOVELTY EVALUATION
   Overall Novelty: 0.7200 (72.00%)
   Music Relevance: 0.85
   Depth Score: 0.72
   Personal Insight: 0.68
   Novel Statements: 25
   Music-Related Ratio: 92.0%

   Insight Types Found:
     Personal Reactions: 2/2
     Technical Analysis: 1/2
     Creative Interpretations: 2/2
     Cultural Context: 1/2
     Comparative Analysis: 1/2

5. OVERALL ASSESSMENT
   Overall Score: 0.7775 (77.75%)

模型综合评估结果：
- 选择题准确率: 75.00% (3/4)
- 完整性评分: 13.5/16.0 (84.4%)
- 事实准确率: 80.00%
- 新颖性得分: 72.00%
- 综合得分: 77.75%
================================================================================
```

## 🚀 Running the Demo

```bash
cd eval_benchmark
python demo_comprehensive_benchmark.py
```

The demo showcases all evaluation components and provides insights into model performance across different dimensions, including novelty assessment.

## 🔍 Key Features

- **Flexible Answer Extraction**: Handles verbose model responses with embedded answers
- **Multi-dimensional Evaluation**: Comprehensive assessment across knowledge, completeness, accuracy, and creativity
- **Detailed Reporting**: Rich output with component-wise analysis and insights
- **Novelty Recognition**: Identifies and categorizes creative insights and personal perspectives
- **Music-Relevance Filtering**: Ensures novel content is actually about music aspects
- **Extensible Architecture**: Easy to add new evaluation components
- **Robust Error Handling**: Graceful handling of API failures and parsing errors
- **Ground Truth Integration**: Uses verified song information for precision and novelty evaluation

## 📋 Future Enhancements

1. **Confidence Calibration**: Evaluate model confidence accuracy across all components
2. **Multi-language Support**: Extend evaluation to other languages
3. **Audio Feature Integration**: Incorporate actual audio analysis results
4. **Comparative Analysis**: Compare multiple models side-by-side
5. **Dynamic Weighting**: Adjust component weights based on use case requirements
6. **Semantic Similarity**: Use embeddings for more nuanced novelty detection 