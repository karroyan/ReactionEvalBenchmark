# Music Appraisal Evaluation Benchmark

A comprehensive benchmark system for evaluating AI models' capabilities in music analysis and appraisal. This benchmark tests models across multiple dimensions including factual knowledge, content completeness, precision, and creative insights.

## 🎯 Overview

This benchmark system evaluates music appraisal models through four complementary evaluation components:

1. **📝 Option-based Question Answering** - Tests specific music knowledge and comprehension
2. **🔍 LLM-based Completeness Scoring** - Evaluates comprehensive coverage of required elements  
3. **🎯 Precision Evaluation** - Checks factual accuracy against verified information
4. **✨ Novelty Assessment** - Rewards personal insights and creative interpretations

## 🌟 Key Features

- **Multi-dimensional Evaluation**: Holistic assessment across knowledge, completeness, accuracy, and creativity
- **Flexible Architecture**: Modular design allows using individual components or the full comprehensive benchmark
- **LLM-powered Scoring**: Advanced evaluation using large language models for nuanced assessment
- **Detailed Analytics**: Comprehensive reporting with breakdowns across all evaluation dimensions
- **Easy Integration**: Simple API for evaluating any audio model that follows the specified interface

## 🚀 Quick Start

### Basic Usage

```python
from typing import List
from music_appraisal_benchmark import run_comprehensive_benchmark

# Define your model functions
def my_qa_model(audio_path: str, question: str, options: List[str]) -> str:
    # Your QA model implementation
    return "A"  # Should return A/B/C/D or option text

def my_appraisal_model(audio_path: str) -> str:
    # Your music appraisal model implementation  
    return "This song is a beautiful piece with..."

# Run comprehensive evaluation
result = run_comprehensive_benchmark(
    qa_model_function=my_qa_model,
    appraisal_model_function=my_appraisal_model,
    qa_data_path="data/option_qa.jsonl",
    song_details_path="data/song_details.jsonl",
    output_path="results.json"
)

print(f"Overall Score: {result.overall_score:.2%}")
```

### Individual Component Usage

```python
# Use only QA evaluation
from audio_qa import run_benchmark

qa_result = run_benchmark(
    model_function=my_qa_model,
    data_path="data/option_qa.jsonl"
)

# Use only completeness evaluation
from llm_evaluator import AudioLLMEvaluator

evaluator = AudioLLMEvaluator(api_key="your-key", base_url="your-url")
completeness_result = evaluator.evaluate_appraisal("Your music appraisal text...")
```

## 📁 Project Structure

```
ReactionEvalBenchmark/
├── audio_qa.py                    # Option-based QA evaluation
├── llm_evaluator.py              # LLM-based completeness scoring
├── precision_evaluator.py        # Factual accuracy evaluation  
├── novelty_evaluator.py         # Creative insight assessment
├── music_appraisal_benchmark.py # Main comprehensive benchmark
├── demo_comprehensive_benchmark.py # Usage examples
├── prompt/
│   └── prompt_llm_score.txt     # LLM scoring prompts
└── data/
    ├── option_qa.jsonl          # QA questions dataset
    └── song_details.jsonl       # Ground truth song information
```

## 📊 Evaluation Components

### 1. Option-based Question Answering
- **Purpose**: Tests specific music knowledge and listening comprehension
- **Format**: Multiple choice questions (A/B/C/D) about audio content
- **Metrics**: Accuracy, response time, confidence scores (optional)

### 2. LLM-based Completeness Scoring  
- **Purpose**: Evaluates whether appraisals contain all required elements
- **Scoring**: 16-point structured rubric covering various aspects
- **Metrics**: Total score, percentage completeness, qualitative assessment

### 3. Precision Evaluation
- **Purpose**: Checks factual accuracy against verified ground truth
- **Method**: LLM-based fact-checking of specific claims
- **Metrics**: Precision score, correct/incorrect claim counts, detailed breakdown

### 4. Novelty Assessment
- **Purpose**: Rewards personal insights and creative interpretations beyond basic facts
- **Categories**: Personal reactions, technical analysis, creative interpretations, cultural context, comparative analysis
- **Metrics**: Novelty score, depth analysis, insight type identification

## 🔧 Requirements

```bash
# Core dependencies
pip install -r requirements.txt
```

### Environment Setup

1. **LLM API Access**: Configure API credentials for evaluation components
```python
LLM_API_KEY = "your-api-key"
LLM_BASE_URL = "your-api-endpoint"  
LLM_MODEL = "deepseek-v3"  # or your preferred model
```

2. **Data Format**: Prepare your evaluation data in JSONL format

**QA Data Format (`option_qa.jsonl`)**:
```json
{"audio_path": "path/to/audio.wav", "question": "What genre is this song?", "options": ["Pop", "Rock", "Jazz", "Classical"], "answer": "A", "metadata": {}}
```

**Song Details Format (`song_details.jsonl`)**:
```json
{"audio_path": "path/to/audio.wav", "artist": "Artist Name", "title": "Song Title", "release_year": "2023", "genre": "Pop", "description": "Song description..."}
```

## 📈 Scoring System

### Overall Score Calculation
- **QA Accuracy**: 30% weight
- **Completeness**: 40% weight  
- **Precision**: 20% weight
- **Novelty**: 10% weight

### Performance Interpretation
- **🏆 Excellent (>80%)**: High-quality comprehensive music analysis
- **✅ Good (60-80%)**: Solid performance with room for improvement
- **⚠️ Fair (40-60%)**: Basic functionality, significant improvements needed
- **❌ Poor (<40%)**: Substantial issues across multiple dimensions

## 🎵 Demo Example

Run the included demo to see the benchmark in action:

```bash
python demo_comprehensive_benchmark.py
```

The demo will:
- Evaluate sample models on all components
- Show detailed breakdowns for each evaluation dimension
- Demonstrate the scoring methodology
- Generate comprehensive analysis reports

## 🔬 Advanced Usage

### Custom Evaluation Components

```python
# Enable/disable specific evaluation components
result = run_comprehensive_benchmark(
    qa_model_function=my_qa_model,
    appraisal_model_function=my_appraisal_model,
    enable_precision_eval=True,   # Enable precision evaluation
    enable_novelty_eval=True,     # Enable novelty evaluation
    include_qa_confidence=True    # Include confidence scoring
)
```

### Detailed Analysis

```python
# Get comprehensive statistics
from music_appraisal_benchmark import MusicAppraisalBenchmark

benchmark = MusicAppraisalBenchmark(qa_data_path, llm_api_key, llm_base_url)
result = benchmark.evaluate_model(qa_model, appraisal_model)

# Generate detailed report
benchmark.print_detailed_report(result)

# Save results
benchmark.save_results(result, "detailed_results.json")
```

## 🏗️ Model Interface Requirements

Your models must implement the following interfaces:

### QA Model Function
```python
def qa_model(audio_path: str, question: str, options: List[str]) -> str:
    """
    Args:
        audio_path: Path to the audio file
        question: The question to answer
        options: List of possible answer options
    
    Returns:
        The predicted answer (A/B/C/D or option text)
    """
    pass
```

### Appraisal Model Function
```python
def appraisal_model(audio_path: str) -> str:
    """
    Args:
        audio_path: Path to the audio file
    
    Returns:
        A comprehensive music appraisal text
    """
    pass
```

### Optional: Confidence-enabled QA Model
```python
def qa_model_with_confidence(audio_path: str, question: str, options: List[str]) -> Tuple[str, float]:
    """
    Returns:
        Tuple of (predicted_answer, confidence_score)
    """
    pass
```

## 📋 Example Output

```
🎵 COMPREHENSIVE EVALUATION SUMMARY
=====================================
📊 Question Answering Performance:
   - Accuracy: 85.2%
   - Questions answered: 23/27

📝 Completeness Evaluation:
   - Score: 14.1/16.0
   - Percentage: 88%

🎯 Precision Evaluation:
   - Factual accuracy: 91.3%
   - Total factual claims: 46
   - Correct claims: 42
   - Incorrect claims: 4

✨ Novelty Evaluation:
   - Overall novelty: 73.5%
   - Music relevance: 89.2%
   - Depth score: 67.8%
   - Personal insight: 71.1%
   - Novel statements: 28

🏆 Overall Performance: 84.7%
```

## 🤝 Contributing

We welcome contributions to improve the benchmark system:

1. **Add New Evaluation Dimensions**: Extend the framework with additional assessment criteria
2. **Improve Scoring Methods**: Enhance the precision and reliability of evaluation metrics  
3. **Expand Dataset**: Contribute high-quality QA pairs and ground truth data
4. **Optimize Performance**: Improve efficiency and scalability of evaluation processes

## 📄 License

This project is released under the MIT License.

## 🙏 Acknowledgments

This benchmark system is designed to advance research in automated music analysis and appraisal. We thank the contributors who helped develop and validate the evaluation methodologies.

---

**🎼 Ready to evaluate your music appraisal models? Get started with the comprehensive benchmark today!**
