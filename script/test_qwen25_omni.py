import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Tuple
import re
import time
import json
import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


def check_gpu_memory():
    """Check and print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {gpu_memory:.2f}GB total")


def cleanup_gpu_memory():
    """Aggressively clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Set environment variables for better CUDA error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Add debugging function
def debug_inputs(inputs, processor, prefix=""):
    """Debug function to check input tensor validity"""
    print(f"{prefix} Input debugging:")
    if hasattr(inputs, 'input_ids'):
        print(f"  input_ids shape: {inputs.input_ids.shape}")
        print(f"  input_ids max: {inputs.input_ids.max().item()}")
        print(f"  input_ids min: {inputs.input_ids.min().item()}")
        print(f"  input_ids dtype: {inputs.input_ids.dtype}")
        print(f"  input_ids device: {inputs.input_ids.device}")
        print(f"  processor vocab_size: {processor.tokenizer.vocab_size}")
        
        # Check tensor integrity
        if torch.isnan(inputs.input_ids).any():
            print(f"  WARNING: Found NaN values in input_ids!")
        if torch.isinf(inputs.input_ids).any():
            print(f"  WARNING: Found Inf values in input_ids!")
    
    if hasattr(inputs, 'audio_values') and inputs.audio_values is not None:
        print(f"  audio_values shape: {inputs.audio_values.shape}")
        print(f"  audio_values dtype: {inputs.audio_values.dtype}")
        print(f"  audio_values device: {inputs.audio_values.device}")
        print(f"  audio_values min/max: {inputs.audio_values.min().item():.4f}/{inputs.audio_values.max().item():.4f}")
        
        # Check audio tensor integrity
        if torch.isnan(inputs.audio_values).any():
            print(f"  WARNING: Found NaN values in audio_values!")
        if torch.isinf(inputs.audio_values).any():
            print(f"  WARNING: Found Inf values in audio_values!")
    
    if hasattr(inputs, 'attention_mask'):
        print(f"  attention_mask shape: {inputs.attention_mask.shape}")
        print(f"  attention_mask sum: {inputs.attention_mask.sum().item()}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        print(f"  GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    
    return True


def safe_generate(model, inputs, **generation_kwargs):
    """Safe generation wrapper with detailed CUDA error handling"""
    try:
        print("Attempting model generation...")
        
        # Check input tensor dimensions and ranges
        if hasattr(inputs, 'input_ids'):
            seq_len = inputs.input_ids.shape[1]
            print(f"  Sequence length: {seq_len}")
            
            # Check for any obviously problematic values
            unique_tokens = torch.unique(inputs.input_ids)
            print(f"  Unique token count: {len(unique_tokens)}")
            print(f"  Token range: {unique_tokens.min().item()} to {unique_tokens.max().item()}")
        
        # Try generation with minimal parameters first
        default_kwargs = {
            'max_new_tokens': 50,  # Very conservative
            'do_sample': False,
            'num_beams': 1,
            'pad_token_id': getattr(model.config.thinker_config, 'pad_token_id', 151643)
        }
        default_kwargs.update(generation_kwargs)
        
        print(f"  Generation kwargs: {default_kwargs}")
        
        with torch.no_grad():
            text_ids, audio = model.generate(**inputs, **default_kwargs)
            
        print("  Generation successful!")
        return text_ids, audio
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"  RuntimeError during generation: {error_msg}")
        
        if "indexSelectLargeIndex" in error_msg:
            print("  DETECTED: IndexSelectLargeIndex error - likely embedding index out of bounds")
            print("  This suggests a token ID is referencing an invalid embedding index")
            
        elif "device-side assert" in error_msg:
            print("  DETECTED: Device-side assertion failure")
            print("  This often indicates index out of bounds in CUDA kernels")
            
        elif "CUDA" in error_msg:
            print(f"  DETECTED: Other CUDA error: {error_msg}")
            
        # Re-raise the error for proper handling upstream
        raise e


# model_path = "/fs-computility/niuyazhe/shared/Qwen2.5-Omni-7B"
# model_path = "/fs-computility/niuyazhe/lixueyan/acapella/LLaMA-Factory/scripts/merged_model_checkpoint_0627"
model_path = "/fs-computility/niuyazhe/lixueyan/acapella/LLaMA-Factory/scripts/merged_model_checkpoint_add_gemini_0731"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    #attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


def _normalize_answer(answer: str, options: Optional[List[str]] = None) -> str:
    """
    Normalize answer for comparison (handles A/B/C/D/... and option text).
    
    Args:
        answer: The raw answer string from the model
        options: List of available options to match against
    
    Returns:
        Normalized answer (letter A-Z or original text)
    """
    if not isinstance(answer, str):
        return str(answer).strip().upper()
    
    answer = answer.strip()
    
    # Determine the valid option letters based on number of options
    if options:
        num_options = len(options)
        valid_letters = [chr(ord('A') + i) for i in range(num_options)]
    else:
        # Default to A-D if no options provided (backward compatibility)
        valid_letters = ['A', 'B', 'C', 'D']
    
    # First, try to extract option letters from the response using regex
    # Look for patterns like "A", "option A", "choice A", "(A)", "A.", "A)", etc.
    letter_pattern = '|'.join(valid_letters)
    patterns = [
        # Remove the problematic single letter pattern or make it more restrictive
        # rf'\b({letter_pattern})\b',  # Single letter - too broad, causes issues with R&B/Soul
        rf'(?:option|choice|answer|select|选择|答案)\s*({letter_pattern})\b',  # "option A", "choice B", etc.
        rf'\(({letter_pattern})\)',  # "(A)", "(B)", etc.
        rf'\[({letter_pattern})\]',  # "[A]", "[B]", etc. - only single letters
        rf'\b({letter_pattern})[.)]\s',  # "A. ", "B) ", etc.
        rf'^({letter_pattern})[.)]\s',  # Starting with "A. " or "B) "
        rf'(?<!\[)(?<!\w)({letter_pattern})[：:]',  # "A:", "B：" but not inside brackets or after word chars
        rf'^\s*({letter_pattern})\s*$',  # Only standalone single letters (whole string is just the letter)
    ]
    
    # Try each pattern to extract the letter
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            # Return the first valid match
            first_match = matches[0].upper()
            if first_match in valid_letters:
                return first_match
    
    # PRIORITY: If we have options, try to match the answer content against option text first
    # This includes content from square brackets like [Rock], [EDM], etc.
    if options:
        answer_clean = answer.lower().strip()
        
        # Extract content from square brackets [content] for option matching
        bracket_matches = re.findall(r'\[([^\]]+)\]', answer)
        if bracket_matches:
            # Use bracket content for matching if it's not a single letter
            bracket_content = bracket_matches[0].strip()
            if not (len(bracket_content) == 1 and bracket_content.upper() in valid_letters):
                # Remove trailing colons or other punctuation from bracket content
                bracket_content = re.sub(r'[：:.,!?;]$', '', bracket_content)
                answer_clean = bracket_content.lower().strip()
        
        # Remove common punctuation and extra spaces
        answer_clean = re.sub(r'[，。！？；：""''（）()[\]{}.,!?;:"\'()\[\]{}]', '', answer_clean)
        answer_clean = re.sub(r'\s+', ' ', answer_clean).strip()
        
        for i, option in enumerate(options):
            option_clean = option.lower().strip()
            # Remove common punctuation and extra spaces  
            option_clean = re.sub(r'[，。！？；：""''（）()[\]{}.,!?;:"\'()\[\]{}]', '', option_clean)
            option_clean = re.sub(r'\s+', ' ', option_clean).strip()
            
            # Check for exact match
            if answer_clean == option_clean:
                return chr(ord('A') + i)
            
            # Check for substring match (option in answer or answer in option)
            if option_clean and (option_clean in answer_clean or answer_clean in option_clean):
                # Additional check: make sure it's not a very short match that could be coincidental
                if len(option_clean) >= 2 or len(answer_clean) >= 2:
                    return chr(ord('A') + i)
            
            # Check for fuzzy matching for similar terms
            # Split into words and check for word-level matches
            answer_words = set(answer_clean.split())
            option_words = set(option_clean.split())
            
            if answer_words and option_words:
                # Calculate word overlap ratio
                overlap = len(answer_words.intersection(option_words))
                union = len(answer_words.union(option_words))
                if union > 0 and overlap / union > 0.5:  # More than 50% word overlap
                    return chr(ord('A') + i)

    # If no option content matches, try to find valid letters anywhere in the string
    answer_upper = answer.upper()
    for letter in valid_letters:
        if not re.search(rf'\[[\w\s]*{letter}[\w\s]*\]', answer_upper) and re.search(rf'\b{letter}\b', answer_upper):
            # Check if it's likely to be the answer choice (not part of another word)
            # Look for the letter with word boundaries or common separators
            # But exclude letters that are inside square brackets
            return letter
    
    # As a fallback, check if the answer is already a valid letter
    answer_clean = answer.upper().strip()
    if answer_clean in valid_letters:
        return answer_clean
        
    # If still no match, return the cleaned original answer
    # This allows for text-based matching if needed
    return answer_clean


@torch.no_grad()
def my_qa_model(audio_path: str, question: str, options: List[str]) -> str:
    from qwen_omni_utils import process_mm_info
    
    try:
        print(f"\n=== Processing: {audio_path} ===")
        
        # Clear GPU cache before processing
        torch.cuda.empty_cache()
        
        # Check if audio file exists and is valid
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file does not exist: {audio_path}")
            return "A"  # Return default answer
            
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question + f'\n请你从下列四个选项中选择一个回答：{options}，最终输出格式为：[A/B/C/D:]'},
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]

        # Preparation for inference
        print("Applying chat template...")
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        print(f"Template length: {len(text)}")
        
        # Add error handling for audio processing
        print("Processing audio...")
        try:
            audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
            print("Audio processing successful")
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return "A"  # Return default answer
            
        print("Processing inputs...")
        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
        
        # Debug inputs before moving to device
        if not debug_inputs(inputs, processor, "Before device transfer"):
            print("Invalid inputs detected, returning default answer")
            return "A"
        
        inputs = inputs.to(model.device).to(model.dtype)
        
        # Debug inputs after moving to device
        debug_inputs(inputs, processor, "After device transfer")
        
        print("Starting generation...")

        # Inference: Generation of the output text and audio
        text_ids, audio = safe_generate(model, inputs)
        print("Generation completed successfully")
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
        ]
        text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer_clean = _normalize_answer(text[0], options)
        print(f'Generated text: {text[0][:100]}..., answer: {answer_clean}')
        
        return answer_clean
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error in my_qa_model for {audio_path}: {e}")
            # Clear cache and try to recover
            torch.cuda.empty_cache()
            return "A"  # Return default answer
        else:
            raise e
    except Exception as e:
        print(f"Unexpected error in my_qa_model for {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return "A"  # Return default answer
    finally:
        # Always clear cache after processing
        torch.cuda.empty_cache()


@torch.no_grad()
def my_appraisal_model(audio_path: str) -> str:
    from qwen_omni_utils import process_mm_info

    try:
        # Clear GPU cache before processing
        torch.cuda.empty_cache()
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file does not exist: {audio_path}")
            return "Unable to process audio file."

        prompt = """
        # 你是一位音乐评论者\n\n## 评论风格\n一位学院派出身的音乐高级教师，拥有正统声乐训练背景与扎实的音乐理论素养，能够从演唱、编曲、和声、情绪处
        理等多个维度层层解析作品，也能够在专业与共鸣之间找到精准的平衡点。能够指出编曲和歌手技巧，进一步解释这些技巧在情绪传达中起到了什么作用。语言风格属于低情绪波动中的高共
        情类型，语速中等，逻辑清晰，结构分明。善于通过温和但坚定的语气，将复杂的音乐结构解释给观众。经常使用对比、演绎性转述和结构拆解等语言方式。像一个老师稳扎稳打地带学生听
        完整首歌，获得扎实的理解。偏好结构清晰、表达合理的作品。容易被编排得体的抒情段落、情绪不外露但有层次的表达、整体和谐的表演所打动。会直接批评结构混乱、编曲奇怪以及歌曲
        和演唱不和谐的段落。推崇\"少即是多\"的哲学，强调克制感与内敛的张力。习惯于将歌曲连接到自己的教学案例、经典演出、教材范式之中。经常引用历史作品、典型唱段进行对照式讲解
        。熟悉网络语境，但不会刻意迎合，偶尔带入一些轻松用语。喜欢把听到的作品放入更大的文化背景中理解。
        """
        # TODO: add song info in prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]

        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        try:
            audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return "Unable to process audio file."
            
        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        
        # Add validation for input tensors
        if hasattr(inputs, 'input_ids') and inputs.input_ids.max() >= processor.tokenizer.vocab_size:
            print(f"Warning: Invalid token ID detected in appraisal model")
            return "Unable to process audio due to tokenization error."
            
        # Add generation parameters to prevent memory issues
        generation_kwargs = {
            'max_length': 1024,  # Reasonable length for appraisal
            'do_sample': True,
            'temperature': 0.7,
            'num_beams': 1,
        }

        # Inference: Generation of the output text and audio
        text_ids, audio = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
        ]
        text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text[0]
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error in my_appraisal_model for {audio_path}: {e}")
            torch.cuda.empty_cache()
            return "CUDA error occurred during audio processing."
        else:
            raise e
    except Exception as e:
        print(f"Unexpected error in my_appraisal_model for {audio_path}: {e}")
        return "Unexpected error occurred during audio processing."
    finally:
        # Always clear cache after processing
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from datetime import datetime
    
    print("Starting benchmark with GPU memory monitoring...")
    check_gpu_memory()
    
    # from audio_qa import run_benchmark
    
    # qa_result = run_benchmark(
    #     model_function=my_qa_model,
    #     data_path="data/option_qa.jsonl",
    #     output_path=f"qa_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # )
    
    try:
        from music_appraisal_benchmark import run_comprehensive_benchmark

        # Clean up memory before starting
        cleanup_gpu_memory()
        print("Memory cleaned. Starting comprehensive benchmark...")
        check_gpu_memory()

        result = run_comprehensive_benchmark(
            qa_model_function=my_qa_model,
            appraisal_model_function=my_appraisal_model,
            qa_data_path="data/option_qa.jsonl",
            song_details_path="data/song_details.jsonl",
            output_path=f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        print(f"Overall Score: {result.overall_score:.2%}")
        
    except Exception as e:
        print(f"Error during benchmark execution: {e}")
        print("Final GPU memory state:")
        check_gpu_memory()
        # Try to clean up and save any partial results
        cleanup_gpu_memory()
        raise e
    finally:
        print("Final cleanup...")
        cleanup_gpu_memory()
        check_gpu_memory()
