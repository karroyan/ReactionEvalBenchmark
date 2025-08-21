from typing import List, Optional, Tuple, NamedTuple, Union
import os
import re
import time
import json
import numpy as np
import torch
import librosa
import base64
import requests
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser
import sys
import gc
sys.path.append("/fs-computility/niuyazhe/shared/lixueyan/acapella/ReactionEvalBenchmark")

# Get model path from command line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]

class ModelHandler:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_api = "fs-computility" not in model_path
        if self.is_api:
            self.api_url = "https://poloai.top/v1/chat/completions"
            self.headers = {
                'Accept': '',
                'Authorization': os.getenv("API_KEY"),
                'Content-Type': 'application/json'
            }
        else:
            self.llm = LLM(
                model=model_path,
                max_model_len=8192,
                max_num_seqs=5,
                limit_mm_per_prompt={
                    "audio": 1,
                },
            )
            self.sampling_params = SamplingParams(temperature=0, max_tokens=256)

    def encode_audio(self, audio_path: str) -> str:
        with open(audio_path, 'rb') as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')

    def generate(self, query_result: 'QueryResult') -> str:
        if self.is_api:
            # Prepare audio data for API
            audio_path = query_result.inputs["multi_modal_data"]["audio"][0][0]
            audio_base64 = self.encode_audio(audio_path)

            payload = {
                "model": self.model_path,
                "modalities": ["text", "audio"],
                "audio": {
                    "voice": "alloy",
                    "format": "wav"
                },
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": default_system}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query_result.inputs["prompt"]
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "mp3"
                                }
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )

            print(f'response:{response.text}')
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            outputs = self.llm.generate(query_result.inputs, sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text

class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]

default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
)

# model_path = "/fs-computility/niuyazhe/shared/Qwen2.5-Omni-7B"



def _normalize_answer(answer: str, options: Optional[List[str]] = None) -> str:
    """
    Normalize answer for comparison using the following logic:
    1. First try to match \boxed{A/B/C/D/E/F/G/H} format
    2. Try to match the content from options
    3. Try to match A/B/C/D/E/F/G/H directly in the text
    4. If all above fail, return empty string (indicating wrong answer)
    
    Args:
        answer: The raw answer string from the model
        options: List of available options to match against
    
    Returns:
        Normalized answer (letter A-H or empty string for wrong answer)
    """
    if not isinstance(answer, str):
        return ""
    
    answer = answer.strip()
    
    # Determine valid letters based on number of options
    if options:
        num_options = len(options)
        valid_letters = [chr(ord('A') + i) for i in range(num_options)]
    else:
        valid_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # 1. Try to match \boxed{A} format
    boxed_pattern = r'\\boxed\{([A-H])\}'
    boxed_matches = re.findall(boxed_pattern, answer)
    if boxed_matches:
        first_match = boxed_matches[0].upper()
        if first_match in valid_letters:
            return first_match
    
    # 2. Try to match option content
    if options:
        answer_clean = answer.lower().strip()
        # Remove common punctuation and extra spaces
        answer_clean = re.sub(r'[，。！？；：""''（）()[\]{}.,!?;:"\'()\[\]{}]', '', answer_clean)
        answer_clean = re.sub(r'\s+', ' ', answer_clean).strip()
        
        for i, option in enumerate(options):
            option_clean = option.lower().strip()
            # Remove common punctuation and extra spaces
            option_clean = re.sub(r'[，。！？；：""''（）()[\]{}.,!?;:"\'()\[\]{}]', '', option_clean)
            option_clean = re.sub(r'\s+', ' ', option_clean).strip()
            
            # Check for exact match or substring match
            if answer_clean == option_clean or option_clean in answer_clean:
                return chr(ord('A') + i)
    
    # 3. Try to match A/B/C/D/E/F/G/H directly in the text
    letter_pattern = '|'.join(valid_letters)
    direct_matches = re.findall(rf'\b({letter_pattern})\b', answer.upper())
    if direct_matches:
        first_match = direct_matches[0]
        if first_match in valid_letters:
            return first_match
    
    # 4. If no match found, return empty string (indicating wrong answer)
    return ""

def get_multi_audios_query(text, audio, is_api=False) -> QueryResult:
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # For API calls, we only need the audio path
    if is_api:
        return QueryResult(
            inputs={
                "prompt": prompt,
                "multi_modal_data": {
                    "audio": [
                        [audio, None]  # Match the format of librosa.load() but use path directly
                    ],
                },
            },
            limit_mm_per_prompt={
                "audio": 1,
            },
        )
    
    # For local model, use librosa to load audio data
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [
                    librosa.load(audio, sr=None)
                ],
            },
        },
        limit_mm_per_prompt={
            "audio": 1,
        },
    )


@torch.no_grad()
def my_qa_model(audio_path: str, question: str, options: List[str]) -> str:
    text = question + f"""
请你从下列四个选项中选择一个回答：{options}。
你必须先经过推理，最后只给出你认为最正确的选项，
并用 LaTeX 格式 \\boxed{{选项字母}} 表示，例如：\\boxed{{A}},\\boxed{{B}},\\boxed{{C}},\\boxed{{D}}。
最终答案必须单独占一行，且不包含其他内容。
"""

    # Preparation for inference
    query_result = get_multi_audios_query(text, audio_path, is_api=model_handler.is_api)
    generated_text = model_handler.generate(query_result)
    answer_clean = _normalize_answer(generated_text, options)
    return answer_clean


@torch.no_grad()
def my_appraisal_model(audio_path: str) -> str:
    prompt = """
    # 你是一位音乐评论者\n\n## 评论风格\n一位学院派出身的音乐高级教师，拥有正统声乐训练背景与扎实的音乐理论素养，能够从演唱、编曲、和声、情绪处
    理等多个维度层层解析作品，也能够在专业与共鸣之间找到精准的平衡点。能够指出编曲和歌手技巧，进一步解释这些技巧在情绪传达中起到了什么作用。语言风格属于低情绪波动中的高共
    情类型，语速中等，逻辑清晰，结构分明。善于通过温和但坚定的语气，将复杂的音乐结构解释给观众。经常使用对比、演绎性转述和结构拆解等语言方式。像一个老师稳扎稳打地带学生听
    完整首歌，获得扎实的理解。偏好结构清晰、表达合理的作品。容易被编排得体的抒情段落、情绪不外露但有层次的表达、整体和谐的表演所打动。会直接批评结构混乱、编曲奇怪以及歌曲
    和演唱不和谐的段落。推崇\"少即是多\"的哲学，强调克制感与内敛的张力。习惯于将歌曲连接到自己的教学案例、经典演出、教材范式之中。经常引用历史作品、典型唱段进行对照式讲解
    。熟悉网络语境，但不会刻意迎合，偶尔带入一些轻松用语。喜欢把听到的作品放入更大的文化背景中理解。
    """
    # TODO: add song info in prompt

    # Preparation for inference
    query_result = get_multi_audios_query(prompt, audio_path, is_api=model_handler.is_api)
    generated_text = model_handler.generate(query_result)
    return generated_text


if __name__ == "__main__":
    from datetime import datetime
    from music_appraisal_benchmark import run_comprehensive_benchmark

    print(f"Running benchmark for {model_path}")
    
    try:
        # Initialize model handler
        global model_handler
        model_handler = ModelHandler(model_path)
        
        model_name = os.path.basename(model_path)
        result = run_comprehensive_benchmark(
            qa_model_function=my_qa_model,
            appraisal_model_function=my_appraisal_model,
            qa_data_path="data/option_qa.jsonl",
            song_details_path="data/song_details.jsonl",
            output_path=f"script/result_to_test/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        print(f"Overall Score: {result.overall_score:.2%}")

    except Exception as e:
        print(f"Error processing model: {str(e)}")
        sys.exit(1)

        