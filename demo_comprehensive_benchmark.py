#!/usr/bin/env python3
import random
from typing import List
from music_appraisal_benchmark import run_comprehensive_benchmark
# export PYTHONPATH=./:$PYTHONPATH
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def demo_qa_model(audio_path: str, question: str, options: List[str]) -> str:

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("/fs-computility/niuyazhe/lixueyan/zj/Qwen2-Audio/Qwen2-Audio-7B-Instruct", device_map="auto")

    
    user_prompt = f"根据<|AUDIO|>。阅读{question}，选出{options}中正确的选项。"
    system_prompt=open("/fs-computility/niuyazhe/lixueyan/acapella/ReactionEvalBenchmark/prompt/model_qa_sysp.txt", "r").read()

    conversation = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': [
            {"type": "audio", "audio_path": audio_path},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    # 读取本地音频
    audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    audios = [audio]

    inputs = processor(text=text, audios=audio, return_tensors="pt", padding=True)
    # 获取模型所在设备
    device = next(model.parameters()).device
    # 将所有张量转到模型所在设备
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    generate_ids = model.generate(**inputs, max_new_tokens=1024)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response
    
    # answers = [
    #     "After careful listening, I believe the answer is A.",
    #     "My analysis suggests the correct choice is B.",
    #     "Based on the musical features, I think C is the right answer.",
    #     "The audio evidence points to option D."
    # ]
    # return random.choice(answers)


def demo_appraisal_model(audio_path: str) -> str:


    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("/fs-computility/niuyazhe/lixueyan/zj/Qwen2-Audio/Qwen2-Audio-7B-Instruct", device_map="auto")

    
    user_prompt = f"根据提示词分析<|AUDIO|>。"
    system_prompt=open("/fs-computility/niuyazhe/lixueyan/acapella/ReactionEvalBenchmark/prompt/model_appraisal_sysp.txt", "r").read()

    conversation = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': [
            {"type": "audio", "audio_path": audio_path},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    # 读取本地音频
    audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    audios = [audio]

    inputs = processor(text=text, audios=audio, return_tensors="pt", padding=True)
    # 获取模型所在设备
    device = next(model.parameters()).device
    # 将所有张量转到模型所在设备
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    generate_ids = model.generate(**inputs, max_new_tokens=1024)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

#     appraisals = ["""告五人乐队的《爱人错过》是一首极具魔性又充满迷幻摇滚色彩的歌曲，从旋律到歌词都充满了独特的魅力。整首歌以“我肯定在几百年前就说过爱你”作为开篇，瞬间抓住听众的耳朵，既浪漫又带着一丝荒诞的宿命感。  
# 音乐风格上，这首歌融合了迷幻摇滚与流行元素，节奏轻快却又不失迷离感，尤其是重复的旋律和洗脑的副歌，让人一听就忍不住跟着哼唱。编曲上，告五人采用了简洁但富有层次的配器，吉他和鼓点的搭配恰到好处，营造出一种既梦幻又略带戏谑的氛围。  
# 歌词部分更是亮点十足，既有“走过 路过 没遇过 / 回头 转头 还是错”这样充满哲学意味的句子，又突然插入“你妈没有告诉你 / 撞到人要说对不起”这种无厘头的市井幽默，形成强烈的反差感，让人忍俊不禁却又莫名觉得合理。这种“小学生式”的直白表达，反而让歌曲更具记忆点和传播性，甚至成为网络热梗。  
# 演唱方面，主唱犬青和云安的嗓音搭配极具辨识度，犬青的声线清澈透亮，而云安的演绎则带着一丝慵懒和痞气，两人的和声部分更是让整首歌的情感层次更加丰富。  
# MV的视觉呈现也很有创意，以红、蓝、绿三原色为主调，讲述了一个“色盲”视角下的奇幻爱情故事，与歌曲主题完美呼应。  
# 总的来说，《爱人错过》是一首兼具艺术性和流行度的作品，既有深度又足够“接地气”，难怪能成为告五人的代表作之一，并在各大音乐平台和短视频中疯狂传播。"""]

#     return random.choice(appraisals)


def main():
    print("Music Appraisal Comprehensive Benchmark Demo")
    print("=" * 60)
    print("This demo showcases all evaluation components:")
    print("1. Option-based QA evaluation")
    print("2. LLM-based completeness scoring")
    print("3. Precision evaluation against ground truth")
    print("4. Novelty/detail assessment")
    print("5. Overall performance assessment")
    print("=" * 60)
    
    random.seed(42)
    
    try:
        # Run the comprehensive benchmark with all components enabled
        result = run_comprehensive_benchmark(
            qa_model_function=demo_qa_model,
            appraisal_model_function=demo_appraisal_model,
            qa_data_path="data/option_qa.jsonl",
            song_details_path="data/song_details.jsonl",
            output_path="comprehensive_demo_results.json",
            enable_precision_eval=True,  # Enable precision evaluation
            enable_novelty_eval=True     # Enable novelty evaluation
        )
        
        print("\nDemo completed successfully!")
        print("Check 'comprehensive_demo_results.json' for detailed results.")
        
        # Show detailed analysis
        print(f"\n" + "="*60)
        print("🎵 DEMO ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📊 Question Answering Performance:")
        print(f"   - Accuracy: {result.qa_accuracy:.1%}")
        print(f"   - Questions answered: {result.qa_correct_answers}/{result.qa_total_questions}")

        print(f"📝 Appraisal Text:")
        print(f"   - {result.appraisal_text}")
        
        print(f"\n📝 Completeness Evaluation:")
        print(f"   - Score: {result.completeness_score:.1f}/16.0")
        print(f"   - Percentage: {result.completeness_score/16*100:.0f}%")
        
        if result.precision_score is not None:
            print(f"\n🎯 Precision Evaluation:")
            print(f"   - Factual accuracy: {result.precision_score:.1%}")
            if result.precision_details:
                details = result.precision_details
                print(f"   - Total factual claims: {details.get('total_claims', 0)}")
                print(f"   - Correct claims: {details.get('total_correct_claims', 0)}")
                print(f"   - Incorrect claims: {details.get('total_incorrect_claims', 0)}")
        else:
            print(f"\n🎯 Precision Evaluation: Skipped (no ground truth)")
        
        if result.novelty_score is not None:
            print(f"\n✨ Novelty Evaluation:")
            print(f"   - Overall novelty: {result.novelty_score:.1%}")
            if result.novelty_details:
                details = result.novelty_details
                print(f"   - Music relevance: {details.get('avg_music_relevance_score', 0):.1%}")
                print(f"   - Depth score: {details.get('avg_depth_score', 0):.1%}")
                print(f"   - Personal insight: {details.get('avg_personal_insight_score', 0):.1%}")
                print(f"   - Novel statements: {details.get('total_novel_statements', 0)}")
                
                # Show insight types found
                insight_counts = details.get('insight_type_counts', {})
                files_count = details.get('files_evaluated', 1)
                print(f"\n   💡 Insight Types Found:")
                print(f"     🎭 Personal reactions: {insight_counts.get('has_personal_reactions', 0)}/{files_count}")
                print(f"     🔧 Technical analysis: {insight_counts.get('has_technical_analysis', 0)}/{files_count}")
                print(f"     🎨 Creative interpretations: {insight_counts.get('has_creative_interpretations', 0)}/{files_count}")
                print(f"     🌍 Cultural context: {insight_counts.get('has_cultural_context', 0)}/{files_count}")
                print(f"     📊 Comparative analysis: {insight_counts.get('has_comparative_analysis', 0)}/{files_count}")
        else:
            print(f"\n✨ Novelty Evaluation: Skipped (no ground truth)")
        
        print(f"\n🏆 Overall Performance: {result.overall_score:.1%}")
        
        print(f"\n" + "="*60)
        print("🔍 EVALUATION INSIGHTS:")
        print("="*60)
        print("• QA component tests specific music knowledge and comprehension")
        print("• Completeness scoring evaluates comprehensive coverage of required elements")
        print("• Precision evaluation checks factual accuracy against verified information")
        print("• Novelty assessment rewards personal insights and creative interpretations")
        print("• Combined metrics provide holistic assessment of music appraisal quality")
        
        # Provide performance insights
        if result.precision_details:
            avg_precision = result.precision_details.get('average_score', 0)
            if avg_precision > 0.8:
                print("✅ High precision - model makes accurate factual claims")
            elif avg_precision > 0.5:
                print("⚠️  Medium precision - some factual errors detected")
            else:
                print("❌ Low precision - significant factual inaccuracies")
        
        if result.novelty_details:
            avg_novelty = result.novelty_details.get('avg_novelty_score', 0)
            if avg_novelty > 0.7:
                print("🌟 High novelty - rich personal insights and creative interpretations")
            elif avg_novelty > 0.4:
                print("💫 Medium novelty - some personal elements and depth")
            else:
                print("📝 Low novelty - mostly basic factual content")
        
        print(f"\n🎯 The benchmark successfully evaluates music appraisal models across")
        print(f"   knowledge accuracy, content completeness, factual precision, and creative depth!")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 