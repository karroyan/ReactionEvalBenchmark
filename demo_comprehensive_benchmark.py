#!/usr/bin/env python3
import random
from typing import List
from music_appraisal_benchmark import run_comprehensive_benchmark


def demo_qa_model(audio_path: str, question: str, options: List[str]) -> str:
    """
    A sample of answer to qa question
    """
    # For demo purposes, use some simple heuristics
    question_lower = question.lower()
    
    answers = [
        "After careful listening, I believe the answer is A.",
        "My analysis suggests the correct choice is B.",
        "Based on the musical features, I think C is the right answer.",
        "The audio evidence points to option D."
    ]
    return random.choice(answers)


def demo_appraisal_model(audio_path: str) -> str:
    """
    A sample of answer to reaction question
    """
    # Generate different types of appraisals to test all evaluation components
    # appraisals = [
    #     # High quality appraisal
    #     """这首《倒带》是蔡依林在2004年发布的经典作品，收录在她的专辑《城堡》中。但这首歌对我来说不仅仅是一首流行歌曲，它像是一台情感的时光机。
    #         听到这首歌的第一秒，我就仿佛被带回到了那个青涩的年代。蔡依林的声音在这里有种特殊的温柔，就像丝绸划过水面一样顺滑，又带着一种淡淡的忧郁。特别是副歌部分"如果时间能够倒带"，她的音色仿佛在耳边轻抚，让人不禁想起自己的初恋回忆。
    #         从制作角度来说，这首歌的编曲层次非常丰富。鼓点的处理特别巧妙，每一下击打都像心跳一样规律而深沉，但又不会抢夺人声的风头。弦乐的运用更是画龙点睛，在关键的情感转折处，那些弦乐就像眼泪一样缓缓流淌下来。
    #         我觉得这首歌和王菲的《红豆》有异曲同工之妙，都是用简单的旋律承载复杂的情感。但《倒带》更多了一种青春的躁动，那种想要重新来过但又无能为力的矛盾心情，被表达得淋漓尽致。
    #         从文化角度来看，这首歌出现在2004年，正好是华语流行音乐黄金时代的尾声。它既保留了90年代情歌的深沉，又融入了新世纪的制作理念，可以说是承上启下的经典之作。
    #         每次听到这首歌，我都会想起那个下着小雨的傍晚，我和朋友在KTV里一遍遍地循环播放，那种青春的美好和遗憾交织在一起的感觉，就像歌词里说的那样，真希望时间能够倒带。""",
        
    #     # Medium novelty
    #     """蔡依林的《倒带》确实是她转型期的代表作之一。这首歌在2004年推出时，我还在上高中，当时就觉得这首歌有种特别的魅力。
    #         音乐制作上，整首歌的编曲很精致，特别是鼓点的设计很有层次感。蔡依林的演唱技巧在这首歌中展现得很好，她的音色既清甜又有一定的成熟度。
    #         歌曲的主题关于回忆和时光倒流，很容易引起听众的共鸣。我觉得这首歌比她之前的一些作品更有深度，在情感表达上有了明显的提升。整体来说是一首很成功的流行歌曲。""",
        
    #     # Low novelty
    #     """这首《倒带》是蔡依林演唱的歌曲，音乐风格属于流行类型。歌曲的编曲比较不错，蔡依林的演唱也很专业。
    #         这首歌表达了对回忆的怀念，是一首比较经典的作品。整体听起来很舒服，是一首值得推荐的好歌。"""
    # ]
    appraisals = ["这首歌是一首器乐曲，主要特色是吉他。吉他在曲目中扮演了主导的角色，用不同效果器处理过的声音创造出独特的氛围。从柔和而梦幻的开头逐渐过渡到更加强烈和有力的部分，展示了吉他的变化和魅力。整体而言，这首歌给人的感觉非常宏大和激励人心，适合做某种系列节目的背景音乐。"]

    # Randomly select an appraisal to test different novelty levels
    return random.choice(appraisals)


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
            qa_data_path="/fs-computility/niuyazhe/lixueyan/acapella/eval_benchmark/data/option_qa.jsonl",
            song_details_path="data/song_details.jsonl",
            output_path="/fs-computility/niuyazhe/lixueyan/acapella/eval_benchmark/comprehensive_demo_results.json",
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