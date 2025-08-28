# for each file in /fs-computility/niuyazhe/shared/lixueyan/acapella/ReactionEvalBenchmark/data/ground_truth, save them into song_details.jsonl, add autio_path according to the name in cut_songs

import os
import json

for file in os.listdir("/fs-computility/niuyazhe/shared/lixueyan/acapella/ReactionEvalBenchmark/data/ground_truth"):
    with open(f"/fs-computility/niuyazhe/shared/lixueyan/acapella/ReactionEvalBenchmark/data/ground_truth/{file}", "r", encoding="utf-8") as f:
        item = json.load(f)
        item["audio_path"] = f"/fs-computility/niuyazhe/shared/lixueyan/acapella/ReactionEvalBenchmark/data/cut_songs/{file.split('_ground_truth')[0]}.mp3"
        with open("data/song_details.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")