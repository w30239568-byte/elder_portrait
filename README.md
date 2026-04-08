# 老年画像模型（StoryWell 扩展标签版）

默认训练数据：`dataset/storywell_raw_split_new.csv`  
`event_bio` 支持带空格标签（如 `B-Education background_pro`），并以 `\t` 分隔每个字符标签。

## 统一标签集（event_bio）
- `O`
- `B/I-protagonist`
- `B/I-participant_par`
- `B/I-location_pro`
- `B/I-location_par`
- `B/I-Health_pro`
- `B/I-Health_par`
- `B/I-Identity_pro`
- `B/I-Identity_par`
- `B/I-Achievement_pro`
- `B/I-Achievement_par`
- `B/I-Interest_pro`
- `B/I-Interest_par`
- `B/I-Occupation_pro`
- `B/I-Occupation_par`
- `B/I-Education background_pro`
- `B/I-Education background_par`
- `B/I-Social Activity_pro`

## 训练
```bash
python train_elder_portrait.py --data_path dataset/storywell_raw_split_new.csv
```

默认输出目录：`runs/elder_portrait`
- `best_model.pt`
- `label_mapping.json`
- `history.json`
- `test_report.json`
- `tokenizer/`

## 推理（供后端调用）
```bash
python run_model_api.py \
  --checkpoint runs/elder_portrait/best_model.pt \
  --mapping_path runs/elder_portrait/label_mapping.json \
  --input_json contracts/analyze_v1.request.example.json \
  --output_json runs/analyze_result.json
```

## 说明
- 模型联合学习两项任务：`event_bio` 抽取 + `sentiment` 分类。
- 推理默认先走模型；低置信度/空抽取时才规则兜底。
- 新旧标签会通过 `elder_portrait/tag_schema.py` 做兼容映射。
