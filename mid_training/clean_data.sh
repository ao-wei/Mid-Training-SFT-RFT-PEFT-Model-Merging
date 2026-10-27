shuf /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_2048.jsonl | sed 's/<|im_end|>//g' | sed 's/<|im_start|>//g' > /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_2048_clean_64w.jsonl
head -160000 /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_512_clean_230w.jsonl > /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_512_clean_16w.jsonl
cat /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_2048_clean_64w.jsonl /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_512_clean_16w.jsonl | shuf > /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_mid_training_80w.jsonl
rm /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_2048_clean_64w.jsonl
rm /Volumes/AoWei/大模型训练营/阶段5/代码/5.mid_sft_stage/mid_training/data/processed_dataset_512_clean_16w.jsonl
