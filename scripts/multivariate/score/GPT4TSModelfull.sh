python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 100}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/GPT4TSModelfull"

