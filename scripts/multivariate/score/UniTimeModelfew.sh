python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.UniTimeModel" --model-hyper-params '{"batch_size": 32, "dataset": "UV", "horizon": 1, "max_backcast_len": 96, "max_token_num": 17, "norm": true, "sampling_rate": 0.05, "seq_len": 96}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/UniTimeModelfew"

