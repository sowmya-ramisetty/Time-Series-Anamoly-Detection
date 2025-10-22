python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 1, "max_backcast_len": 512, "max_token_num": 80, "norm": true, "sampling_rate": 1, "seq_len": 512, "stride": 16}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/UniTimeModelfull"

