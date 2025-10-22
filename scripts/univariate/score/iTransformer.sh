python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 16, "d_ff": 128, "d_model": 64, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 3, "seq_len": 100}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/iTransformer"

