python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 128, "e_layers": 2, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 100}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/iTransformer"

