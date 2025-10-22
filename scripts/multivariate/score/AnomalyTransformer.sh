python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.AnomalyTransformer" --model-hyper-params '{"batch_size": 64, "lr": 0.001, "num_epochs": 3, "win_size": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/AnomalyTransformer"

