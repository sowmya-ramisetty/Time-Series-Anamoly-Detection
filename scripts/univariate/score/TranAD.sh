python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "self_impl.TranAD" --model-hyper-params '{"batch_size": 128, "lr": 0.0001, "n_window": 100, "num_epochs": 5, "patience": 3}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/TranAD"

