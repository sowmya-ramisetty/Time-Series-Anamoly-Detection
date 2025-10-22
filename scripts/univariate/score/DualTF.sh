python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "self_impl.DualTF" --model-hyper-params '{"batch_size": 8, "lr": 1e-05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/DualTF"

