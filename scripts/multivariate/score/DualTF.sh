python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.DualTF" --model-hyper-params '{"batch_size": 8}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/DualTF"

