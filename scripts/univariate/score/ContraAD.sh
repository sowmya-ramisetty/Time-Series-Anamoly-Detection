python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "self_impl.ContraAD" --model-hyper-params '{"n_window": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/ContraAD"

