python eval.py --config /path/to/logs/task/timestamp/config.json \
 --ckpt /path/to/logs/task/timestamp/ckpt/policy_last.ckpt \
 --num_action 20 --num_inference_step 20 --max_steps 1000 --seed 233 \
 --discretize_rotation --ensemble_mode act --vis \
 --record --record_path /path/to/record/path \
 --enable_exploration