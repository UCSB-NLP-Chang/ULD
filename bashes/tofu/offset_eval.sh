master_port=43144

ckpt_path=""
split="forget10"
EVAL_OUTDIRNAME="outputs_eval/tofu-offset"

python scripts/eval_tofu.py \
	data=tofu \
	data.dataset.split=${split}_perturbed \
	data.dataset.eval.retain_result="data/${split}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
	data.dataset.eval.batch_size=4 \
	model=tofu-llama-2 \
	model_mode=offset \
	ckpt_path=$ckpt_path \
	OUTDIRNAME=${EVAL_OUTDIRNAME}/${split}