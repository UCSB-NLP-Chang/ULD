master_port=43144

ckpt_path=""
split="forget10"
weight=-0.8
top_logit_filter=1e-2
EVAL_OUTDIRNAME="outputs_eval/tofu-uld"

python scripts/eval_tofu.py \
	data=tofu \
	data.dataset.split=${split}_perturbed \
	data.dataset.eval.retain_result="data/${split}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
	data.dataset.eval.batch_size=4 \
	model=tofu-llama-2 \
	model_mode=uld \
	model_mode.weight=$weight \
	ckpt_path=$ckpt_path \
	OUTDIRNAME=${EVAL_OUTDIRNAME}/${split}