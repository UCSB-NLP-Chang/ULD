master_port=43144

ckpt_path=""
weight=-0.7
top_logit_filter=1e-2
EVAL_OUTDIRNAME="outputs_eval/tofu-uld"

python scripts/eval_tofu.py \
	data=harry \
	data.dataset.eval.batch_size=4 \
	model=mistral \
	model_mode=uld \
	model_mode.weight=$weight \
	ckpt_path=$ckpt_path \
	OUTDIRNAME=${EVAL_OUTDIRNAME}/${split}