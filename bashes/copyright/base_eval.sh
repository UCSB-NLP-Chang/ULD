master_port=43144

ckpt_path=""
EVAL_OUTDIRNAME="outputs_eval/harry-base"

python scripts/eval_tofu.py \
	data=harry \
	data.dataset.eval.batch_size=4 \
	model=mistral \
	model_mode=base \
	ckpt_path=$ckpt_path \
	OUTDIRNAME=${EVAL_OUTDIRNAME}/${split}