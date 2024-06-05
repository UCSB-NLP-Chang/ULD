master_port=43144

splits=(
    # "forget01"
    # "forget05"
    "forget10"
)
    
data_modes=(
    "forget_more_retain_perturb"
)

OUTPUTMODELDIR=outputs_trained_models/tofu/
EVAL_OUTDIRNAME="outputs_eval/tofu-uld"

# ! lora
lr=1e-3
COMMON="data=tofu trainer.batch_size=8 trainer.learning_rate=$lr trainer.gradient_accumulation_steps=2 model=tofu-llama-2 model_mode=uld model_mode.num_layer=8 trainer.strategy=dpo unlearn_loss=remember+uniform OUTPUTMODELDIR=$OUTPUTMODELDIR"

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

for split in ${splits[@]}; do
    for i in "${!data_modes[@]}"; do
        data_mode=${data_modes[$i]}

        CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=$((master_port + i*100)) \
			scripts/hf_forget_train.py \
            project="debug" \
            data.dataset.split=${split}_perturbed \
            $COMMON

		# export CUDA_VISIBLE_DEVICES=6,7
		unset CUDA_VISIBLE_DEVICES
        rawcheckpoints=($(find $OUTPUTMODELDIR -type d | grep $model_mode | grep "checkpoint-[0-9]*"))
		checkpoints=($(printf "%s\n" "${rawcheckpoints[@]}" | awk -F'-' '{print $NF, $0}' | sort -rn | cut -d' ' -f2-))
		
		gpuid=0
		for ((i=0; i<${#checkpoints[@]}; i+=1)); do
			checkpoint="${checkpoints[$i]}"
			gpuid=$((6+ (gpuid+1) % $num_devices))
			checkpoint="${checkpoint//=/\\=}"

			weights=(-0.8)
			CUDA_VISIBLE_DEVICES=$gpuid python scripts/eval_tofu.py \
				data=tofu \
				data.dataset.split=${split}_perturbed \
				data.dataset.eval.retain_result="data/${split}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
				data.dataset.eval.batch_size=4 \
				model=tofu-llama-2 \
				model_mode=uld \
				model_mode.weight=-0.8 \
				model_mode.top_logit_filter=1e-2 \
				ckpt_path=$checkpoint \
				OUTDIRNAME=${EVAL_OUTDIRNAME}/${split} &
			if (((i+1) % $num_devices == 0)); then
				wait
			fi
		done
		wait

        sleep 2
		unset CUDA_VISIBLE_DEVICES
		# rm -rf $OUTPUTMODELDIR/*
    done
done