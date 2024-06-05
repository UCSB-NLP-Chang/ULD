master_port=43144
model_name=mistral
model_path="mistralai/Mistral-7B-Instruct-v0.2"

unlearn_losses=(
    "remember_uniform"
)


data_modes=(
    "forget_more_retain"
)

COMMON="lightning.trainer.devices=2 data.batch_size=2 gradient_accumulation_steps=8 model_train.weight_decay=0.01 OUTPUTMODELDIR=trained_models2/hf-baseline-copyright lightning.trainer.strategy=deepspeed_stage_3 BASELOGDIR=hf-outputs_lightning_tune-copyright"

lr=1e-5
for i in "${!model_modes[@]}"; do
    model_mode=${model_modes[$i]}
    data_mode=${data_modes[$i]}

    CUDA_VISIBLE_DEVICES=1,0 torchrun --nproc_per_node=4 --master_port=$((master_port + i*100)) \
    	scripts/hf_forget_train.py \
        data=harry \
        lr=$lr \
        project="harry-ours-hf" \
        model_train=$model_mode \
        model_train.model_name=$model_name \
        model_train.model_path=$model_path \
        model_train.learning_rate=$lr \
        data_mode=$data_mode \
        $COMMON

	export CUDA_VISIBLE_DEVICES=1,0
    num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

    rawcheckpoints=($(find $OUTPUTMODELDIR -type d | grep $model_mode | grep "checkpoint-[0-9]*"))
	checkpoints=($(printf "%s\n" "${rawcheckpoints[@]}" | awk -F'-' '{print $NF, $0}' | sort -rn | cut -d' ' -f2-))

	for ((i=0; i<${#checkpoints[@]}; i+=1)); do
		checkpoint="${checkpoints[$i]}"
		gpuid=$(((gpuid+1) % 4))
		checkpoint="${checkpoint//=/\\=}"
        weight=(-0.5)

    	python scripts/eval_copyright.py \
            BASEDIR=${BASEDIR} \
			data=${data} \
			model=${model} \
			model.model_path=${model_path} \
			remember=${remember} \
			remember.num_layer=${num_layer} \
			remember.weight=${weight} \
			remember.save_path="${checkpoint}" \
			remember.is_lora=True \
			OUTDIRNAME=$OUTDIRNAME \
			gpu=gpu$((gpuid)) \
			remember.top_logit_filter=1e-2  &

		if (((i+1) % $num_devices == 0)); then
            wait
            break
		fi
	done
	wait

    sleep 2
	unset CUDA_VISIBLE_DEVICES
	rm -rf $OUTPUTMODELDIR/*
done