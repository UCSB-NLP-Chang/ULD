master_port=43144

splits=(
    "forget01"
    "forget05"
    "forget10"
)
    
unlearn_losses=(
    ga
    ga+gd
    ga+kl

    npo
    npo+gd
    npo+kl

    dpo
    dpo+gd
    dpo+kl
)


data_modes=(
    forget
    forget_retain
    forget_retain

    forget
    forget_retain
    forget_retain

    dpo
    dpo_retain
    dpo_retain
)

OUTPUTMODELDIR=outputs_trained_models/tofu-base/
EVAL_OUTDIRNAME="outputs_eval/tofu-base"

lr=1e-5
# ! Non lora
COMMON="data=tofu trainer.batch_size=4 trainer.learning_rate=$lr trainer.gradient_accumulation_steps=4 model=tofu-llama-2 model_mode=base trainer.strategy=deepspeed OUTPUTMODELDIR=$OUTPUTMODELDIR"

export CUDA_VISIBLE_DEVICES=1,0
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

for split in ${splits[@]}; do
    for i in "${!unlearn_losses[@]}"; do
        unlearn_loss=${unlearn_losses[$i]}
        data_mode=${data_modes[$i]}

        CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=$((master_port + i*100)) \
            scripts/hf_forget_train.py \
            data.dataset.split=${split}_perturbed \
            project="debug" \
            unlearn_loss=$unlearn_loss \
            data_mode=$data_mode \
            $COMMON

		unset CUDA_VISIBLE_DEVICES
    done
done