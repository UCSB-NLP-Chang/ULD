master_port=43124

model_name=mistral
model_path="mistralai/Mistral-7B-Instruct-v0.2"

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

OUTPUTMODELDIR=outputs_trained_models/harry-base/
EVAL_OUTDIRNAME="outputs_eval/harry-base"
COMMON="data=harry trainer.batch_size=4 trainer.learning_rate=$lr trainer.gradient_accumulation_steps=8 model=mistral model_mode=offset trainer.strategy=deepspeed OUTPUTMODELDIR=$OUTPUTMODELDIR"

lr=1e-5
for i in "${!unlearn_losses[@]}"; do
    unlearn_loss=${unlearn_losses[$i]}
    data_mode=${data_modes[$i]}

    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=$((master_port + i*100)) \
        scripts/hf_forget_train.py \
        project="debug" \
        unlearn_loss=$unlearn_loss \
        data_mode=$data_mode \
        $COMMON

done