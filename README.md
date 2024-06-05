# Reversing the Forget-Retain Objectives: An Efficient LLM Unlearning Framework from Logit Difference
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2406.08607-B21A1B)](https://arxiv.org/abs/2406.08607)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![GitHub Stars](https://img.shields.io/github/stars/UCSB-NLP-Chang/ULD?style=social)](https://github.com/UCSB-NLP-Chang/ULD/stargazers)

This is the implementation for the paper [*Reversing the Forget-Retain Objectives: An Efficient LLM Unlearning Framework from Logit Difference*](https://arxiv.org/abs/2406.08607).

## Install
```bash
conda create -n uld python=3.10 -y
conda activate uld
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install flash-attn==2.5.6 --no-build-isolation
pip install deepspeed
pip install -e .
```

## Training

### ULD Training
```bash
python scripts/hf_forget_train.py \
    data=[tofu|harry] \
    data.dataset.split=${DATASPLIT} \
    model=[tofu-llama-2|mistral] \
    model_mode=uld \
    model_mode.num_layer=8 \
    unlearn_loss=remember+uniform \
    trainer.strategy=ddp \
    OUTPUTMODELDIR=${OUTPUTMODELDIR}
```
For more detailed training options, please refer to `bashes/tofu/uld_train_eval.sh` and `bashes/harry/uld_train_eval.sh`. This would save the assistant model to `${OUTPUTMODELDIR}`.

### Offset Training
```bash
python scripts/hf_forget_train.py \
    data=[tofu|harry] \
    data.dataset.split=${DATASPLIT} \
    model=[tofu-llama-2|mistral] \
    model_mode=offset \
    unlearn_loss=${UNLEARN_LOSS} \
    trainer.strategy=deepspeed \
    OUTPUTMODELDIR=${OUTPUTMODELDIR}
```
For more detailed training options, please refer to `bashes/tofu/offset_train_eval.sh` and `bashes/harry/offset_train_eval.sh`. This would save the assistant model to `${OUTPUTMODELDIR}`.


### Other Training
```bash
python scripts/hf_forget_train.py \
    data=[tofu|harry] \
    data.dataset.split=${DATASPLIT} \
    model=[tofu-llama-2|mistral] \
    model_mode=base \
    unlearn_loss=${UNLEARN_LOSS} \
    trainer.strategy=deepspeed \
    OUTPUTMODELDIR=${OUTPUTMODELDIR}
```
For more detailed training options, please refer to `bashes/tofu/base_train_eval.sh` and `bashes/harry/base_train_eval.sh`. This would save the unlearned model to `${OUTPUTMODELDIR}`.

## Evaluation

```bash
python scripts/eval_tofu.py \
    data=[tofu|harry] \
    model=[tofu-llama-2|mistral] \
    model_mode=[base|uld|offset] \
    ckpt_path=${CHECKPOINT_DIR} \
    data.dataset.split=${DATASPLIT} 
```
For more detailed options, please refer to `bashes/tofu/tofu_eval.sh` and `bashes/harry/harry_eval.sh`.


## Development

We also implement several other unlearning methods employed in previous works, including:
* [Offset Unlearning for Large Language Models](https://arxiv.org/abs/2404.11045)
* [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)
* [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)

You can follow the guide below to implement other unlearning methods.

### Code Structure
* `scripts/`: scripts for training and evaluation
* `uld/data/`: data processing and dataloader
* `uld/models/`: model definition
* `uld/trainer/`: unlearn trainer and unlearn losses. 

### Add other dataset
* Add dataset to `uld/data/` and register it in `uld/data/__init__.py`
* Implement new dataset class by inheriting `TrainDataModule` class, reference implementation for ToFU dataset is in `uld/data/tofu.py`. Typically, you need to implement the logic to load *forget data* and *retain data*. 

### Add other unlearn loss
* Add unlearn loss to `uld/trainer/unlearn_losses.py` and add it in `configs/unlearn_loss`.
* Implement new unlearn loss class by defining the `forget_loss_func` and `retain_loss_func` for `ForgetRetainLoss` class, reference implementation is in `create_unlearn_loss` function.



## Citation
If you find this work useful, please consider cite our paper:
```bibtex
@article{ji2024reversing,
  title   = {Reversing the Forget-Retain Objectives: An Efficient LLM Unlearning Framework from Logit Difference},
  author  = {Jiabao Ji and Yujian Liu and Yang Zhang and Gaowen Liu and Ramana Rao Kompella and Sijia Liu and Shiyu Chang},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2406.08607}
}
```

## Acknowledgement
Huge thanks for following repos that greatly help our implementation:
* https://github.com/licong-lin/negative-preference-optimization
* https://github.com/OPTML-Group/SOUL
* https://github.com/locuslab/tofu
* https://github.com/EleutherAI/lm-evaluation-harness
* https://github.com/voidism/dola
