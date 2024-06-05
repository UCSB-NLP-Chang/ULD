conda create -n uld python=3.10 -y
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install flash-attn==2.5.6 --no-build-isolation
pip install deepspeed
pip install -r requirements.txt
pip install -e .