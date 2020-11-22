. ~/anaconda3/etc/profile.d/conda.sh
conda env create -f pong.yml --prefix=/media/Data/env/pong-env
conda activate pong-env
pip install --upgrade pip setuptools==44.1.0
sudo apt install libcurl4-openssl-dev libssl-dev
pip install -r requirement.txt



