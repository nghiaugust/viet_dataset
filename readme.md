python -m venv .venv
.\.venv312\Scripts\Activate.ps1

python -m pip install "pip<24.1" 
pip install tqdm
pip install trdg
pip install "setuptools<70.0.0"
pip install "Pillow<10.0.0"

python code/generate_dataset.py


bản v2 bổ sung về đệm giữa các biên