python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install tqdm

pip install "setuptools<70.0.0"
pip install "Pillow<10.0.0"

python code/generate_dataset.py