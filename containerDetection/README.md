
python -m venv venv

venv\Scripts\activate

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

python -m pip install -r requirements.txt



mlflow ui

tensorboard --logdir=runs