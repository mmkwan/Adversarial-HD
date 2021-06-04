# cs295-adversarialHD

## Setup
To install the dependencies, run `pip install -r requirements.txt`

## Training a model
To train a PyTorch model using the HDC code, run `python train.py --save_path model.pkl --quantize_levels 256`

## Attacking it
To generate an adversarial dataset against the model, run `python GA.py --model_path model.pkl --save_path adversarial_data.npz --samples 200`