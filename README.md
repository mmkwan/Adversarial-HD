# cs295-adversarialHD

## Setup
To install the dependencies, run `pip install -r requirements.txt`

## Training a model
To train a PyTorch model using the HDC code, run `python train.py --save_path model.pkl --quantize_levels 256`

## Attacking it
To generate an adversarial dataset against the model, run `python GA.py --model_path model.pkl --save_path adversarial_data.npz --samples 200`

## Testing it
To test a model, load a model in, and run `model(input)`.
```python
    from train import MNIST
    from torch.utils.data import DataLoader

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    
    adversarial_data = numpy.load("adversarial_data.npz")
    adv_loader = DataLoader(MNIST(adversarial_data["X"], adversarial_data["y"]), batch_size=32, shuffle=True)
    for batch in adv_loader:
        data, labels = batch
        output = model(data)
        predictions = output["predictions"]
        accuracy = (labels == predictions).float().mean()
```