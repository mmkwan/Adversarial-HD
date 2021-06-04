from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import numpy as np
from HDC import MNISTHDC
from tqdm import tqdm
import torch as th
import pickle as pkl
from argparse import ArgumentParser

DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
th.manual_seed(5) #To fix the training outcome.

class MNIST(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        self.data = inputs
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(save_path: str, value_precision: int):
    le = preprocessing.LabelEncoder()
    dataset = datasets.fetch_openml("mnist_784", version=1)
    dataset.target = le.fit_transform(dataset.target)
    dataset.data = np.array(dataset.data)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, shuffle=True)
    train_loader = DataLoader(MNIST(X_train, y_train), batch_size=32, shuffle=True)
    model = MNISTHDC(10, value_quantize_precision=value_precision)
    model = model.to(DEVICE)
    for i, batch in enumerate(tqdm(train_loader)):
        data, labels = batch
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        model.fit(data, labels)
    model = model.to("cpu")
    with open(save_path, "wb") as f:
        pkl.dump(model, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="The path to which to save the trained model as a pickle")
    parser.add_argument("--quantize_levels", type=int, required=False, default = 256, help="The number of levels to which to quantize the color intensities to")
    args = parser.parse_args()
    train(args.save_path, args.quantize_levels)

