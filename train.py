from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import numpy as np
from HDC import MNISTHDC
from tqdm import tqdm
import torch as th

class MNIST(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        self.data = inputs
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train():
    le = preprocessing.LabelEncoder()
    dataset = datasets.fetch_openml("mnist_784", version=1)
    dataset.target = le.fit_transform(dataset.target)
    dataset.data = np.array(dataset.data)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, shuffle=True)
    train_loader = DataLoader(MNIST(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST(X_test, y_test), batch_size=64, shuffle=True)
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
    model = MNISTHDC(10)
    model = model.to(device)
    for i, batch in enumerate(tqdm(train_loader)):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        model.fit(data, labels)
    predictions = []
    ground_truth = []
    for i, batch in enumerate(tqdm(test_loader)):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        preds = model.forward(data)
        ground_truth = ground_truth + list(labels.cpu().numpy())
        predictions =  predictions + list(preds.cpu().numpy())
    print("Acuracy: {}".format(accuracy_score(np.array(ground_truth), np.array(predictions))))

if __name__ == "__main__":
    train()

