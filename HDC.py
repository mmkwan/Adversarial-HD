import torch as th
from typing import Optional, Tuple, List
import numpy as np


class MNISTHDC:
    def __init__(self, classes: int, dim: Optional[int] = 10000, num_channels: Optional[int] =784, value_range: Optional[Tuple[int]] = (0,255), value_quantize_precision: Optional[int] = 256):
        assert value_quantize_precision >= 2, "Must quantize to at least 2 values"
        self.dim = dim
        self.binary_model = th.randint(0, 2, size=(classes, dim))
        self.model= th.randint(0, 2, size=(classes, dim))
        self.seen_counts = th.zeros(classes, 1, dtype=th.int)
        self.channel_vectors = th.randint(0, 2, size=(num_channels, dim))
        self.value_vectors = th.zeros(size=(value_quantize_precision, dim), dtype=th.int)
        self.bins = th.tensor(np.linspace(value_range[0], value_range[1], num=value_quantize_precision))
        self._setup_encoding(value_range, value_quantize_precision)

    def _setup_encoding(self, value_range: Tuple[int], value_quantize_precision: int):
        range_vectors = th.randint(0, 2, size=(2, self.dim))
        index_util = th.ones(self.dim)
        for i in range(value_quantize_precision):
            bin = self.bins[i]
            indices = index_util.multinomial(self.dim, replacement=False)
            n = int(self.dim* ((bin - value_range[0]) / (value_range[1] - value_range[0])))
            vector = th.zeros(self.dim, dtype=th.int)
            vector[:n] = range_vectors[1, indices[:n]]
            vector[n:] = range_vectors[0, indices[n:]]
            self.value_vectors[i] = vector

    def _encode(self, input: th.Tensor) -> th.Tensor:
        assert input.shape[1] == self.channel_vectors.shape[0], "Unexpected input channels"
        encoded = self.value_vectors[th.bucketize(input, self.bins)]
        encoded = th.bitwise_xor(encoded, self.channel_vectors)  #Binding
        encoded = th.sum(encoded, dim=1) - self.channel_vectors.shape[0]//2  #Bundling
        return th.clip(encoded, 0, 1) #Majority rule

    def forward(self, input: th.Tensor) -> th.Tensor:
        encoded = self._encode(input)
        scores = th.cdist(encoded.float(), self.binary_model.float(), p=0) #Hamming distance
        return scores.argmin(dim=1)
    
    def fit(self, input: th.Tensor, labels: th.Tensor) -> None:
        encoded = self._encode(input)
        self.model.index_add_(0, labels, encoded) #Accumulation
        self.seen_counts.index_add_(0, labels, th.ones(labels.shape[0], 1, device=self.seen_counts.device, dtype=th.int))
        self._normalize()

    def _normalize(self) -> None:
        self.binary_model = th.clip(self.model-self.seen_counts//2, 0, 1) #Majority rule

    def to(self, *args):
        self.binary_model = self.binary_model.to(*args)
        self.model = self.model.to(*args)
        self.seen_counts = self.seen_counts.to(*args)
        self.channel_vectors = self.channel_vectors.to(*args)
        self.value_vectors = self.value_vectors.to(*args)
        self.bins = self.bins.to(*args)
        return self






