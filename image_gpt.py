# Here is where the transformer will go
import torch
import torch.nn as nn

class ImageClassifier(nn.module):
    def __init__(self, vocab_size):
        super().__init__()