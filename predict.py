import torch
import torch.nn

import sys
import numpy as np
import matplotlib.pyplot as plt

from model import ImageClassifier
from utils import load_mnist

model_fn = "./model.pth"

# to run the GPU in apple_silicon M1 notebooks
device = torch.device("mps") if torch.backends.mps.is_avaiable() else "cpu"

def load(fn, device):
    d = torch.load(fn, map_location=device)

    return d['model']

def plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28, 28)
        # .detach
        # PyTorch docs에 따르면
        # "Returns a new Tensor, detached from the current graph.
        # The result will never require gradient."
        # 즉 graph에서 분리한 새로운 tensor를 리턴한다.
        # 파이토치는 tensor에서 이루어진 모든 연산을 추적해서 기록해놓는다(graph).
        # 이 연산 기록으로 부터 도함수가 계산되고 역전파가 이루어지게 된다.
        # detach()는 이 연산 기록으로 부터 분리한 tensor을 반환하는 method이다.

        #.cpu()
        # Returns a copy of this object in CPU memory.
        # GPU에 올려져 있는 tensor를 CPU로 복사하는 method

        #.numpy()
        # numpy method를 사용하고자 할 때, cpu() 뒤에 . 연결
        # cpu에 올려져 있는 tensor만 .numpy() method 사용 가능

        plt.imshow(img, cmap='gray')
        plt.show()
        print("Predict:", float(torch.argmax(y_hat, dim=1)))

def test(model, x, y, to_be_shown=True):
    model.eval() # evaluation 선언

    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=1)).sum()
        # Squeeze함수는 쉽게 얘기해서 tensor가 가지고 있는 차원 중 1인 차원을 제거하는 함수이다.
        # torch.Tensor(1, 2, 3).squeeze() -> (2, 3)
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print(f"Accuracy: {accuracy:.4f}")

        if to_be_shown:
            plot(x, y_hat)

# Load MNIST test set.
x, y = load_mnist(is_train=False)
# Reshape tensor to chunk of 1-d vectors.
x = x.view(x.size(0), -1)
# Move dataset to the GPU
x, y = x.to(device), y.to(device)

model = ImageClassifier(28**2, 10).to(device)
model.load_state_dict(load(model_fn, device))

test(model, x[:20], y[:20], to_be_shown=True)


