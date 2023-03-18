import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from utils import load_mnist

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--mps', type=int, default=0 if torch.backends.mps.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8) # train set / vaildation set

    p.add_argument('--batch_size', type=int, default=64) # mini-batch size
    p.add_argument('--n_epochs', type=int, default=20) # epoch 수
    p.add_argument('--verbose', type=int, default=2) # 1은 epoch이 끝날 때마다

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.mps < 0 else torch.device("mps")

    x, y = load_mnist(is_train=True)
    # Reshape tensor to chunk of 1-d vectors.
    x = x.view(x.size(0), -1) # (|data_set_size|, 28, 28) - > (|data_set_size|, 784)

    train_cnt = int(x.size(0) * config.train_ratio) # train set과 validation set을 미리 정의
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0) # |x| = (train_cnt, 784)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0) # |y| = (valid_cnt, 784)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    model = ImageClassifier(28**2, 10).to(device) # 784 -> 10
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss() # cross-entropy

    trainer = Trainer(model, optimizer, crit)

    trainer.train((x[0], y[0]), (x[1], y[1]), config) # (train_data, valid_data, config)

    # Save best model weights. pickle이랑 크게 다르지 않음. config까지 저장해야 되기에, dict 자료형을 사용하면 펀함
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
