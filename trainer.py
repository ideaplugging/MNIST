from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit # cross-entropy loss

        super().__init__()

    def _train(self, x, y, config):
        self.model.train() # 학습의 train / eval 모드를 명시 해야 함!!

        # Shuffle before begin.
        indices = torch.randperm(x.size(0), device=x.device) # which order should we shuffle. random permutation (수열)
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0) # rearrange by index and split the tensor into the batch size
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0) # rearrange by index and split the tensor into the batch size

        # x와 y를 같이 shuffling 하는 것이 중요 포인트

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze()) # the form we expect is the same as (bs, ). In case (bs, 1) comes in, we use squeeze method on y_i.

            # Initialize the gradients of the model.
            self.optimizer.zero_grad() # model weight parameters에 gradient가 혹시 저장되어 있을지 모르니, 미리 0으로 초기화
            loss_i.backward() # loss에 대해 gradient가 back propagate

            self.optimizer.step()

            if config.verbose >= 2: # 진행 현황 print
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i) # float 안 씌우면 tensor type. 모든 computation의 그래프가 물려 있어서, 메모리 leak 발생되니, float를 씌워야 함

        return total_loss / len(x) # 길이 만큼 나눠서 리턴

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval() # 꼭 바꿔줘야 함

        # Turn on the no_grad mode to make more efficiently.
        with torch.no_grad(): # 빠르고 메모리 덜먹게
            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i) # total loss를 구하기만 하면 됨

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs): # epoch 만큼 for 문을 돎
            train_loss = self._train(train_data[0], train_data[1], config) # (data, label, config), average loss
            valid_loss = self._validate(valid_data[0], valid_data[1], config) # (data, label, config), average loss

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss: # infinite loss value 미리 선언
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)
