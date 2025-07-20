from collections import OrderedDict
import torch
from torch import nn


class BC(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = 4
        self.output_shape = 1
        self.first_hidden_units = 34
        self.second_hidden_units = 34

        self.lr = 1e-4

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()

        self.l2_loss_weight = 1
        self.l1_loss_weight = 1

        self.epoch = 200

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_shape, self.first_hidden_units),
            nn.ReLU(),
            nn.Linear(self.first_hidden_units, self.second_hidden_units),
            nn.ReLU(),
            nn.Linear(self.second_hidden_units, self.output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)

    def predict(self, obs) -> torch.Tensor:
        return self.linear_relu_stack(obs)

    def compute_losses(self, predictions, target):
        losses =  OrderedDict()

        losses["l2_loss"] = self.l2_loss(predictions, target)
        losses["l1_loss"] = self.l1_loss(predictions, target)

        action_loss = (self.l2_loss_weight * losses["l2_loss"] +
                       self.l1_loss_weight * losses["l1_loss"])

        return action_loss


    def train_loop(self, dataloader):
        self.train()
        size = len(dataloader.dataset)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for t in range(self.epoch):
            # print(f"\n epoch :{t}  {'-'*20}")
            for batch, (obs, target) in enumerate(dataloader):

                pred = self.predict(obs)
                loss = self.compute_losses(pred, target)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if t % 10 == 0 and batch == 0:
                    loss, current = loss.item(), batch * 64
                    print(f" loss: {loss:>7f} epoch:{t:>5d} / {self.epoch}")

    def save_check_points(self, file_name):
        torch.save({
            'epoch' : self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },
        file_name)

    def save_model(self, save_name):
        torch.save(self, save_name)
        print(f"model is saved as {save_name}")

    def test(self, dataloader):
        self.eval()

        size = len(dataloader.dataset)
        loss = 0
        l1_loss = 0

        with torch.no_grad():
            for obs, target in dataloader:
                pred = self.predict(obs)
                loss += self.compute_losses(pred, target)
                l1_loss += (pred - target).abs().sum().item()

            loss /= size

            # the output action will be scaled by 100 in issac sim
            print(f"test loss: {loss:>7f}, total l1 loss: {l1_loss}, action l1 loss: {100 * l1_loss/size}")
