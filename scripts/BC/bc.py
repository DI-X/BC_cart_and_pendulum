import os
from datetime import datetime
import yaml
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

OPTIMIZERS ={
    'adam': torch.optim.Adam,
    'sdg': torch.optim.SGD,
}

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

class BC(nn.Module):
    def __init__(self, config=None, device='cpu'):
        super().__init__()

        if config is None:
            print(f" config: default value")
            self.input_shape = 4
            self.output_shape = 1

            self.first_hidden_units = 32
            self.second_hidden_units = 32
            self.act_fn_name = 'relu'

            self.lr = 1e-4

            self.l2_loss_weight = 1
            self.l1_loss_weight = 1

            self.n_epochs = 200
            self.batch_size = 64
            self.optimizer_name = 'adam'

            self.n_check_point_to_save = 1
            self.device = device
        else:
            print(f" config file: {config}")
            self.load_yaml(config)

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()
        self.activation_fn = ACTIVATIONS[self.act_fn_name]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_shape, self.first_hidden_units),
            self.activation_fn,
            nn.Linear(self.first_hidden_units, self.second_hidden_units),
            self.activation_fn,
            nn.Linear(self.second_hidden_units, self.output_shape)
        )
        self.optimizer = OPTIMIZERS[self.optimizer_name](self.parameters(), lr=self.lr)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)

    def predict(self, obs:torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(obs)

    def compute_losses(self, predictions:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        losses =  OrderedDict()
        losses["l2_loss"] = self.l2_loss(predictions, target)
        losses["l1_loss"] = self.l1_loss(predictions, target)

        action_loss = (self.l2_loss_weight * losses["l2_loss"] +
                       self.l1_loss_weight * losses["l1_loss"])

        return action_loss


    def train_loop(self, dataloader:DataLoader, save_dir:str) -> None:
        date_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        save_dir = os.path.join(save_dir, date_time)
        self.train()
        data_size = len(dataloader.dataset)

        os.makedirs(os.path.dirname(os.path.join(save_dir, 'dummy.jpg')), exist_ok=True)
        check_point_time_step = int((self.n_epochs * data_size) / self.n_check_point_to_save)

        for t in range(self.n_epochs):
            for batch, (obs, target) in enumerate(dataloader):
                pred = self.predict(obs)
                loss = self.compute_losses(pred, target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (t+1) % 10 == 0 and batch == 0:
                    loss, current = loss.detach().item(), batch * self.batch_size
                    print(f" loss: {loss:>7f} epoch:{(t+1):>5d} / {self.n_epochs}")

                if ((t+1) * data_size + batch * self.batch_size) % check_point_time_step == 0:
                    file_name = os.path.join(save_dir, f"checkpoint_{(t+1) * self.batch_size + batch}.pth")
                    self.save_check_points(file_name)

        file_name = os.path.join(save_dir, "model.pth")
        self.save_check_points(file_name)
        self.save_yaml(save_dir)


    def save_check_points(self, file_name: str) -> None:
        torch.save({
            'epoch' : self.n_epochs,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },
        file_name)

        print(f"checkpoint saved to {file_name}")


    def evaluate(self, dataloader:DataLoader) -> None:
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
            print(f"test average loss: {loss:>7f}, total l1 loss: {l1_loss:>7f}, action l1 loss: {100 * l1_loss/size:>6.3f} N/m")

    def load_yaml(self, config: str) -> None:
        with open(config) as stream:
            config_data = yaml.safe_load(stream)

            self.batch_size = config_data['batch_size']
            self.n_epochs = config_data['n_epochs']
            self.lr = config_data['learning_rate']
            self.n_check_point_to_save = config_data['num_check_point_to_save']
            self.device = config_data['device']

            if 'act_fn_name' in config_data['policy_kwargs']:
                self.act_fn_name = config_data['policy_kwargs']['act_fn_name']
            else:
                raise ValueError( f"activation_fn is missing from config file: {config} [policy_kwargs]" )

            if len(config_data['policy_kwargs']['net_arch']) == 2:
                self.first_hidden_units = config_data['policy_kwargs']['net_arch'][0]
                self.second_hidden_units = config_data['policy_kwargs']['net_arch'][1]
            else:
                raise ValueError(f"length of net_arch mismatch, expected 2 but got {config_data['policy_kwargs']['net_arch']}")

            if 'input_shape' in config_data['policy_kwargs']:
                self.input_shape = config_data['policy_kwargs']['input_shape']
            else:
                raise ValueError(f"input_shape is missing from config file: {config} [policy_kwargs]")

            if 'output_shape' in config_data['policy_kwargs']:
                self.output_shape = config_data['policy_kwargs']['output_shape']
            else:
                raise ValueError(f"output_shape is missing from config file: {config} [policy_kwargs]")

            if 'l1_loss_weight' in config_data['reward_kwargs']:
                self.l1_loss_weight = config_data['reward_kwargs']['l1_loss_weight']
            else:
                raise ValueError(f"l1_loss_weight is missing from config file: {config} [reward_kwargs]" )
            if 'l2_loss_weight' in config_data['reward_kwargs']:
                self.l2_loss_weight = config_data['reward_kwargs']['l2_loss_weight']
            else:
                raise ValueError(f"l2_loss_weight is missing from config file: {config} [reward_kwargs]")

            self.optimizer_name = config_data['optimizer']

    def save_yaml(self, save_dir: str) -> None:
        params = {
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'learning_rate': self.lr,
            'policy_kwargs': {
                'act_fn_name': self.act_fn_name,
                'net_arch': [
                    self.first_hidden_units,
                    self.second_hidden_units,
                ],
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            },
            'reward_kwargs': {
                'l1_loss_weight': self.l1_loss_weight,
                'l2_loss_weight': self.l2_loss_weight,
            },
            'optimizer':self.optimizer_name,
            'num_check_point_to_save': self.n_check_point_to_save,
            'device': f"{self.device}",
        }

        config = os.path.join(save_dir, 'config.yaml')
        with open(config, 'w') as f:
            yaml.dump(params, f,  default_flow_style=False)




