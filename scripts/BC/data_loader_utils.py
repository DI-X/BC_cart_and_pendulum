from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import h5py
import numpy as np
import torch

import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataloader(data_path:str, config: str = None, num_workers:int = 0, device = 'cpu') -> DataLoader:
    if config is None:
        batch_size = 64
    else:
        with open(config) as stream:
            config_data = yaml.safe_load(stream)
            batch_size = config_data["batch_size"]

    processed_data = preprocess_demo_data(data_path)
    dataset = DemoDataset(processed_data, device)
    data_loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    return data_loader

def preprocess_demo_data(data_path: str) -> dict:
    with h5py.File(data_path, "r") as f:
        processed_data = {'actions':[], 'observations': []}

        for name in f:
            if name.startswith('demo'):
                group = f[name]
                if 'action' in group and 'observation' in group:
                    action = group['action'][:]
                    obs = group['observation'][:]
                    processed_data['actions'].extend(action)
                    processed_data['observations'].extend(obs)
                else:
                    logger.warning(f"{name} is missing observation or action")

        data_array = {'actions' : np.array(processed_data['actions']),
                  'observations' : np.array(processed_data['observations'])}

    return data_array

class DemoDataset(Dataset):
    def __init__(self, demo_data, device):
        self.target_actions = torch.tensor(demo_data["actions"], dtype=torch.float).to(device)
        self.obs = torch.tensor(demo_data["observations"], dtype=torch.float).to(device)

    def __len__(self):
        return len(self.target_actions)

    def __getitem__(self, idx):
        return self.obs[idx], self.target_actions[idx]

