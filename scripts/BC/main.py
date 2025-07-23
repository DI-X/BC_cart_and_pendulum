import torch.accelerator

from data_loader_utils import get_dataloader
from bc import BC
import os

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
print(f"device: {device}")

data_path = os.path.join("data", "demo", "demo_test.hdf5")
config_path = os.path.join("config", "bc_agent.yaml")
save_dir = os.path.join("logs", "BC")

train_data = get_dataloader(data_path, config = config_path, device=device)

model = BC(config_path, device=device)
# model = BC(device=device)
model.evaluate(train_data)
model.train_loop(train_data, save_dir)
model.evaluate(train_data)

##########3########
# test trained model
####################

# model_test = BC()
# model_test.to(device)
# check_point = torch.load('model.pth', weights_only=True)
# model_test.load_state_dict(check_point['model_state_dict'])
# model_test.test(train_data)