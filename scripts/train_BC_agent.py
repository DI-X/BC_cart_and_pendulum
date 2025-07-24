import torch.accelerator

from BC.data_loader_utils import get_dataloader
from BC.bc import BC
import os

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
print(f"device: {device}")

data_path_test = os.path.join("../data", "demo", "demo_test.hdf5")
data_path_train = os.path.join("../data", "demo", "demo.hdf5")
config_path = os.path.join("../config", "bc_agent.yaml")
save_dir = os.path.join("../logs", "BC")

train_data = get_dataloader(data_path_train, config = config_path, device=device)
test_data = get_dataloader(data_path_test, config = config_path, device=device)

model = BC(config_path, device=device)
model.evaluate(test_data)
model.train_loop(train_data, save_dir)
model.evaluate(test_data)

##########3########
# test trained model
####################

# model_test = BC()
# model_test.to(device)
# check_point = torch.load('model.pth', weights_only=True)
# model_test.load_state_dict(check_point['model_state_dict'])
# model_test.test(train_data)