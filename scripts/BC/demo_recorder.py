import os
import h5py
import numpy as np
import torch

class DemoRecorder:
    def __init__(self, file_path:str, num_demo:int, num_env: int, max_epi:int):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.num_demo_to_collect = num_demo
        self.num_saved_demo = 0
        self.num_env = num_env
        self.num_max_episodes = max_epi
        self.obs_data = [ [] for _ in range(num_env)]
        self.actions_data = [ [] for _ in range(num_env)]
        self.demo_data = h5py.File(file_path, 'a')

    def record(self, obs:np.ndarray, actions:np.ndarray, episode_buffer:torch.Tensor) -> None:
        for env_index in range(self.num_env):
            self.obs_data[env_index].append(obs[env_index])
            self.actions_data[env_index].append(actions[env_index])

            if (episode_buffer[env_index] >= self.num_max_episodes - 1 and
                self.num_saved_demo < self.num_demo_to_collect):

                if not self.demo_data:
                    raise RuntimeError(f"Try to write a closed file")

                obs_arr = np.stack(self.obs_data[env_index], axis=0)
                actions_arr = np.stack(self.actions_data[env_index], axis=0)

                data_pointer = self.demo_data.create_group(f"demo_{self.num_saved_demo}")
                data_pointer.create_dataset("observation", data=obs_arr)
                data_pointer.create_dataset('action', data=actions_arr)

                self.num_saved_demo += 1

                self.obs_data[env_index] = []
                self.actions_data[env_index] = []

                if self.num_saved_demo == self.num_demo_to_collect:
                    self.close()
                    print(f"demo collection completed")

    def close(self):
        if not self.demo_data:
            raise RuntimeError(f"File is already closed")
        else:
            self.demo_data.close()
            self.demo_data = None
            print(f" Demo data file closed")
