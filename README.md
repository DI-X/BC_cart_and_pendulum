# BC_cart_and_pendulum
- AI agent will be trained using Behavior Cloning (BC) algorithm with demonstrations from  another AI agent which is trained with PPO.
- This rep was developed just to learn  the implementation of BC algorithm 
# Files

/scripts/train_expert.py:
 - train an AI agent using PPO

/scripts/record_demos.py :
- records the demonstration from an expert agent 

/scripts/train_BC_agent.py:
 - train an AI agent using Behavior Cloning (BC)
 - let's call this AI agent "BC agent"

/scripts/test_BC_agent.py
- test a BC agent in Isaac Sim physics simulator

/script/BC
- core scripts to record demonstration and to train BC agents

```text
.
├── scripts
│   ├── train_expert.py        # Train an AI agent using PPO
│   ├── record_demos.py        # Record demonstrations from an expert agent
│   ├── train_BC_agent.py      # Train an AI agent using Behavior Cloning (BC)
│   ├── test_BC_agent.py       # Test a BC agent in Isaac Sim physics simulator
│   └── BC/                    # Core scripts to record demonstrations and train BC agents
```


## Prerequisite 
Isaac Lab and all of its dependencies must be installed. Follow the instruction in Isaac Lab website: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html 

## How to use
Following Isaac Lab instruction, it recommended to use the conda environment
### 1. How to train an expert agent
- `issaclab.sh -p train_expert.py --headless --num_envs 4064`
-  you can modify the setting of this expert agent in a config file `${directory_of_this_repo}/config/sb3_agent.yaml`
- trained an agent (weight of NN) will be saved in `${directory_of_this_repo}/logs/BC/${date_and_time}`

### 2. Record demonstrations from an expert agent
- `issaclab.sh -p record_demos.py --log_time ${date_and_time_of_directory} --checkpoint ${name_of_check_point}`

### 3. Train a BC agent
- `python3 train_BC_agent.py`
- you can modify the settng of BC agent in a config file located `${directory_of_this_repo}/config/bc_agent.yaml`

### 4. Test a BC agent
- `issaclab.sh -p test_BC_agent.py --log_time ${date_and_time_of_directory} --checkpoint ${name_of_check_point}`

## Tips

- Just for your convenience, add the following into your bachrc 
- `alias  isaaclab="${parents_directory_of_IsaacLab}/IsaacLab/isaaclab.sh"`
-  with the above, now you can use isaaclab.sh with the command, "isaaclab" from anywhere

## BC trained results:

coming soon:

