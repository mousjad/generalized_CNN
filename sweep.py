# Import the W&B Python Library and log into W&B
import wandb

wandb.login()

sweep = {
    "method": "random",
    "name": "homemade_CNN_sweep",
    "metric": {
        "name": '"',
        "goal": "minimize"
    },
    "parameters": {
        "batch_size": {
            "values": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        },
        "lr": {
            "distribution": "uniform",
            "max": 0.1,
            "min": 0.0001
        },
        "w1": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 16
        },
        "w2": {
            "distribution": "int_uniform",
            "min": 16,
            "max": 32
        },
        "w3": {
            "distribution": "int_uniform",
            "min": 32,
            "max": 64
        },
        "w4": {
            "distribution": "int_uniform",
            "min": 64,
            "max": 128
        },
        "w5": {
            "distribution": "int_uniform",
            "min": 128,
            "max": 256
        },
        "w6": {
            "distribution": "int_uniform",
            "min": 256,
            "max": 1024
        },
        "w7": {
            "distribution": "int_uniform",
            "max": 1024,
            "min": 512
        },
        "w8": {
            "distribution": "int_uniform",
            "max": 512,
            "min": 128
        },
        "w9": {
            "distribution": "int_uniform",
            "max": 128,
            "min": 8
        },
        "w10": {
            "distribution": "int_uniform",
            "max": 8,
            "min": 2
        },
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 10
    }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep, project="generalized CNN")
from model import train_generalized_CNN, homemade_cnn
wandb.agent(sweep_id, function=train_generalized_CNN, count=50)
