from datasets import load_dataset
import numpy as np

orca_data = load_dataset("Open-Orca/OpenOrca", split="train")

def get_random_prompt():
    return orca_data[np.random.randint(0, len(orca_data))]["question"]
