import numpy as np

class Node:
    def __init__(self, name, num_gpus, flops_per_gpu, memory_gb,
                 bandwidth_ul_mhz, bandwidth_dl_mhz,
                 tx_power_ul_dbm, tx_power_dl_dbm,
                 pathloss_model):
        self.name = name
        self.num_gpus = num_gpus
        self.flops_per_gpu = flops_per_gpu  # e.g., 1.33e12 FLOPs
        self.memory = memory_gb
        self.bandwidth_ul_hz = bandwidth_ul_mhz * 1e6
        self.bandwidth_dl_hz = bandwidth_dl_mhz * 1e6
        self.tx_power_ul_dbm = tx_power_ul_dbm
        self.tx_power_dl_dbm = tx_power_dl_dbm
        self.pathloss_model = pathloss_model

def create_nodes():
    edge_server = Node(
        name="Edge Server",
        num_gpus=20,
        flops_per_gpu=1.33e12,
        memory_gb=32,
        bandwidth_ul_mhz=20,
        bandwidth_dl_mhz=20,
        tx_power_ul_dbm=20,
        tx_power_dl_dbm=43,
        pathloss_model="rayleigh"
    )

    uav = Node(
        name="UAV",
        num_gpus=1,
        flops_per_gpu=1.33e12,
        memory_gb=32,
        bandwidth_ul_mhz=1,
        bandwidth_dl_mhz=1,
        tx_power_ul_dbm=10,
        tx_power_dl_dbm=10,
        pathloss_model="free_space"
    )

    vehicle = Node(
        name="Vehicle",
        num_gpus=3,
        flops_per_gpu=1.33e12,
        memory_gb=32,
        bandwidth_ul_mhz=2,
        bandwidth_dl_mhz=2,
        tx_power_ul_dbm=23,
        tx_power_dl_dbm=30,
        pathloss_model="manhattan"
    )

    return edge_server, uav, vehicle
