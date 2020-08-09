import os
import tensorflow as tf
from glob import glob
from processors import SimpleDataGenerator
from readers import LivoxDataReader
from config import Parameters
from network import build_point_pillar_graph

DATA_ROOT = "/media/nicholas/rkancharla/LiDAR/data/Livox/eval/"  
MODEL_ROOT = "./logs"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == "__main__":

    params = Parameters()
    pillar_net = build_point_pillar_graph(params, is_training=False)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    pillar_net.summary()

    data_reader = LivoxDataReader()

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "points", "*.bin")))
    label_files = sorted(glob(os.path.join(DATA_ROOT, "labels", "*.pkl")))

    eval_gen = SimpleDataGenerator(data_reader, params.eval_batch_size, lidar_files)

    pred = pillar_net.predict(eval_gen, batch_size=1)

    print(pred)
    