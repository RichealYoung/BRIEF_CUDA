import os
import numpy as np
from omegaconf import OmegaConf
import argparse
import time
from utils.tool import read_img, save_img

def merge(origin_path, decompressed_dir, chunks_dir):
    origin_data = read_img(origin_path)
    decompressed_data = np.zeros_like(origin_data)
    chunk_names = os.listdir(chunks_dir)
    for chunk_name in chunk_names:
        chunk_path = os.path.join(chunks_dir, chunk_name)
        d1, d2, h1, h2, w1, w2 = [int(n) for n in chunk_name.split('.')[0].split('_')]
        chunk_data = read_img(chunk_path)
        decompressed_data[d1:d2,h1:h2,w1:w2] = chunk_data.reshape(decompressed_data[d1:d2,h1:h2,w1:w2].shape)
    
    origin_name = os.path.basename(origin_path).split('.')[0]
    save_img(os.path.join(decompressed_dir, origin_name + '_decompressed.tif'), decompressed_data)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='single task for compression')
    parser.add_argument('-d', type=str, default='outputs/divide_2023_0613_090347', help='project dir')
    args = parser.parse_args()
    project_dir = args.d
    script_dir = os.path.join(project_dir, 'script')
    for file in os.listdir(script_dir):
        if '.yaml' in file:
            opt_path = os.path.join(script_dir, file)
    decompressed_dir = os.path.join(project_dir, 'decompressed')
    if not os.path.exists(decompressed_dir):
        os.mkdir(decompressed_dir)
    chunks_dir = os.path.join(decompressed_dir, 'chunks')
    if not os.path.exists(chunks_dir):
        os.mkdir(chunks_dir)
    opt = OmegaConf.load(opt_path)
    origin_path = opt['compress']['data']['path']
    json_dir = os.path.join(project_dir, 'opts')
    json_names = os.listdir(json_dir)
    for json_name in json_names:
        json_path = os.path.join(json_dir, json_name)
        order = f'CUDA_VISIBLE_DEVICES=3 build/BRIEF -p {json_path} --only-decompress'
        os.system(order)
    merge(origin_path, decompressed_dir, chunks_dir)
    
