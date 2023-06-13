import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from typing import List
from omegaconf import OmegaConf
import time
import argparse
import shutil
import subprocess
from utils.misc import omegaconf2dotlist, CONCAT, omegaconf2dict
from utils.Multiprocess import Task, Queue

timestamp = time.strftime("_%Y_%m%d_%H%M%S")

def gen_task_list(yaml_path:str, main_script_path:str):
    task_list = []
    opt = OmegaConf.load(yaml_path)
    # create a temp dir to save the opt in .yaml file
    temp_dir = opj(opd(yaml_path),'temp_opt'+timestamp)
    os.makedirs(temp_dir, exist_ok=True)
    temp_stdout = opj(opd(yaml_path),'temp_stdout'+timestamp)
    os.makedirs(temp_stdout, exist_ok=True)
    static = omegaconf2dotlist(opt.Static)
    dynamic_list = CONCAT(opt.Dynamic)
    dotlist_list = [static+dynamic for dynamic in dynamic_list]
    # task
    for task_idx,dotlist in enumerate(dotlist_list):
        task_opt_yaml = OmegaConf.from_dotlist(dotlist)
        task_opt_path = opj(temp_dir, str(task_idx) + '.yaml')
        OmegaConf.save(task_opt_yaml, task_opt_path)
        # task_opt_json = omegaconf2dict(task_opt_yaml)
        # task_opt_path = opj(temp_dir, str(task_idx)+'.json')
        # f_json = open(task_opt_path, 'w+')
        # json.dump(task_opt_json, f_json)
        # f_json.close()
        devide = False if task_opt_yaml.devide.type == 'None' else True
        command = "python {} -p {}".format(main_script_path, task_opt_path)
        stdout = opj(temp_stdout, str(task_idx) + '.log')
        task_list.append(Task(command, stdout, devide))
    return task_list, temp_dir, temp_stdout

def main():
    task_list, temp_dir, temp_stdout = gen_task_list(args.p, args.stp)
    try:
        queue = Queue(task_list, args.g)
        queue.start(args.t, remind=True, batch_compress=True)
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_stdout)
    except:
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_stdout)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Batch Compression')
    parser.add_argument('-p', type=str, default=opj(opd(__file__),
                        'opt', 'MultiTask', 'default.yaml'), help='config file path')
    parser.add_argument('-stp',type=str,default=opj(opd(__file__),'main.py'),help='the singletask script path')
    parser.add_argument('-g', help='availabel gpu list',default='0,1,2,3', type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-t',type=float,default=2,help='the time interval between each task-assigning loop. For batch compression')
    args = parser.parse_args()
    main()