from copy import copy
import os 
import numpy as np
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
import argparse
import json
import shutil
import copy
import time
from omegaconf import OmegaConf
from utils.logger import MyLogger, reproduc
from utils.Multiprocess import Task, Queue
from utils.partition import partition, param_allocate
from utils.tool import read_img, save_img, get_folder_size
from utils.metrics import eval_performance
from utils.misc import omegaconf2dict

class CompressFramework:
    def __init__(self, opt, Log) -> None:
        self.opt = opt
        self.Log = Log
        self.compress_opt = opt['compress']
        self.origin_path = opt['compress']["data"]["path"]
        self.metrics = {}
    
    def save_chunk_data(self, chunk_name, chunk_data, chunk_feature):
        # save chunk data
        chunk_path = opj(self.Log.chunks_dir, chunk_name + '.tif')
        save_img(chunk_path, chunk_data)
        # change chunk opt
        opt = copy.deepcopy(self.compress_opt)
        opt['data']['path'] = opj(self.Log.chunks_dir, chunk_name + '.tif')
        opt['compressed_path'] = opj(self.Log.compressed_dir, chunk_name + '.msgpack')
        opt['decompressed_path'] = opj(self.Log.decompressed_chunks_dir, chunk_name + '.tif')
        opt['network']['n_neurons'] = chunk_feature
        # save chunk opt
        opt_path = opj(self.Log.opts_dir, chunk_name + '.json')
        f = open(opt_path, 'w+')
        json.dump(opt, f, indent=2)
        f.close()
        return opt_path
    
    def cal_param(self, chunk_datas):
        file_ratio = self.opt['param']['file_ratio']
        given_size = self.opt['param']['given_size']     
        assert file_ratio*given_size==0 and file_ratio+given_size> 0, "file_ratio or given_size error"
        self.metrics['origin_bytes'] = os.path.getsize(self.origin_path)
        # ideal params
        if file_ratio > 0:
            self.metrics['set_ratio'] = int(file_ratio)
            self.metrics['ideal_bytes'] = int(self.metrics['origin_bytes']/file_ratio)
        else:
            self.metrics['set_ratio'] = int(self.metrics['origin_bytes']/given_size)
            self.metrics['ideal_bytes'] = int(given_size)
        self.metrics['ideal_params'] = int(self.metrics['ideal_bytes']/2)
        # theory params
        chunk_features, self.metrics['theory_params'] = param_allocate(self.opt['devide']['allocate'], 
            chunk_datas, self.metrics['ideal_params'], self.compress_opt['network']['n_hidden_layers'])
        self.metrics['theory_bytes'] = self.metrics['theory_params']*2
        self.metrics['theory_ratio'] = int(self.metrics['origin_bytes']/self.metrics['theory_bytes'])
        return chunk_features

    def merge(self):
        chunks_dir = self.Log.decompressed_chunks_dir
        origin_data = read_img(self.origin_path)
        decompressed_data = np.zeros_like(origin_data)
        chunk_names = os.listdir(chunks_dir)
        for chunk_name in chunk_names:
            chunk_path = opj(chunks_dir, chunk_name)
            d1, d2, h1, h2, w1, w2 = [int(n) for n in chunk_name.split('.')[0].split('_')]
            chunk_data = read_img(chunk_path)
            decompressed_data[d1:d2,h1:h2,w1:w2] = chunk_data.reshape(decompressed_data[d1:d2,h1:h2,w1:w2].shape)
        
        origin_name = opb(self.origin_path).split('.')[0]
        save_img(opj(self.Log.decompressed_dir, origin_name + '_decompressed.tif'), decompressed_data)
        self.metrics['psnr'], self.metrics['ssim'], self.metrics['acc200'], self.metrics['acc500'] = eval_performance(origin_data, decompressed_data)
    
    def get_metrics(self):
        self.metrics['actual_bytes'] = get_folder_size(self.Log.compressed_dir)
        self.metrics['actual_ratio'] =  self.metrics['origin_bytes']/self.metrics['actual_bytes']

        important_metrics = {}
        important_metrics['set_ratio'] = self.metrics['set_ratio']
        important_metrics['theory_ratio'] = self.metrics['theory_ratio']
        important_metrics['actual_ratio'] = self.metrics['actual_ratio']
        important_metrics['time'] = self.metrics['time']
        important_metrics['psnr'] = self.metrics['psnr']
        important_metrics['ssim'] = self.metrics['ssim']
        important_metrics['acc200'] = self.metrics['acc200']
        important_metrics['acc500'] = self.metrics['acc500']

        Unimportant_metrics = {}
        for key in self.metrics.keys():
            if not key in important_metrics.keys():
                Unimportant_metrics[key] = self.metrics[key]

        f_metrics = open(opj(self.Log.project_dir, 'important_metrics.json'), 'w+')
        json.dump(important_metrics, f_metrics)
        f_metrics.close()

        f_metrics = open(opj(self.Log.project_dir, 'unimportant_metrics.json'), 'w+')
        json.dump(Unimportant_metrics, f_metrics)
        f_metrics.close()

        self.Log.log_metrics(important_metrics, 0)
        self.Log.close()

    def compress(self):
        time_start = time.time()
        origin_data = read_img(self.origin_path)
        # partition and parallel operation
        task_list = []
        chunk_names, chunk_datas, partition_result = partition(self.opt['devide']['type'], origin_data)
        save_img(opj(self.Log.project_dir, 'partition(' + self.opt['devide']['type'] + ').tif'), partition_result)
        chunk_features = self.cal_param(chunk_datas)
        
        f_command = open(opj(self.Log.script_dir, 'command.txt'), 'w+')
        for i in range(len(chunk_names)):
            chunk_name = chunk_names[i]
            chunk_data = chunk_datas[i]
            chunk_feature = chunk_features[i]
            opt_path = self.save_chunk_data(chunk_name, chunk_data, chunk_feature)

            command = f'build/BRIEF -p {opt_path}'
            stdout = opj(self.Log.stdout_dir, chunk_name + '.log')
            devide = True if self.opt['devide']['type'] == 'None' else False
            task_list.append(Task(command, stdout, devide))
            
            f_command.write(command + '\n')
        f_command.close()

        try:
            queue = Queue(task_list, args.g)
            queue.start(args.t, remind=False, batch_compress=False)
        except:
            pass
        self.metrics['time'] = time.time()-time_start
        print('Compression time: ' + str(self.metrics['time']) + 's')
        self.merge()
        self.get_metrics()

def load_config(config_path):
    if '.json' in config_path:
        f_opt = open(config_path,'r+')
        opt = json.load(f_opt)
    elif '.yaml' in config_path:
        opt = omegaconf2dict(OmegaConf.load(config_path))
    return opt
        
def main():
    opt = load_config(args.p)
    Log = MyLogger(**opt['log'])
    shutil.copy(args.p, Log.script_dir)
    shutil.copy(__file__, Log.script_dir)
    reproduc(opt["reproduc"])

    Compressor = CompressFramework(opt=opt, Log=Log)
    Compressor.compress()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single task for compression')
    parser.add_argument('-p', type=str, default=opj(opd(__file__),
                        'opt', 'SingleTask', 'default.yaml'), help='config file path')
    parser.add_argument('-g', help='availabel gpu list', default='0,1,2,3',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-t',type=float,default=2,help='the time interval between each task-assigning loop. For compress_divide')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.g])
    main()
