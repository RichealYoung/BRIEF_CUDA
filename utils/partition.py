import os
import copy
import math
import numpy as np
from utils.adaptive_blocking import OctTree, cal_feature

def no_partition(devide_type, origin_data):
    shape = origin_data.shape
    chunk_names = [f'0_{shape[0]}_0_{shape[1]}_0_{shape[2]}']
    chunk_datas = [origin_data]
    partition_result = copy.deepcopy(origin_data)
    return chunk_names, chunk_datas, partition_result

def equal_partition(devide_type, origin_data):
    dn, hn, wn = [int(n) for n in devide_type.split('_')[1:]]
    shape = origin_data.shape
    assert shape[0]%dn == 0 and shape[1]%hn == 0 and shape[2]%wn == 0, "Data can't be devided equally"
    chunk_names = []
    chunk_datas = []
    partition_result = copy.deepcopy(origin_data)
    for i in range(dn):
        for j in range(hn):
            for k in range(wn):
                d1, d2, h1, h2, w1, w2 = int(i/dn*shape[0]), int((i+1)/dn*shape[0]), int(j/hn*shape[1]), int((j+1)/hn*shape[1]), int(k/wn*shape[2]), int((k+1)/wn*shape[2])
                chunk_name = f'{d1}_{d2}_{h1}_{h2}_{w1}_{w2}'
                chunk_data = origin_data[d1:d2, h1:h2, w1:w2]
                chunk_names.append(chunk_name)
                chunk_datas.append(chunk_data)
                partition_result[d1,h1:h2,w1:w2] = 2000
                # partition_result[d2-1,h1:h2,w1:w2] = 2000
                partition_result[d1:d2,h1,w1:w2] = 2000
                # partition_result[d1:d2,h2-1,w1:w2] = 2000
                partition_result[d1:d2,h1:h2,w1] = 2000
                # partition_result[d1:d2,h1:h2,w2-1] = 2000
    return chunk_names, chunk_datas, partition_result


def adaptive_partition(devide_type, origin_data):
    if len(devide_type.split('_')) == 2:
        Nb = devide_type.split('_')[-1]
        devide_type = f'adaptive_-1_-1_0_0_{Nb}_1'
    # adaptive_maxl_minl_varthr_ethr_Nb_Type
    maxl, minl, varthr, ethr, Nb, Type = [int(n) for n in devide_type.split('_')[1:]]
    data = copy.deepcopy(origin_data)
    if minl == -1:
        minl = math.floor(math.log(Nb, 8))
    if maxl == -1:
        maxl = minl + 2
    tree = OctTree(data, maxl, minl, Type, varthr, ethr)
    tree.solve_optim(Nb)
    info = 'maxl:{},minl:{},var_thr:{},e_thr:{},Nb:{}'.format(maxl,minl,varthr,ethr,Nb)
    print(info)
    print('number of blocks:{}'.format(len(tree.get_active())))

    chunk_names = [f'{patch.z}_{patch.z+patch.d}_{patch.y}_{patch.y+patch.h}_{patch.x}_{patch.x+patch.w}' for patch in tree.get_active()]
    chunk_datas = [patch.data for patch in tree.get_active()]
    partition_result = copy.deepcopy(origin_data)
    partition_result = tree.draw(partition_result)

    return chunk_names, chunk_datas, partition_result

def partition(devide_type, origin_data):
    if 'equal' in devide_type:
        return equal_partition(devide_type, origin_data)
    elif 'None' in devide_type:
        return no_partition(devide_type, origin_data)
    elif 'adaptive' in devide_type:
        return adaptive_partition(devide_type, origin_data)
    else:
        raise NotImplemented

def param_allocate(allocate_type, chunk_datas, ideal_params, layer):
    ratios = []
    for i in range(len(chunk_datas)):
        chunk = chunk_datas[i]
        if allocate_type == 'equal':
            ratios.append(1)
        elif allocate_type == 'by_size':
            size = chunk.size
            ratios.append(size)
        elif allocate_type == 'by_var':
            var = ((chunk-chunk.mean())**2).mean()
            ratios.append(var)
        elif allocate_type == 'by_d':
            d = 1/cal_feature(chunk)
            ratios.append(d)
        elif allocate_type == 'by_dv':
            dv = chunk.size/cal_feature(chunk)
            ratios.append(dv)
        elif allocate_type == 'by_aoi':
            aoi = (chunk>0).sum()
            ratios.append(aoi)
        else:
            raise NotImplemented
    ratios_sum = sum(ratios)
    chunk_params = [ratio/ratios_sum*ideal_params for ratio in ratios]

    features = []
    theory_params = 0
    a, b = layer-1, layer+4
    for chunk_param in chunk_params:
        # 3*n+n+(l-1)(n^2+n)+n+1=p -> (l-1)n^2+(l+4)n+(1-p)=0
        c = 1 - chunk_param
        feature = (-b+math.sqrt(b**2-4*a*c))/(2*a)
        feature = round(feature/8)*8
        if feature<8:
            feature = 8
        features.append(feature)
        theory_params += (layer-1)*feature**2+(layer+4)*feature+1

    return features, theory_params


