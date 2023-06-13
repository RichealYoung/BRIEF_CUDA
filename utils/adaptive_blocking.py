import gurobipy as gp
import cv2
import numpy as np
from math import log
import copy
import math
import tifffile
from tqdm import tqdm
import sys

def cal_feature(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # f = np.fft.fft(np.fft.fft(gray,axis=0),axis=1)
        f = np.fft.fftn(gray, axes=(0,1))
    elif len(image.shape) == 4:
        # f = np.fft.fft(np.fft.fft(np.fft.fft(image,axis=0),axis=1),axis=2)
        f = np.fft.fftn(image, axes=(0,1,2))
    f = np.abs(f)
    feature = int(f.max())/int(f.sum())
    return feature

class Patch3d():
    def __init__(self, optim_model, parent, level, orderx, ordery, orderz) -> None:
        self.level = level
        self.orderx = orderx
        self.ordery = ordery
        self.orderz = orderz
        self.parent = parent
        self.children = []
        self.optim_model = optim_model
        self.data_active = False
        self.feature = 0
        self.active()
    def get_children(self):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    orderx = 2*self.orderx + k
                    ordery = 2*self.ordery + j
                    orderz = 2*self.orderz + i
                    child = Patch3d(self.optim_model,parent=self,level=self.level+1,orderx=orderx,ordery=ordery,orderz=orderz)
                    self.children.append(child)
        return self.children
    def init_data(self,data,d,h,w):
        self.data_active = True
        self.d = d//(2**self.level)
        self.h = h//(2**self.level)
        self.w = w//(2**self.level)
        self.z = self.d*self.orderz
        self.y = self.h*self.ordery
        self.x = self.w*self.orderx
        self.data = copy.deepcopy(data[self.z:self.z+self.d,self.y:self.y+self.h,self.x:self.x+self.w])
        self.var = ((self.data-self.data.mean())**2).mean()
        self.mean = abs(self.data.mean())
        return self.data
    def get_feature(self, Type):
        if Type == 0:
            self.feature = self.var
        elif Type == 1:
            self.feature = cal_feature(self.data)
        else:
            raise NotImplementedError
        return self.feature
    def active(self):
        self.prune = False
        self.active = self.optim_model.addVar(vtype=gp.GRB.BINARY,name=f"{self.level}-{self.orderx}-{self.ordery}-{self.orderz}")
    def deactive(self):
        self.prune = True
        self.optim_model.remove(self.active)

class OctTree():
    def __init__(self, data, max_level, min_level, Type, var_thr, e_thr):
        # data (d,h,w)
        self.Type = Type
        self.data = data
        self.d = data.shape[0]  
        self.h = data.shape[1]
        self.w = data.shape[2]
        self.max_level = max_level
        self.min_level = min_level
        assert len(self.data.shape) == 3 or (len(self.data.shape) == 4 and self.data.shape[-1]==1),"data must be 3d!"
        assert self.d%(2**max_level) ==0 and self.h%(2**max_level) == 0 and self.w%(2**max_level) == 0,"image size error!"
        # optimizer
        self.optim_model = gp.Model()
        # octtree and addvar
        self.tree = Patch3d(self.optim_model,parent=None,level=0, orderx=0, ordery=0, orderz=0)
        self.patch_list = []   
        self.patch_level_list = [] 
        self.patch_dict = {}   
        self.init_tree(self.tree,0)
        self.tree2list(self.tree)
        self.tree2dict(self.tree)
        self.init_data()
        self.prune(var_thr,e_thr)
        self.get_feature()
        self.optim_model.update()
    def init_tree(self,parent,level):
        if level < self.max_level:
            children = parent.get_children()
            for child in children:
                self.init_tree(child,level+1)
    def tree2list(self,patch):
        self.patch_list.append(patch)
        if patch.level >= self.min_level and patch.level <= self.max_level: 
            self.patch_level_list.append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2list(child)
    def tree2dict(self,patch):
        if not (str(patch.level) in self.patch_dict):
            self.patch_dict[str(patch.level)] = [patch]
        else:
            self.patch_dict[str(patch.level)].append(patch)
        children = patch.children
        if len(children) != 0:
            for child in children:
                self.tree2dict(child)
    def init_data(self):
        for patch in self.patch_level_list:
            patch.init_data(self.data,self.d,self.h,self.w)
    def get_depth(self):
        patch = self.tree
        while len(patch.children) != 0:
            patch = patch.children[0]
        return patch.level
    def get_feature(self):
        # for path in self.patch_level_list:
        for patch in tqdm(self.patch_level_list, desc='Cal feature', leave=True, file=sys.stdout):
            if patch.prune == False: 
                patch.get_feature(self.Type)
    def get_descendants(self,patch):
        descendants = []
        children = patch.children
        descendants += children
        if len(children) == 0:
            return []
        for child in children:
            descendants += self.get_descendants(child)
        return descendants
    def get_genealogy(self,patch):  
        genealogy = [patch]
        while (patch.parent != None):
            genealogy.append(patch.parent)
            patch = patch.parent
        return genealogy
    def solve_optim(self, Nb):
        self.Nb = Nb
        Obj = []
        Constr = []
        for patch in self.patch_list:
            if patch.prune == False:    
                Obj.append(patch.feature*patch.active/(8**patch.level))
                Constr.append(patch.active)
                # 4.the active chunk's level should larger than the min_level
                if patch.level < self.min_level:
                    self.optim_model.addConstr(patch.active == 0)
        if self.Type == 0:
            self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MINIMIZE)
        elif self.Type == 1:
            self.optim_model.setObjective(gp.quicksum(Obj), gp.GRB.MAXIMIZE)
        else:
            raise NotImplementedError
        # Add constraints
        # 1.the total numbers of the active chunks should not be larger than the set value
        self.optim_model.addConstr(gp.quicksum(Constr) <= self.Nb)
        # 2.only one member can be active in the same genealogy
        depth = self.get_depth()
        deepest_layer = self.patch_dict[str(depth)]
        for patch in deepest_layer:
            genealogy = self.get_genealogy(patch)
            actives = []
            for patch in genealogy:
                if patch.prune == False: 
                    actives.append(patch.active)
            # 3.if one member is pruned, the numbers of the other active members in the same genealogy should lease than one
            if len(actives) < len(genealogy) and len(actives) >= 2:
                self.optim_model.addConstr(gp.quicksum(actives) <= 1)
                # print(len(genealogy)-len(actives))
            elif len(actives) == len(genealogy):
                self.optim_model.addConstr(gp.quicksum(actives) == 1)
        # Solve it!
        self.optim_model.optimize()
        # print(f"Optimal objective value: {self.optim_model.objVal}")
    def prune(self,var_thr:float=0,e_thr:float=0):
        count = 0
        for patch in self.patch_list:
            if patch.data_active and patch.var <= var_thr and patch.mean <= e_thr:
                # print(patch.variance)
                patch.deactive()
                count += 1
                descendants = self.get_descendants(patch)
                for descendant in descendants:
                    descendant.deactive()
                    count += 1
        print(f'prune numbers:{count}')
    def get_active(self):
        self.active_patch_list = []
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    self.active_patch_list.append(patch)
        return self.active_patch_list
    def draw(self,data:np.array=None):
        if data.any() == None:
            data = copy.deepcopy(self.data)
        for patch in self.patch_list:
            if patch.prune == False:
                if int(patch.active.x) == 1:
                    x,y,z,w,h,d = patch.x,patch.y,patch.z,patch.w,patch.h,patch.d
                    data[z,y:y+h,x:x+w] = 2000
                    # data[z+d-1,y:y+h,x:x+w] = 2000
                    data[z:z+d,y,x:x+w] = 2000
                    # data[z:z+d,y+h-1,x:x+w] = 2000
                    data[z:z+d,y:y+h,x] = 2000
                    # data[z:z+d,y:y+h,x+w-1] = 2000
        return data
    def draw_tree(self):    
        actives = {}
        for patch in self.patch_list:
            if not (str(patch.level) in actives):
                actives[str(patch.level)] = [int(not patch.prune)]
            else:
                actives[str(patch.level)].append(int(not patch.prune))
        for key in actives.keys():
            print(actives[key])

def adaptive_cal_tree(img, var_thr:float=-1,e_thr:float=-1, maxl:int=-1, minl:int=-1, Nb:int=-1, Type=1):
    data = copy.deepcopy(img)

    minl = math.floor(log(Nb, 8))
    maxl = minl + 2
    tree = OctTree(data, maxl, Type, var_thr, e_thr)

    tree.solve_optim(Nb,minl)
    save_data = copy.deepcopy(img)
    save_data = tree.draw(save_data)
    info = 'maxl:{},minl:{},var_thr:{},e_thr:{},Nb:{}'.format(maxl,minl,var_thr,e_thr,Nb)
    print(info)
    print('number of blocks:{}'.format(len(tree.get_active())))
    return tree, save_data

if __name__=="__main__":
    data_path = 'data/test.tif'
    data = tifffile.imread(data_path)
    adaptive_cal_tree(data, Nb = 20)
    
