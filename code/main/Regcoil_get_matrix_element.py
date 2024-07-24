import numpy as np
import vector_field_on_TS
import toroidal_surface
import pickle
import logging

def Regcoil_get_matrix_element(G,I,phisize,paths=None,surfs=None,cache=True):
    """compute the matrix elements for Regcoil, use """
    if paths is not None:
        path_csw,path_plasma=paths
        # loading of the surfaces
        cws=pickle.load(open(path_csw,'rb'))
        plasma_surf=pickle.load(open(path_plasma,'rb'))
    else:
        cws,plasma_surf=surfs
    #computations of the matrix elements
    div_free=vector_field_on_TS.Div_free_vector_field_on_TS(cws)
    key=str(cws.X)+str(plasma_surf.X)+str(G)+str(I)+str(phisize[0]+phisize[1])
    dic=OnlyOne()
#    (A,b)=div_free.compute_matrix_cost(plasma_surf,G,I,phisize)
    if (key in dic.instance.val.keys()) and cache:
        logging.info('A and b were found in cache,no computation needed')
        (A_B,b_B,tensor_j_K,tensor_b_K)=dic.instance.val[key]
    else:
        (A_B,b_B)=div_free.compute_matrix_cost(plasma_surf,G,I,phisize)
        tensor_j_K,tensor_b_K=div_free.get_Chi_K(phisize,G,I)
        dic.instance.val[key]=(A_B,b_B,tensor_j_K,tensor_b_K)
        dic.save()
    #ponderation is not included in A and b

    return (A_B.copy(),tensor_j_K.copy()),(b_B.copy(),tensor_b_K.copy()),div_free,cws,plasma_surf
class OnlyOne:
    """singleton, contain a dictionary with already computed simulation"""
    path='cache/reg'
    class __OnlyOne:
        def __init__(self):
            try:
                with open(OnlyOne.path, 'rb') as fp:
                    self.val = pickle.load(fp)
                    print('opening '+OnlyOne.path)
            except IOError as err:
                print('not existing file : '+OnlyOne.path)
                print(err)
                self.val={}
    instance = None
    def __init__(self):
        if not OnlyOne.instance:
            OnlyOne.instance = OnlyOne.__OnlyOne()
        else :
            pass
    def add(self,key,a):
        self.update()
        print('key',key,'value',a)
        OnlyOne.instance.val[key]=a
        self.save()
    def update(self):
        try:
            with open(OnlyOne.path, 'rb') as fp:
                OnlyOne.instance.val = pickle.load(fp)
        except:
            print('unable to update')
    def save(self):
        print('saving file')
        with open(OnlyOne.path, 'wb') as handle:
            pickle.dump(OnlyOne.instance.val, handle)