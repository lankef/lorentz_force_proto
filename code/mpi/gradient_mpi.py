from mpi4py import MPI
import sys
import time
import pickle
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
sys.path.append('./code/main/')
import toroidal_surface
from regcoil import *
import avg_laplace_force
from tqdm import tqdm
from vector_field_on_TS import *
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nproc_u=size-1
main=nproc_u # main proc number

#param
lu,lv=64+1,64+1
lst_theta=range(lu-1)
lst_zeta=range(lv-1)
path_output='grad'
path_cws='code/li383/cws.txt'

phisize=4,4
m,n=phisize
I,G=1e6,2e5
np.random.seed(987)
lc=2* (m*(2*n+1)+n)
lst_coeff=1e3*(2*np.random.random(lc)-1)/(np.arange(1,lc+1)**2)
C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(lst_coeff,(m,n))
coeff=(G,I,C,S)
#initialization of the surface
cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(lu,lv),Np=3)
avg=avg_laplace_force.Avg_laplace_force(cws)
#send the gradient of j with respect to the coeff
djdc=np.zeros((lc,lu,lv,3))
if rank == main:
    djdc=get_djdc_naif(phisize,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid)
comm.Bcast(djdc, root=main)
#print(djdc.shape)
#preparation of the mpi parameters
step=math.ceil((len(lst_theta))/nproc_u)
blocks=lst_theta[rank*step:(rank+1)*step]


if rank == main:
    grad=np.zeros((lu-1,lv-1,lc,3))
    for k in range(nproc_u):
        #print('wainting for {}'.format(k))
        grad[k*step:(k+1)*step]=comm.recv( source = k)
    with open(path_output, 'wb') as fp:
            pickle.dump(grad,fp,protocol=2)
            print('all done')
else:
    t1=time.time()
    partial_grad=avg.grad_1_f_laplace(lu-1,lv-1,coeff,djdc,blocks,lst_zeta)
    partial_grad+=avg.grad_2_f_laplace(lu-1,lv-1,coeff,djdc,blocks,lst_zeta)
    print('proc {} done in {}'.format(rank,time.time()-t1))
    comm.send(partial_grad, dest = main)