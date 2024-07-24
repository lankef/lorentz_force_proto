from mpi4py import MPI
import sys
import time
import pickle
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
sys.path.append('./code/main/')
import toroidal_surface
from vector_field_on_TS import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nproc_u=size-1
main=nproc_u # main proc number

#param
phisize=16,16
G=1e8
I=0
path_csw='code/input/cws'
path_plasma='code/input/plasma_surface'
path_output='intermediate_reg'

# loading of the surfaces
cws=pickle.load(open(path_csw,'rb'))
plasma_surf=pickle.load(open(path_plasma,'rb'))
#preparation for the computations
div_free=Div_free_vector_field_on_TS(cws)
m,n=phisize
l=2* ((m-1)*(2*n+1)+n)# nb of degree of freedom
if rank == main:
    A=np.zeros((plasma_surf.dim,l))
    CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(np.zeros(l),phisize)
    b=div_free.compute_normal_B(plasma_surf,(G,I,CC,CS))
    for k in range(l):
        recvarray=comm.recv( source = k%nproc_u)
        A[:,k]=recvarray
    with open(path_output, 'wb') as fp:
            pickle.dump((A,b),fp,protocol=2)
else:
    i=rank
    while i<l:
        t1=time.time()
        logging.info('computation of A : {}/{} by proc {}'.format(i,l,rank))
        zk=np.zeros(l)
        zk[i]=1
        CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,phisize)
        data=div_free.compute_normal_B(plasma_surf,(0,0,CC,CS))
        t2=time.time()
        print('proc {} successfully completed task {}/{} in {} s'.format(rank,i,l,t2-t1))
        comm.send(data, dest = main)
        i=i+nproc_u
    print("{:d} done".format(rank))
