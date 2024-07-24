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
from f_e import get_f
from tqdm import tqdm
from vector_field_on_TS import *
import math
import scipy.optimize
import configparser

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nproc_u=size-1
main=nproc_u # main proc number
#param
config = configparser.ConfigParser()
config.read(sys.argv[1])
#print(config.sections())
Np=int(config['geometry']['Np'])
ntheta_plasma = int(config['geometry']['ntheta_plasma'])+1
ntheta_coil   = int(config['geometry']['ntheta_coil'])+1
nzeta_plasma = int(config['geometry']['nzeta_plasma'])+1
nzeta_coil   = int(config['geometry']['nzeta_coil'])+1
mpol_coil  = int(config['geometry']['mpol_coil'])
ntor_coil  = int(config['geometry']['ntor_coil'])
net_poloidal_current_Amperes = float(config['other']['net_poloidal_current_Amperes'])#11884578.094260072
net_toroidal_current_Amperes = float(config['other']['net_toroidal_current_Amperes'])#0
curpol=float(config['other']['curpol'])#4.9782004309255496
#to delete later :
lu,lv=ntheta_coil,nzeta_coil
#lamb1=1.2e-14
#lamb2=2.5e-16
#lamb3=5.1e-19

lamb=float(config['other']['lamb'])
lamb_init=float(config['other']['lamb_init'])
gamma=float(config['other']['gamma'])

if config.has_option('other','lamb_H1dot'):
    lamb_H1dot=float(config['other']['lamb_H1dot'])
else:
    lamb_H1dot=0
path_plasma=str(config['geometry']['path_plasma'])#'code/li383/plasma_surf.txt'
path_cws=str(config['geometry']['path_cws'])#'code/li383/cws.txt'
path_bnorm=str(config['other']['path_bnorm'])#'code/li383/bnorm.txt'
path_output=str(config['other']['path_output'])#'coeff_full_opt'
if config.has_option('other','use_f_e'):
    use_f_e = bool(config['other']['use_f_e'])
else :
    use_f_e=False

    

#number of coeffient for j
lc=2* (mpol_coil*(2*ntor_coil+1)+ntor_coil)

if use_f_e:
    c0 = float(config['other']['c0'])
    c1 = float(config['other']['c1'])
    f_e,grad_f_e=get_f(c0,c1,lu,lv,lc)
#mc is just a multiplicatif coefficient to increase the cost
mc=1e5


phisize=(ntor_coil,mpol_coil)
G,I=net_poloidal_current_Amperes/Np,net_toroidal_current_Amperes


plasma_surf=toroidal_surface.Toroidal_surface(W7x_pathfile=path_plasma,nbpts=(ntheta_plasma,nzeta_plasma),Np=3)
cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(ntheta_coil,nzeta_coil),Np=3)
div_free=vector_field_on_TS.Div_free_vector_field_on_TS(cws)
avg=avg_laplace_force.Avg_laplace_force(cws)
#send the gradient of j with respect to the coeff
lst_k=range(lc)
step=math.ceil((len(lst_k))/nproc_u)
blocks=lst_k[rank*step:(rank+1)*step]

djdc=np.zeros((lc,ntheta_coil,nzeta_coil,3))
if rank == main:
    #djdc_2=get_djdc_naif(phisize,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid)
    #djdc=djdc_2
    for k in range(nproc_u):
        djdc[k*step:(k+1)*step]=comm.recv( source = k)
    #np.testing.assert_almost_equal(djdc,djdc_2)
    #print('OK')
    logging.info('djdc computation is successfull')
else:
    #pass
    djdc_partial=get_djdc_partial(phisize,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid,blocks)
    comm.send(djdc_partial, dest = main)

comm.Bcast(djdc, root=main)
matrix_gradj=np.zeros((lc+1,2,lu,lv,3))
if rank==main:
    matrix_gradj=div_free.get_matrix_gradj(phisize,G,I)
comm.Bcast(matrix_gradj, root=main)

matH1dot=np.zeros((lc+1,lc+1))
if rank==main:
    matH1dot=get_norm_H1p(phisize,cws.g_upper,cws.dS,cws.grid,matrix_gradj)
comm.Bcast(matH1dot, root=main)
logging.debug('proc {} done'.format(rank))
coeff_l=np.zeros(lc)
#We start by computing Regcoil
if rank==main:
    logging.info('Get info for Regcoil')
    (A_B,tensor_j_K),(b_B,tensor_b_K),div_free,cws,plasma_surf=Regcoil_get_matrix_element.Regcoil_get_matrix_element(G,I,phisize,surfs=(cws,plasma_surf))
    #normalization of the matrices:
    norm_normal_plasma_vec = plasma_surf.dS[:-1,:-1].flatten()
    diag_norm_normal_plasma = np.diag(norm_normal_plasma_vec)
    #computation of the matrices for the optimization
    if path_bnorm is not None:
        b_array=get_bnorm(path_bnorm,plasma_surf)
        b_vec=-curpol*(b_array[:-1,:-1]).flatten()
        b_B+=b_vec
    Mat_B=np.dot(A_B.transpose(),np.dot(diag_norm_normal_plasma,A_B))/((plasma_surf.lu-1)*(plasma_surf.lv-1))#A^T S A
    RHS_B=np.dot(b_B.transpose(),np.dot(diag_norm_normal_plasma,A_B))/((plasma_surf.lu-1)*(plasma_surf.lv-1))#b^T S A

    tmp_K=np.reshape(tensor_j_K,(tensor_j_K.shape[0],-1))#without the surface element
    tensor_j_K_S=np.einsum('ijkl,jk->ijkl', tensor_j_K, cws.dS[:-1,:-1]) #with the surface element
    tmp_K_S=np.reshape(tensor_j_K_S,(tensor_j_K_S.shape[0],-1))#with the surface element
    Mat_K=np.dot(tmp_K_S,tmp_K.transpose())/((cws.lu-1)*(cws.lv-1))#A^T S A
    RHS_K=np.dot(tmp_K_S,tensor_b_K.flatten())/((cws.lu-1)*(cws.lv-1))#b^T S A

    Mat=Mat_B+lamb*Mat_K
    RHS=RHS_B+lamb*RHS_K
    #for initialization
    if lamb_init!=-1:
        Mat_init=Mat_B+lamb_init*Mat_K
        RHS_init=RHS_B+lamb_init*RHS_K
        coeff_l=np.linalg.solve(Mat_init,RHS_init)
        logging.info('Regcoil optimization successfull')
    else :
        path_init_param=str(config['other']['init_param'])
        f=open(path_init_param,'rb')
        coeff_l=pickle.load(f)
        f.close()
        logging.info('loading of initial coeff_l successfull')
    err=[]
    
    #the cost function
    def cost(coeff_l):
        #logging.info(str(coeff_l))
        surf_err=np.dot(A_B,coeff_l)-b_B
        avg_surf_err=np.dot(surf_err,np.dot(diag_norm_normal_plasma,surf_err))/((plasma_surf.lu-1)*(plasma_surf.lv-1))
        j_err2=np.einsum('i...,i->...',tensor_j_K,coeff_l)-tensor_b_K
        avg_err_j=3*np.mean(j_err2**2*cws.dS[:-1,:-1,np.newaxis])
        sol_C,sol_S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(coeff_l,phisize)
        coeff=(G,net_toroidal_current_Amperes,sol_C,sol_S)
        laplace_array=avg.f_laplace_optimized(ntheta_coil-1,nzeta_coil-1,coeff,coeff)
        if use_f_e:
            avg_Laplace=f_e(np.linalg.norm(laplace_array,axis=2),cws.dS)
        else:
            error2=np.linalg.norm(laplace_array,axis=2)**2
            avg_Laplace=np.sum(error2*cws.dS[:-1,:-1])/((cws.lu-1)*(cws.lv-1))
        coeff_l_extended=np.concatenate((coeff_l,[1.]))# we add a 1 at the end of coeff_l
        #logging.info(str(matH1dot))
        H1dot_error=np.dot(np.dot(coeff_l_extended,matH1dot),coeff_l_extended)
        #logging.info(str(H1dot_error))
        err_tot=Np*(avg_surf_err+lamb*avg_err_j+gamma*avg_Laplace+lamb_H1dot*H1dot_error)
        logging.info('cost : surface error = {:.5E} ,\n j norm = {:.5E} \n Laplace norm = {:.5E}\n H1dot norm ={:.5E}\n  total cost = {:.5E}'.format(avg_surf_err,avg_err_j,avg_Laplace,H1dot_error,err_tot))
        return mc*err_tot#(avg_surf_err,avg_err_j,avg_Laplace,err_tot)
    
    def cost2(coeff_l):
        #to check compatibility
        sol_C,sol_S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(coeff_l,phisize)
        b_array=get_bnorm(path_bnorm,plasma_surf)
        b_vec=(b_array[:-1,:-1]).flatten()
        err_B=np.abs(div_free.compute_normal_B(plasma_surf,(G,I,sol_C,sol_S))-b_vec)
        surf_err=np.mean(err_B*err_B*plasma_surf.dS[:-1,:-1].flatten())
        surf_err=np.mean(err_B*err_B*plasma_surf.dS[:-1,:-1].flatten())
        j=vector_field_on_TS.get_full_j(sol_C,sol_S,div_free.surf.dpsidu,div_free.surf.dpsidv,div_free.surf.dS,div_free.surf.grid,G,I)
        norm_j=np.linalg.norm(j,axis=2)[:-1,:-1]
        err_j=np.mean(norm_j*norm_j*div_free.surf.dS[:-1,:-1])
        coeff=(G,I,sol_C,sol_S)
        laplace_array=avg.f_laplace_optimized(ntheta_coil-1,nzeta_coil-1,coeff,coeff)
        error2=np.linalg.norm(laplace_array,axis=2)**2
        avg_Laplace=np.sum(error2*cws.dS[:-1,:-1])/((cws.lu-1)*(cws.lv-1))
        err_tot=surf_err+lamb2*err_j+gamma*avg_Laplace
        print('cost : surface error = {:.5E} ,\n j norm = {:.5E} \n Laplace norm = {:.5E}\n total cost = {:.5E}'.format(surf_err,err_j,avg_Laplace,err_tot))
comm.Bcast(coeff_l, root=main)

#we start the gradient descent algorithm:
lst_theta=range(lu-1)
lst_zeta=range(lv-1)
step=math.ceil((len(lst_theta))/nproc_u)
blocks=lst_theta[rank*step:(rank+1)*step]
bool_continue=True
if rank==main:
    def grad(coeff_l):
        # we broadcast coeff_l to all
        comm.Bcast(coeff_l, root=main)
        #we compute L
        C,S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(coeff_l,phisize)
        coeff=(G,net_toroidal_current_Amperes,C,S)
        laplace_array=avg.f_laplace_optimized(ntheta_coil-1,nzeta_coil-1,coeff,coeff)
        # we save the temporary result
        f=open(path_output+'.tmp','wb')
        pickle.dump(coeff_l,f)
        f.close()
        # we get back gradient result
        grad_L=np.zeros((lu-1,lv-1,lc,3))
        for k in range(nproc_u):
            grad_L[k*step:(k+1)*step]=comm.recv( source = k)
        if use_f_e:
            laplace_norm=np.linalg.norm(laplace_array,axis=2)
            grad_normL=np.einsum('...kl,...l->...k',grad_L,laplace_array)/laplace_norm[:,:,np.newaxis]
            dcLdc=grad_f_e(laplace_norm,grad_normL,cws.dS)
        else:
        #we compute the gradient of the cost L^2:
            dcLdc=2*np.einsum('ijl,ijkl->k',laplace_array*cws.dS[:-1,:-1,np.newaxis],grad_L)/((cws.lu-1)*(cws.lv-1))#lu x lv x lc x 3
        #we compute the gradient with respect to the regcoil cost
        dcdreg=2*(np.dot(Mat,coeff_l)-RHS)
        H1dot_grad=2*np.dot(coeff_l,matH1dot[:-1,:-1])+2*matH1dot[:-1,-1]
        logging.info('eval grad successfull')
        return mc*Np*(dcdreg+gamma*dcLdc+lamb_H1dot*H1dot_grad)
    
    if False:
        coeff_test=np.random.random((lc))
        grad_th=grad(coeff_test)
        grad_num=np.zeros((lc))
        cost_ref=cost(coeff_test)
        dc=1e-3
        for i in range(lc):
            ncoeff=coeff_test.copy()
            ncoeff[i]+=dc
            grad_num[i]=(cost(ncoeff)-cost_ref)/dc
        bool_continue=False
        logging.info('done')
        comm.Bcast(-1*np.ones(lc), root=main)#signal to stop1e-15
        print(grad_th/grad_num)
        np.testing.assert_almost_equal(grad_th,grad_num)

    res=scipy.optimize.minimize(cost, coeff_l, args=(), method='CG', jac=grad)
    # #print(res)
    # #print(cost(res))
    f=open(path_output,'wb')
    pickle.dump(res,f)
    f.close()
    logging.info('done')
    comm.Bcast(-1*np.ones(lc), root=main)#signal to stop
else:
    flag=0
    while bool_continue:
        comm.Bcast(coeff_l, root=main)
        if np.equal(-1*np.ones(lc),coeff_l).all():
            bool_continue=False
        else:
            C,S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(coeff_l,phisize)
            coeff=(G,net_toroidal_current_Amperes,C,S)
            t1=time.time()
            #partial_grad=avg.grad_1_f_laplace(lu-1,lv-1,coeff,djdc,blocks,lst_zeta)
            #partial_grad+=avg.grad_2_f_laplace(lu-1,lv-1,coeff,djdc,blocks,lst_zeta)
            partial_grad=avg.grad_f_laplace(lu-1,lv-1,coeff,djdc,matrix_gradj,blocks,lst_zeta)
            comm.send(partial_grad, dest = main)
            flag+=1
            if rank==0:
                logging.info('proc {} done task {} in {}'.format(rank,flag,time.time()-t1))


exit()
comm.Disconnect()
