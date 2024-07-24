import numpy as np
import vector_field_on_TS
import pickle
import toroidal_surface
#from mayavi import mlab
import Regcoil_get_matrix_element
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from bnorm import *
logging.basicConfig(level=logging.WARNING)
def Regcoil(G,I,lamb,Phisize,cws,plasma,path_bnorm=None,curpol=1):
    # loading of the surfaces and computing the matrices elements
    (A_B,tensor_j_K),(b_B,tensor_b_K),div_free,cws,plasma_surf=Regcoil_get_matrix_element.Regcoil_get_matrix_element(G,I,Phisize,surfs=(cws,plasma))
    #normalization of the matrices:
    norm_normal_plasma_vec = plasma.dS[:-1,:-1].flatten()
    diag_norm_normal_plasma = np.diag(norm_normal_plasma_vec)
    #b_B_0=b_B.copy()
    #computation of the matrices for the optimization
    if path_bnorm is not None:
        b_array=get_bnorm(path_bnorm,plasma)
        b_vec=curpol*(b_array[:-1,:-1]).flatten()
        b_B-=b_vec
    #np.testing.assert_almost_equal(b_B_0+b_vec,b_B)
    Mat_B=np.dot(A_B.transpose(),np.dot(diag_norm_normal_plasma,A_B))/((plasma_surf.lu-1)*(plasma_surf.lv-1))#A^T S A
    RHS_B=np.dot(b_B.transpose(),np.dot(diag_norm_normal_plasma,A_B))/((plasma_surf.lu-1)*(plasma_surf.lv-1))#b^T S A

    tmp_K=np.reshape(tensor_j_K,(tensor_j_K.shape[0],-1))#without the surface element
    #tensor_j_K_S=np.einsum('ijkl,jk->ijkl', tensor_j_K, cws.dS[:-1,:-1]) #with the surface element
    tensor_j_K_S=tensor_j_K*cws.dS[np.newaxis,:-1,:-1,np.newaxis]# lc x lu x lv x 3
    tmp_K_S=np.reshape(tensor_j_K_S,(tensor_j_K_S.shape[0],-1))#with the surface element
    Mat_K=np.dot(tmp_K_S,tmp_K.transpose())/((cws.lu-1)*(cws.lv-1))#A^T S A
    RHS_K=np.dot(tmp_K_S,tensor_b_K.flatten())/((cws.lu-1)*(cws.lv-1))#b^T S A

    Mat=Mat_B+lamb*Mat_K
    RHS=RHS_B+lamb*RHS_K
    #solution of the quadratic minimization
    Res=np.linalg.solve(Mat,RHS)
    Sol=Res
    #computation of the average surface error
#    sumS_cws=np.sum(cws.dS[:-1,:-1].flatten())
#    sumS_plasma=np.sum(plasma_surf.dS[:-1,:-1].flatten())
    #tmp_B=np.dot(A_B,Sol)-b_B
    surf_err=np.dot(A_B,Sol)-b_B
    avg_surf_err=np.dot(surf_err,np.dot(diag_norm_normal_plasma,surf_err))/((plasma_surf.lu-1)*(plasma_surf.lv-1))
    #j_err=np.dot(tmp_K.transpose(),Sol)-tensor_b_K.flatten()
    #avg_err_j=np.dot(np.dot(Sol,tmp_K_S)-(tensor_b_K*cws.dS[:-1,:-1,np.newaxis]).flatten(),j_err)/((cws.lu-1)*(cws.lv-1))

    j_err2=np.einsum('i...,i->...',tensor_j_K,Res)-tensor_b_K
    avg_err_j=3*np.mean(j_err2**2*cws.dS[:-1,:-1,np.newaxis])# 3 times because we take a mean of jx^2+jy^2+jz^2
    #j_err=np.linalg.norm(np.dot(Mat_K,Sol)-RHS_K)
    ##computation of the surface j norm
    #tmp_K=np.dot(A_K,Sol)-b_K
    #j_norm=np.dot(tmp_K.transpose(),tmp_K)
    print('optimization successfull,\n surface error = {:.8E} ,\n avg surface j norm = {:.8E}'.format(avg_surf_err*plasma_surf.Np,avg_err_j*cws.Np))
    print('erreur max : {:.8E}, jmax : {:.8E}'.format(np.max(np.abs(surf_err)),np.max(np.linalg.norm(j_err2,axis=2))))
    #tmp_B=b_B
    #surf_err=np.dot(tmp_B.transpose(),tmp_B)
    ##computation of the surface j norm
    #tmp_K=b_K
    #j_norm=np.dot(tmp_K.transpose(),tmp_K)
    #logging.warning('surface error without action = {}(T.m)**2 ,\n avg surface j norm = {}(MA)**2'.format(surf_err,j_norm*1e-12))
    #np.testing.assert_almost_equal(b_B_0+b_vec,b_B)
    return Sol,div_free

def plot_2d_Regcoil(sol,ax1,ax2,Phisize,plasma_surf,div_free,G,I,path_bnorm=None):
    
    #we plot the error on B :
    sol_C,sol_S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(sol,Phisize)
    
    if path_bnorm is not None:
        b_array=get_bnorm(path_bnorm,plasma)
        b_vec=(b_array[:-1,:-1]).flatten()
        err_B=np.abs(div_free.compute_normal_B(plasma_surf,(G,I,sol_C,sol_S))-b_vec)
    else:
        err_B=np.abs(div_free.compute_normal_B(plasma_surf,(G,I,sol_C,sol_S)))
    surf_err=np.mean(err_B*err_B*plasma_surf.dS[:-1,:-1].flatten())

    j=vector_field_on_TS.get_full_j(sol_C,sol_S,div_free.surf.dpsidu,div_free.surf.dpsidv,div_free.surf.dS,div_free.surf.grid,G,I)
    norm_j=np.linalg.norm(j,axis=2)[:-1,:-1]
    err_j=np.mean(norm_j*norm_j*div_free.surf.dS[:-1,:-1])

    logging.warning('plotting,\n avg surface error = {:.5E}[T^2.m^2] ,\n avg surface j norm = {:.5E}A^2/m^2'.format(surf_err*plasma_surf.Np,err_j*cws.Np))
    # plot
    im1 = ax1.imshow(np.reshape(err_B,(plasma_surf.lu-1,plasma_surf.lv-1)),cmap='jet',origin='lower')
    plt.colorbar(im1,ax=ax1)

    im2 = ax2.imshow(norm_j,cmap='jet',origin='lower')
    plt.colorbar(im2,ax=ax2)
def plot_perf_Regcoil(G,I,Phisize,path_csw,path_plasma,axs):
    ax1,ax2=axs
    lst_lamb=[10**(-i) for i in range(5,22)]
    lst_lamb.append(0)
    maxj=[]
    maxBn=[]
    errB=[]
    errj=[]
    for lamb in tqdm(lst_lamb):
        sol,plasma_surf,div_free=Regcoil(G,I,lamb,Phisize,path_csw,path_plasma)
        sol_C,sol_S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(sol,Phisize)
        err_B=np.abs(div_free.compute_normal_B(plasma_surf,(G,I,sol_C,sol_S)))
        surf_err=np.mean(err_B*err_B*plasma_surf.dS[:-1,:-1].flatten())

        j=vector_field_on_TS.get_full_j(sol_C,sol_S,div_free.surf.dpsidu,div_free.surf.dpsidv,div_free.surf.dS,div_free.surf.grid,G,I)
        norm_j=np.linalg.norm(j,axis=2)[:-1,:-1]
        err_j=np.mean(norm_j*norm_j*div_free.surf.dS[:-1,:-1])
        maxj.append(norm_j.max())
        maxBn.append(err_B.max())
        errB.append(surf_err)
        errj.append(err_j)
    print(lst_lamb)
    print(errB)
    print(errj)
    print(maxBn)
    print(maxj)
    ax1.plot(errj,errB)
    ax2.plot(maxj,maxBn)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")


if __name__=='__main__':
    ntheta_plasma = 64+1
    ntheta_coil   = 64+1
    nzeta_plasma = 64+1
    nzeta_coil   = 64+1
    mpol_coil  = 8
    ntor_coil  = 8
    Np=3
    #net_poloidal_current_Amperes = 3.6*1e6#1.4#/(2*np.pi)
    #net_poloidal_current_Amperes = 1e6/Np#11884578.094260072#/np.pi
    net_poloidal_current_Amperes = 11884578.094260072/Np
    net_toroidal_current_Amperes = 0.#0.3#/(2*np.pi)
    lamb1=1.2e-14
    lamb2=2.5e-16
    lamb3=5.1e-19
    lamb=lamb3
    curpol=4.9782004309255496
    Phisize=(ntor_coil,mpol_coil)
    G,I=net_poloidal_current_Amperes,net_toroidal_current_Amperes
    path_plasma='code/li383/plasma_surf.txt'
    path_cws='code/li383/cws.txt'
    path_bnorm='code/li383/bnorm.txt'
    #path_bnorm=None
    #plasma_surf=toroidal_surface.Toroidal_surface(radii=(3,1),nbpts=(ntheta_coil,nzeta_coil),Np=Np)
    plasma_surf=toroidal_surface.Toroidal_surface(W7x_pathfile=path_plasma,nbpts=(ntheta_coil,nzeta_coil),Np=3)
    total_plasma_surf=np.mean(plasma_surf.dS[:-1,:-1])
    cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(ntheta_plasma,nzeta_plasma),Np=3)
    #cws=toroidal_surface.Toroidal_surface(radii=(3,1.7),nbpts=(ntheta_plasma,nzeta_plasma),Np=Np)
    total_cws_surf=np.mean(cws.dS[:-1,:-1])
    print('plasma surf = {}, cws surf = {}'.format(total_plasma_surf*3,total_cws_surf*3))
    div_free=vector_field_on_TS.Div_free_vector_field_on_TS(cws)
    #sol,div_free=Regcoil(G,I,lamb,Phisize,cws,plasma_surf)
    sol,div_free=Regcoil(G,I,lamb,Phisize,cws,plasma_surf,path_bnorm=path_bnorm,curpol=curpol)
    #from mayavi import mlab
    #plasma_surf.plot_surface()
    #cws.plot_surface()
    #mlab.show()
    #C,S=np.zeros((2,3)),np.zeros((2,3))
    #j=div_free.get_full_j((G,I,C,S))
    # sol1,div_free=Regcoil(G,I,lamb1,Phisize,cws,plasma_surf,path_bnorm=path_bnorm)
    # sol2,div_free=Regcoil(G,I,lamb2,Phisize,cws,plasma_surf)
    # sol3,div_free=Regcoil(G,I,lamb3,Phisize,cws,plasma_surf)
    # #sol3=np.zeros((sol1.shape))
    # fig = plt.figure()
    # ax1 = fig.add_subplot(321)
    # ax2 = fig.add_subplot(322)
    # ax3 = fig.add_subplot(323)
    # ax4 = fig.add_subplot(324)
    # ax5 = fig.add_subplot(325)
    # ax6 = fig.add_subplot(326)
    # #plot_perf_Regcoil(G,I,Phisize,path_csw,path_plasma,(ax1,ax2))
    # plot_2d_Regcoil(sol1,ax2,ax1,Phisize,plasma_surf,div_free,G,I)
    # plot_2d_Regcoil(sol2,ax4,ax3,Phisize,plasma_surf,div_free,G,I)
    # plot_2d_Regcoil(sol3,ax6,ax5,Phisize,plasma_surf,div_free,G,I)
    # plt.show()
