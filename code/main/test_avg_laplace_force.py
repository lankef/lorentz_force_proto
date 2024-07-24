import numpy as np
from avg_laplace_force import *
import toroidal_surface
import unittest
import vector_field_on_TS
import scipy.constants

class Test_avg_laplace_force(unittest.TestCase):
    def test_pi_x(self):
        lu,lv=32,35
        cws=toroidal_surface.Toroidal_surface(flat_torus=True,nbpts=(lu,lv))
        j=np.array([0.3,0.7,3.5])
        pj=np.array([0.3,0.7,0])
        a_j=np.tile(j,(lu,lv,1))
        a_pj=np.tile(pj,(lu,lv,1))
        avg=Avg_laplace_force(cws)
        res=avg.pi_x(a_j)
        np.testing.assert_almost_equal(a_pj,res)
    def test_cartesian_to_torus(self):
        lu,lv=32,35
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(lu,lv))
        j=np.random.random((lu,lv,2))
        j_pushed=(cws.dpsidu*j[:,:,0]+cws.dpsidv*j[:,:,1])
        avg=Avg_laplace_force(cws)
        j_back=avg.cartesian_to_torus(np.moveaxis(j_pushed,0,2))
        np.testing.assert_almost_equal(j_back,j)

    def test_div_flat(self):
        lu,lv=500,510
        cws=toroidal_surface.Toroidal_surface(flat_torus=True,nbpts=(lu,lv))
        ugrid,vgrid=cws.grid
        j=np.zeros((lu,lv,2))
        j[:,:,0]=np.cos(2*np.pi*ugrid)*np.cos(2*np.pi*vgrid)
        j[:,:,1]=np.cos(2*np.pi*ugrid)+np.sin(2*np.pi*vgrid)
        divj=-2*np.pi*np.sin(2*np.pi*ugrid)*np.cos(2*np.pi*vgrid)+2*np.pi*np.cos(2*np.pi*vgrid)
        avg=Avg_laplace_force(cws)
        res=avg.div(j)
        np.testing.assert_almost_equal(res,divj,decimal=1)
    def test_grad(self):
        lu,lv=11+1,13+1
        path_cws='code/li383/cws.txt'
        m,n=2,2
        Phisize=(m,n)
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(lu,lv),Np=3)
        djdc=get_djdc_naif(Phisize,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid)
        I,G=1e6,2e5
        np.random.seed(987)
        l=2* (m*(2*n+1)+n)
        lst_coeff=1e3*(2*np.random.random(l)-1)/(np.arange(1,l+1)**2)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(lst_coeff,(m,n))
        coeff=(G,I,C,S)
        avg=Avg_laplace_force(cws)
        lst_theta=range(lu-1)
        lst_zeta=range(lv-1)
        
        matrix_gradj=avg.get_matrix_gradj(Phisize,G,I)

        dc=1e2*(2*np.random.random(l)-1)/(np.arange(1,l+1)**2)
        dC,dS=Div_free_vector_field_on_TS.array_coeff_to_CS(dc,Phisize)
        #We test grad_1
        grad_1=avg.grad_1_f_laplace(lu-1,lv-1,coeff,djdc,lst_theta,lst_zeta)
        F_c=avg.f_laplace_optimized(lu-1,lv-1,coeff,coeff,lst_theta,lst_zeta)
        F_cpdc=avg.f_laplace_optimized(lu-1,lv-1,(G,I,C+dC,S+dS),coeff,lst_theta,lst_zeta)
        deltaF_1=F_cpdc-F_c
        deltaF_from_grad_1=np.einsum('ijkl,k->ijl',grad_1,dc)
        np.testing.assert_almost_equal(deltaF_1,deltaF_from_grad_1)
        #We test grad_2
        grad_2=avg.grad_2_f_laplace(lu-1,lv-1,coeff,djdc,matrix_gradj,lst_theta,lst_zeta)
        F_c=avg.f_laplace_optimized(lu-1,lv-1,coeff,coeff,lst_theta,lst_zeta)
        F2_cpdc=avg.f_laplace_optimized(lu-1,lv-1,coeff,(G,I,C+dC,S+dS),lst_theta,lst_zeta)
        deltaF_2=F2_cpdc-F_c
        deltaF_from_grad_2=np.einsum('ijkl,k->ijl',grad_2,dc)
        np.testing.assert_almost_equal(deltaF_2,deltaF_from_grad_2)
        #total grad:
        grad=avg.grad_f_laplace(lu-1,lv-1,coeff,djdc,matrix_gradj,lst_theta,lst_zeta)
        np.testing.assert_almost_equal(grad,grad_1+grad_2)
        #print(grad)
    def test_H1_norm(self):
        lu,lv=250+1,250+1
        path_cws='code/li383/cws.txt'
        m,n=2,2
        Phisize=(m,n)
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(lu,lv),Np=3)
        div_free=Div_free_vector_field_on_TS(cws)
        djdc=get_djdc_naif(Phisize,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid)
        G,I=1e6,2e5
        matrix_gradj=div_free.get_matrix_gradj(Phisize,G,I)
        np.random.seed(987)
        l=2* (m*(2*n+1)+n)
        lst_coeff=1e3*(2*np.random.random(l)-1)/(np.arange(1,l+1)**2)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(lst_coeff,(m,n))
        C0,S0=Div_free_vector_field_on_TS.array_coeff_to_CS(np.zeros(l),(m,n))
        j_GI=div_free.get_full_j((G,I,C0,S0))
        mat=get_norm_H1p_naif(Phisize,cws.g_upper,cws.dS,cws.grid,djdc,j_GI)
        mat2=get_norm_H1p(Phisize,cws.g_upper,cws.dS,cws.grid,matrix_gradj)
        np.testing.assert_allclose(mat[:-1,:-1],mat2[:-1,:-1],atol=35)








if __name__=='__main__':
    unittest.main()