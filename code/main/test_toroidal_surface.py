import numpy as np
#from mayavi import mlab
import time
from toroidal_surface import *
import unittest
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(221)
class Test_toroidal_surface(unittest.TestCase):
    @unittest.SkipTest
    def test_metric_computations(self):
        R=6.1
        r=0.6
        lu,lv=256,207
        radii=(R,r)
        pi=np.pi
        torus=Toroidal_surface(radii=radii,nbpts=(lu,lv))
        def dpsi_u(u,v) : return np.array([-2*pi*r*np.sin(2*pi*u)*np.cos(2*pi*v/5),-2*pi*r*np.sin(2*pi*u)*np.sin(2*pi*v/5),r*2*pi*np.cos(2*pi*u)])
        def dpsi_v(u,v) : return np.array([(R+r*np.cos(2*pi*u))*-2*pi/5 *np.sin(2*pi*v/5),(R+r*np.cos(2*pi*u))*2*pi/5 *np.cos(2*pi*v/5),0])
        def g_lower(u,v):
            tmp=np.array([dpsi_u(u,v),dpsi_v(u,v)])
            return np.matmul(tmp,tmp.transpose())
        def g_upper(u,v):
            return np.linalg.inv(g_lower(u,v))
        U,V=torus.grid
        diff_g_lower=np.zeros(U.shape)
        diff_g_upper=np.zeros(U.shape)
        for i in range(U.shape[0]):
            for j in range(V.shape[1]):
                diff_g_lower[i,j]=np.max(np.abs(g_lower(U[i,j],V[i,j])-torus.g_lower[:,:,i,j]).flatten())/np.max(g_lower(U[i,j],V[i,j]).flatten())
                diff_g_upper[i,j]=np.max(np.abs(g_upper(U[i,j],V[i,j])-torus.g_upper[:,:,i,j]).flatten())/np.max(g_upper(U[i,j],V[i,j]).flatten())
        np.testing.assert_almost_equal(np.max(diff_g_lower.flatten()),0,decimal=4)
        np.testing.assert_almost_equal(np.max(diff_g_upper.flatten()),0,decimal=3)
        #we deal with the surface element
        def dS(u,v):
            return np.sqrt(np.linalg.det(g_lower(u,v)))
        dSn=np.zeros(U.shape)
        for i in range(U.shape[0]):
            for j in range(V.shape[1]):
                dSn[i,j]=dS(U[i,j],V[i,j])
        nS=np.mean(dSn.flatten())
        np.testing.assert_almost_equal(dSn/nS,torus.dS/nS,decimal=3)
    
    def test_compatible_with_Regcoil(self):
        """compare with the regcoil.m config"""
        R=3
        r=1
        ntheta_plasma = 32+1
        nzeta_plasma = 34+1
        lu,lv=ntheta_plasma,nzeta_plasma
        radii=(R,r)
        pi=np.pi
        torus=Toroidal_surface(radii=radii,nbpts=(lu,lv),Np=1)
        def dpsi_u(u,v) : return np.array([-2*pi*r*np.sin(2*pi*u)*np.cos(2*pi*v),-2*pi*r*np.sin(2*pi*u)*np.sin(2*pi*v),r*2*pi*np.cos(2*pi*u)])
        def dpsi_v(u,v) : return np.array([(R+r*np.cos(2*pi*u))*-2*pi *np.sin(2*pi*v),(R+r*np.cos(2*pi*u))*2*pi *np.cos(2*pi*v),0])
        U,V=torus.grid
        dpsi_u_exact=np.zeros((lu,lv,3))
        dpsi_v_exact=np.zeros((lu,lv,3))
        for i in range(U.shape[0]):
            for j in range(V.shape[1]):
                dpsi_u_exact[i,j,:]=dpsi_u(U[i,j],V[i,j])
                dpsi_v_exact[i,j,:]=dpsi_v(U[i,j],V[i,j])
        np.testing.assert_almost_equal(dpsi_u_exact,np.moveaxis(torus.dpsidu,0,2),decimal=1)
        np.testing.assert_almost_equal(dpsi_v_exact,np.moveaxis(torus.dpsidv,0,2),decimal=1)
    def test_second_derivative_psi(self):
        lu,lv=264+1,264+1
        ntheta_coil,nzeta_coil=lu,lv
        path_cws='code/li383/cws.txt'
        cws=Toroidal_surface(W7x_pathfile=path_cws,nbpts=(ntheta_coil,nzeta_coil),Np=3)
        gradu=np.gradient(cws.dpsidu,1/264,1/264,axis=(1,2))
        gradv=np.gradient(cws.dpsidv,1/264,1/264,axis=(1,2))
        np.testing.assert_allclose(cws.dpsi_uv[0],gradu[1][0],atol=1)
        np.testing.assert_allclose(cws.dpsi_uu,gradu[0],atol=5)
        np.testing.assert_allclose(cws.dpsi_uv,gradu[1],atol=5)
        np.testing.assert_allclose(cws.dpsi_vv,gradv[1],atol=5)
    def test_dS_derivative(self):
        lu,lv=264+1,264+1
        ntheta_coil,nzeta_coil=lu,lv
        path_cws='code/li383/cws.txt'
        cws=Toroidal_surface(W7x_pathfile=path_cws,nbpts=(ntheta_coil,nzeta_coil),Np=3)
        graddS=np.gradient(cws.dS,1/264,1/264)
        np.testing.assert_allclose(cws.dS_u[3:,3:],graddS[0][3:,3:],atol=6)
        np.testing.assert_allclose(cws.dS_v,graddS[1],atol=6)
if __name__=='__main__':
    unittest.main()