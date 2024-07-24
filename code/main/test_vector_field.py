import numpy as np
from vector_field_on_TS import *
import toroidal_surface
import unittest
import scipy.constants
import logging
#logging.basicConfig(level=logging.INFO)
#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax1 = fig.add_subplot(221, projection='3d')
#ax2 = fig.add_subplot(222, projection='3d')
#ax3 = fig.add_subplot(223, projection='3d')
#ax4 = fig.add_subplot(224)#, projection='3d')
class Test_vector_field_on_TS(unittest.TestCase):
    
    @unittest.SkipTest
    def test_magnetic_field_interface_W7X(self):
        """check the jump condition on the surface"""
        epsilon=5e-3
        lu,lv=1000,1000
        m,n=4,3
        G,I,phisize=1,0.75, (m,n)
        l=2* ((m-1)*(2*n+1)+n)
        X0=np.random.random(l)
        #X0=np.zeros(l)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(X0,phisize)
        coeff=(G,I,C,S)
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(lu,lv))
        #cws=toroidal_surface.Toroidal_surface(radii=(5,1),nbpts=(lu,lv))
        div_free=Div_free_vector_field_on_TS(cws)
        normal=cws.n[:,lu//2,lv//2]
        j=div_free.get_full_j(coeff)

        Pm=div_free.P[lu//2,lv//2]-epsilon*normal #inside points
        Pp=div_free.P[lu//2,lv//2]+epsilon*normal #outside points
        Bm=div_free.compute_B(Pm.reshape((-1,3)),coeff)/scipy.constants.mu_0
        Bp=div_free.compute_B(Pp.reshape((-1,3)),coeff)/scipy.constants.mu_0
        err=np.linalg.norm(np.cross(normal,(Bp-Bm).flatten())-j[lu//2,lv//2])
        print(j[lu//2,lv//2])
        print(np.cross(normal,(Bp-Bm).flatten()))
        np.testing.assert_almost_equal(err,0,decimal=1)
    
    @unittest.SkipTest
    def test_magnetic_field_interface_torus(self):
        """check the jump condition on the surface"""
        epsilon=5e-3
        lu,lv=1000,1000
        m,n=4,3
        G,I,phisize=1,0.75, (m,n)
        l=2* ((m-1)*(2*n+1)+n)
        X0=np.random.random(l)
        #X0=np.zeros(l)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(X0,phisize)
        coeff=(G,I,C,S)
        #cws=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(lu,lv))
        cws=toroidal_surface.Toroidal_surface(radii=(5,1),nbpts=(lu,lv))
        div_free=Div_free_vector_field_on_TS(cws)
        normal=cws.n[:,lu//2,lv//2]
        j=div_free.get_full_j(coeff)

        Pm=div_free.P[lu//2,lv//2]-epsilon*normal #inside points
        Pp=div_free.P[lu//2,lv//2]+epsilon*normal #outside points
        Bm=div_free.compute_B(Pm.reshape((-1,3)),coeff)/scipy.constants.mu_0
        Bp=div_free.compute_B(Pp.reshape((-1,3)),coeff)/scipy.constants.mu_0
        err=np.linalg.norm(np.cross(normal,(Bp-Bm).flatten())-j[lu//2,lv//2])
        #print(j[lu//2,lv//2])
        #print(np.cross(normal,(Bp-Bm).flatten()))
        np.testing.assert_almost_equal(err,0,decimal=1)
    
    @unittest.SkipTest
    def test_magnetic_field_from_flat_torus(self):
        """check the jump condition on the surface"""
        epsilon=1e-3
        lu,lv=1000,1000
        m,n=4,3
        G,I,phisize=1,0.75, (m,n)
        l=2* ((m-1)*(2*n+1)+n)
        X0=np.random.random(l)
        #X0=np.zeros(l)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(X0,phisize)
        coeff=(G,I,C,S)
        cws=toroidal_surface.Toroidal_surface(flat_torus=True,nbpts=(lu,lv))
        div_free=Div_free_vector_field_on_TS(cws)
        normal=np.array([0,0,1])
        j=div_free.get_full_j(coeff)

        Pm=div_free.P[lu//2,lv//2]-epsilon*normal #inside points
        Pp=div_free.P[lu//2,lv//2]+epsilon*normal #outside points
        Bm=div_free.compute_B(Pm.reshape((-1,3)),coeff)/scipy.constants.mu_0
        Bp=div_free.compute_B(Pp.reshape((-1,3)),coeff)/scipy.constants.mu_0
        c=np.cross(normal,(Bp-Bm).flatten())
        np.testing.assert_allclose(c,j[lu//2,lv//2],rtol=2)
    
    @unittest.SkipTest
    def test_Phi_eval(self):
        m,n=17,31
        lu,lv=19,11
        u,v=np.linspace(0,1,lu),np.linspace(0,1,lv)
        ug,vg=np.meshgrid(u,v,indexing='ij')
        C,S=np.random.random((m,2*n+1)),np.random.random((m,2*n+1))
        r=np.zeros((lu,lv))
        for i in range(m):
            for j in range(2*n+1):
                for iu in range(lu):
                    for iv in range(lv):
                        r[iu,iv]+=C[i,j]*np.cos(2*np.pi*(i*u[iu]+(j-n)*v[iv]))+S[i,j]*np.sin(2*np.pi*(i*u[iu]+(j-n)*v[iv]))
        rr=evaluate(C,S,ug,vg)
        np.testing.assert_almost_equal(r,rr)
        #np.testing.assert_almost_equal(r,rrr)
    
    @unittest.SkipTest
    def test_Phi_dPhi(self):
        np.random.seed(seed=5)
        m,n=3,2
        C,S=np.random.random((m,2*n+1)),np.random.random((m,2*n+1))
        Cu,Su,Cv,Sv=dPhi(C,S)
        #precision on u:
        lu,lv=10000,10
        u,v=np.linspace(0,1,lu),np.linspace(0,1,lv)
        ug,vg=np.meshgrid(u,v,indexing='ij')
        dun,dvn=np.gradient(evaluate(C,S,ug,vg),1/(lu-1),1/(lv-1))
        due=evaluate(Cu,Su,ug,vg)
        np.testing.assert_almost_equal(due[1:-1,1:-1],dun[1:-1,1:-1],decimal=5)
        #precision on v:
        lu,lv=10,10000
        u,v=np.linspace(0,1,lu),np.linspace(0,1,lv)
        ug,vg=np.meshgrid(u,v,indexing='ij')
        dun,dvn=np.gradient(evaluate(C,S,ug,vg),1/(lu-1),1/(lv-1))
        dve=evaluate(Cv,Sv,ug,vg)
        np.testing.assert_almost_equal(dve[1:-1,1:-1],dvn[1:-1,1:-1],decimal=5)
    
    @unittest.SkipTest
    def test_compute_B_from_j(self):
        Npos=52
        G,I=8,9
        m,n=7,11
        Np=5
        lu,lv=17,12
        l=2* ((m)*(2*n+1)+n)
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(lu,lv))
        pos=np.random.random((Npos,3))
        lst_coeff=np.random.random(l)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(lst_coeff,(m,n))
        coeff=(G,I,C,S)
        div_free=Div_free_vector_field_on_TS(cws)
        j=get_full_j(C,S,div_free.surf.dpsidu,div_free.surf.dpsidv,div_free.surf.dS,div_free.surf.grid,G,I)
        cwsdS=np.repeat(div_free.surf.dS[:-1,:-1,np.newaxis],3,axis=2)
        B=compute_B_from_j(pos,j[:-1,:-1,:],div_free.P,cwsdS)
        #we compute B 'with hand'
        P=np.array([cws.X,cws.Y,cws.Z])
        BB=np.zeros((Npos,3))
        for o in range(Npos):
            for k in range(lu-1):
                for l in range(lv-1):
                    BB[o,:]+=cws.dS[k,l]*np.cross(j[k,l,:],pos[o,:]-P[:,k,l])/(np.linalg.norm(P[:,k,l]-pos[o,:])**3)
        BB*=scipy.constants.mu_0/(4*np.pi*(lu-1)*(lv-1))
        np.testing.assert_almost_equal(B,BB)
    
    @unittest.SkipTest
    def test_check_div_free(self):
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(129,167))
        div_free=Div_free_vector_field_on_TS(cws)
        m,n=2,3
        G,I=1e6,1e5
        lu,lv=cws.X.shape
        C,S=(np.random.random((m,2*n+1)),np.random.random((m,2*n+1)))
        #C,S=np.zeros((m,2*n+1)),np.zeros((m,2*n+1))
        j=get_full_j(C,S,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid,G,I)
        Cu,Su,Cv,Sv=dPhi(C,S)
        dphi_u=evaluate(Cu,Su,cws.grid[0],cws.grid[1])#lu*lv
        dphi_v=evaluate(Cv,Sv,cws.grid[0],cws.grid[1])#lu*lv
        j_upperu=-dphi_v/cws.dS+G/cws.dS
        j_upperv=dphi_u/cws.dS-I/cws.dS
        # we check the divergence:
        d1,d2=1/(lu-1),1/(lv-1)
        tmp_u,_=np.gradient(j_upperu*cws.dS,d1,d2)
        _,tmp_v=np.gradient(j_upperv*cws.dS,d1,d2)
        tmp_u[0,:],tmp_u[-1,:]=(tmp_u[0,:]+tmp_u[-1,:])/2,(tmp_u[0,:]+tmp_u[-1,:])/2
        tmp_v[:,0],tmp_v[:,-1]=(tmp_v[:,0]+tmp_v[:,-1])/2,(tmp_v[:,0]+tmp_v[:,-1])/2
        np.testing.assert_almost_equal((tmp_u+tmp_v)/cws.dS,np.zeros(cws.X.shape),decimal=1)
        j_push=(cws.dpsidu*j_upperu+cws.dpsidv*j_upperv)
        np.testing.assert_almost_equal(np.moveaxis(j_push,0,2),j)
    
    @unittest.SkipTest
    def test_compute_normal(B):
        plasma_surf=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(23,17))
        cws=plasma_surf.expend_surface(0.05)
        div_free=Div_free_vector_field_on_TS(cws)
        
        G,I=1e6,1e5
        lu,lv=cws.X.shape
        #C,S=(np.random.random((m,2*n+1)),np.random.random((m,2*n+1)))
        m1,n1=2,3
        C1,S1=np.zeros((m1,2*n1+1)),np.zeros((m1,2*n1+1))
        m2,n2=9,6
        C2,S2=np.zeros((m2,2*n2+1)),np.zeros((m2,2*n2+1))
        N1=div_free.compute_normal_B(plasma_surf,(G,I,C1,S1))
        N2=div_free.compute_normal_B(plasma_surf,(G,I,C2,S2))
        np.testing.assert_almost_equal(N1,N2)
    
    @unittest.SkipTest
    def test_get_chi(self):
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(16,16))
        div_free=Div_free_vector_field_on_TS(cws)
        m,n=2,3
        G,I=1e8,0
        lu,lv=cws.X.shape
        #C,S=(np.random.random((m,2*n+1)),np.random.random((m,2*n+1)))
        C,S=np.zeros((m,2*n+1)),np.zeros((m,2*n+1))
        coeff=(G,I,C,S)
        j=div_free.get_full_j(coeff)
        Cu,Su,Cv,Sv=dPhi(C,S)
        dphi_u=evaluate(Cu,Su,cws.grid[0],cws.grid[1])#lu*lv
        dphi_v=evaluate(Cv,Sv,cws.grid[0],cws.grid[1])#lu*lv
        j_upperu=-dphi_v/cws.dS+G/cws.dS
        j_upperv=dphi_u/cws.dS-I/cws.dS
        j_push=(cws.dpsidu*j_upperu+cws.dpsidv*j_upperv)
    #
    @unittest.SkipTest
    def test_compute_j_indep_of_size(self):
        lu,lv=51,121
        luu,lvv=501,1201
        m,n=6,5
        I,G=1,2
        np.random.seed(987)
        #C,S=(np.random.random((m,2*n+1)),np.random.random((m,2*n+1)))
        l=2* (m*(2*n+1)+n)
        #np.random.seed(213)
        lst_coeff=(2*np.random.random(l)-1)/(np.arange(1,l+1)**2)
        #lst_coeff=np.zeros(l)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(lst_coeff,(m,n))
        coeff=(G,I,C,S)
        cws1=toroidal_surface.Toroidal_surface(radii=(5,1),nbpts=(lu,lv),Np=1)
        div_free1=Div_free_vector_field_on_TS(cws1)
        j1=get_full_j(C,S,div_free1.surf.dpsidu,div_free1.surf.dpsidv,div_free1.surf.dS,div_free1.surf.grid,G,I)

        cws2=toroidal_surface.Toroidal_surface(radii=(5,1),nbpts=(luu,lvv),Np=1)
        div_free2=Div_free_vector_field_on_TS(cws2)
        j2=get_full_j(C,S,div_free2.surf.dpsidu,div_free2.surf.dpsidv,div_free2.surf.dS,div_free2.surf.grid,G,I)
        j2compare=j2[::10,::10,:]
        #np.testing.assert_almost_equal(j2compare,j1)
        import matplotlib.pyplot as plt
        tmp=(j2compare-j1)/j1
        plt.subplot(431)
        plt.imshow(tmp[:,:,0])
        plt.subplot(432)
        plt.imshow(j1[:,:,0])
        plt.subplot(433)
        plt.imshow(j2[:,:,0])
        plt.subplot(434)
        plt.imshow(tmp[:,:,1])
        plt.subplot(435)
        plt.imshow(j1[:,:,1])
        plt.subplot(436)
        plt.imshow(j2[:,:,1])
        plt.subplot(437)
        plt.imshow(tmp[:,:,2])
        plt.subplot(438)
        plt.imshow(j1[:,:,2])
        plt.subplot(439)
        plt.imshow(j2[:,:,2])
        plt.subplot(4,3,11)
        plt.imshow(np.linalg.norm(j1,axis=2))
        plt.subplot(4,3,12)
        plt.imshow(np.linalg.norm(j2,axis=2))
        plt.show()
    def test_gradj(self):
        n,m=6,7
        lu,lv=64+1,64+1
        path_cws='code/li383/cws.txt'
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(lu,lv),Np=3)
        div_free=Div_free_vector_field_on_TS(cws)
        phisize=(n,m)
        l=2* (m*(2*n+1)+n)# nb of degree of freedom
        lst_coeff=np.random.random(l)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(lst_coeff,(m,n))
        coeff=(3,4,C,S)
        dj=div_free.get_gradj(coeff)
        j=div_free.get_full_j(coeff)
        dj_num=np.gradient(j,1/(lu-1),1/(lv-1),axis=(0,1))
        #TODO
        #np.testing.assert_almost_equal(np.moveaxis(dj_num[0],2,0),dj[0])





if __name__=='__main__':
    unittest.main()