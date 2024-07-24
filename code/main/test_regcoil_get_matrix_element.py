
import numpy as np
import vector_field_on_TS
import toroidal_surface
import unittest
import Regcoil_get_matrix_element

class Test_regcoil_get_matrix_element(unittest.TestCase):
    @unittest.SkipTest
    def test_all_B(self):
        """test if the A given by compute_matrix_cost is right"""
        R,Ntheta,Nphi=6.2,11,7
        r1,r2=0.2,0.7
        m,n=4,3
        G,I,phisize=1e6,1e5,(m,n)
        plasma_surf=toroidal_surface.Toroidal_surface(radii=(R,r1,Ntheta,Nphi))
        cws=toroidal_surface.Toroidal_surface(radii=(R,r2,Ntheta+3,Nphi+1))
        div_free=vector_field_on_TS.Div_free_vector_field_on_TS(cws)
        (A,b)=div_free.compute_matrix_cost(plasma_surf,G,I,phisize)
        #test of A X0
        l=2* ((m-1)*(2*n+1)+n)
        X0=np.random.random(l)
        C,S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(X0,phisize)
        res1=div_free.compute_normal_B(plasma_surf,(0,0,C,S))
        res2=np.dot(A,X0)
        np.testing.assert_almost_equal(res1,res2)
        # test of int_{plasma_surf} B_T
        (A_B,A_K),(b_B,b_K),_,_,_=Regcoil_get_matrix_element.Regcoil_get_matrix_element(G,I,phisize,surfs=(cws,plasma_surf))
        #computation of the matrices for the optimization
        AA_B=np.dot(A_B.transpose(),A_B)#A^T S A
        Ab_B=np.dot(A_B.transpose(),b_B)#A^T S b
        AA_K=np.dot(A_K.transpose(),A_K)
        Ab_K=np.dot(A_K.transpose(),b_K)
        norm1=np.dot(np.dot(X0.transpose(),AA_B),X0)
        norm2=np.mean(res1*res1*plasma_surf.dS[:-1,:-1].flatten())
        np.testing.assert_almost_equal(norm1,norm2)
    #@unittest.SkipTest
    def test_all_K(self):
        lu,lv=61,57
        m,n=4,3
        G,I,phisize=1e6,1e5,(m,n)
        cws=toroidal_surface.Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(lu,lv))
        plasma_surf=toroidal_surface.Toroidal_surface(radii=(2,0.1,5,6))#does not matter
        div_free=vector_field_on_TS.Div_free_vector_field_on_TS(cws)
        #test of A X0
        A,b=div_free.get_Chi_K(phisize,G,I)
        l=2* ((m-1)*(2*n+1)+n)
        X0=np.random.random(l)
        C,S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(X0,phisize)
        j=vector_field_on_TS.get_j(C,S,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid)
        aux=np.zeros(((lu-1)*(lv-1),3))
        for i in range(3):
            aux[:,i]+=j[:-1,:-1,i].flatten()
        res1=aux.flatten()
        res2=np.dot(A,X0)
        np.testing.assert_almost_equal(res1,res2)
        # test of int_{cws} |j| ^2
        (A_B,A_K),(b_B,b_K),_,_,_=Regcoil_get_matrix_element.Regcoil_get_matrix_element(G,I,phisize,surfs=(cws,plasma_surf))
        #computation of the matrices for the optimization
        AA_B=np.dot(A_B.transpose(),A_B)#A^T S A
        Ab_B=np.dot(A_B.transpose(),b_B)#A^T S b
        AA_K=np.dot(A_K.transpose(),A_K)
        Ab_K=np.dot(A_K.transpose(),b_K)
        #testing of the full norm
        normj1=np.dot(np.dot(X0.transpose(),AA_K),X0)
        norm_j=np.linalg.norm(j,axis=2)
        norm_j_flat=norm_j[:-1,:-1].flatten()
        normj2=np.mean(norm_j_flat*norm_j_flat*cws.dS[:-1,:-1].flatten())
        np.testing.assert_allclose(normj1,normj2)
        #    sol_C,sol_S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(sol,Phisize)

        #err_B=np.abs(div_free.compute_normal_B(plasma_surf,(-1*G,-1*I,sol_C,sol_S)))
        #surf_err=np.mean(err_B*err_B*plasma_surf.dS[:-1,:-1].flatten())

        #j=vector_field_on_TS.get_full_j(sol_C,sol_S,div_free.surf.dpsidu,div_free.surf.dpsidv,div_free.surf.dS,div_free.surf.grid,G,I)
        #norm_j=np.linalg.norm(j,axis=2)[:-1,:-1]
        #err_j=np.mean(norm_j*norm_j*div_free.surf.dS[:-1,:-1])

        




if __name__=='__main__':
    unittest.main()