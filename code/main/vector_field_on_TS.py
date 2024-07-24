import numpy as np
import toroidal_surface
import logging
import scipy.constants
import mytimer
import numba
from numba import prange
from tqdm import tqdm
class Div_free_vector_field_on_TS():

    def __init__(self,surf):
        logging.debug('initialization of div_free_vector_field_on_TS')
        self.surf=surf # the toroidal surface
        self.lu,self.lv=surf.X.shape# we get the number of poloidal / toroidal components.
        self.P=np.zeros((self.lu-1,self.lv-1,3))
        self.P[:,:,0]=self.surf.X[:-1,:-1]
        self.P[:,:,1]=self.surf.Y[:-1,:-1]
        self.P[:,:,2]=self.surf.Z[:-1,:-1]
        #self.dim=1+(self.lv-1)*(self.lu-1)#max dimension of our space
        #2 for du# and dv# -1 for avg \Phi and (self.lv-1)*(self.lu-1) for Phi (in real Fourier representation)

    def compute_B(self,pos,coeff):
        """compute the magnetic field in pos N*3 impose by the magnetic field on self.surf and its symetrical part defined by coeff"""
        Np=self.surf.Np
        rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
        lst_pos=[pos]
        for i in range(Np-1):
            lst_pos.append(np.transpose(np.tensordot(rot,lst_pos[i],axes = (1,1))))
        #we compute j once and for all
        G,I,C,S=coeff
        j=get_full_j(C,S,self.surf.dpsidu,self.surf.dpsidv,self.surf.dS,self.surf.grid,G,I)
        aresrot=np.array([self.compute_B_1sec(lst_pos[i],coeff,j=j) for i in range(Np)])# the magnetic field 'twisted'
        ares=np.array([np.tensordot(np.linalg.matrix_power(rot,i),aresrot[i],axes = (0,1)) for i in range(Np)]) # we 'twist' it back recall that rot^T=rot^{-1}
        return np.sum(ares,axis=0)#/self.Np?

    def compute_B_1sec(self,pos,coeff,j=None):
        """compute the magnetic field in pos N*3 impose by the magnetic field on self.surf defined by coeff"""
        #we compute j
        if j is None:
            G,I,C,S=coeff
            j=get_full_j(C,S,self.surf.dpsidu,self.surf.dpsidv,self.surf.dS,self.surf.grid,G,I)
        #we compute B from j
        dS=np.repeat(self.surf.dS[:-1,:-1,np.newaxis],3,axis=2)
        return compute_B_from_j(pos,j[:-1,:-1,:],self.P,dS)

    def compute_normal_B(self,plasma_surf,coeff):
        """given coeff which parametrize the generated magnetic field, compute the tensor B scal n on plasma surf"""
        pos=np.zeros((len(plasma_surf.X[:-1,:-1].flatten()),3))
        pos[:,0]=plasma_surf.X[:-1,:-1].flatten()
        pos[:,1]=plasma_surf.Y[:-1,:-1].flatten()
        pos[:,2]=plasma_surf.Z[:-1,:-1].flatten()
        assert(len(coeff)==4)#G,I,S,C
        B=self.compute_B(pos,coeff)
        flatn_p=np.reshape(-1*plasma_surf.n[:,:-1,:-1],(3,-1))# -1 because we want the outward normal
        B_T=np.sum(B*flatn_p,axis=0)#( B scalar n_p)
        return B_T

    def array_coeff_to_CS(array_coeff,Phisize):
        """compute C and S from array_coeff"""
        n,m=Phisize
        l=len(array_coeff)
        assert(l==2* (m*(2*n+1)+n))
        a1,a2=array_coeff[:l//2],array_coeff[l//2:]# a1 for S, a2 for C
        C,S=np.zeros((m+1,2*n+1)),np.zeros((m+1,2*n+1))
        S[0,n+1:]=a1[:n]#we start at m=0 n>=1
        C[0,n+1:]=a2[:n]

        S[1:,:]=np.reshape(a1[n:],(m,2*n+1))
        C[1:,:]=np.reshape(a2[n:],(m,2*n+1))
        return (C,S)

    def compute_matrix_cost(self,plasma_surf,G,I,Phisize):
        """compute the matrix associated to the linear a
        Phisize : m,n
        """
        from tqdm import tqdm
        n,m=Phisize
        l=2* (m*(2*n+1)+n)# nb of degree of freedom
        logging.info('Computation of the matrix elements for Regcoil. Plasma size {}x{}, cws size : {}x{}, nb Fourier components {} (m={},n={})'.format(plasma_surf.X.shape[0],plasma_surf.X.shape[1],self.surf.X.shape[0],self.surf.X.shape[1],l,m,n))
        A=np.zeros((plasma_surf.dim,l))
        CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(np.zeros(l),Phisize)
        b=self.compute_normal_B(plasma_surf,(G,I,CC,CS))
        for k in tqdm(range(l)):
            logging.debug('computation of A : {}/{})'.format(k,l))
            zk=np.zeros(l)
            zk[k]=1
            CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,Phisize)
            A[:,k]=self.compute_normal_B(plasma_surf,(0,0,CC,CS))
        return (A,-b)

    def get_Chi_K(self,phisize,G,I):
        from tqdm import tqdm
        m,n=phisize
        logging.info('Computation of Chi_K. cws size : {}x{}, Fourier components m={}, n={}'.format(self.surf.X.shape[0],self.surf.X.shape[1],m,n))
        flat_dS=self.surf.dS[:-1,:-1].flatten()
        l=2* (m*(2*n+1)+n)# nb of degree of freedom
        zk=np.zeros(l)
        C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,phisize)
        j0=self.get_full_j((G,I,C,S))
        tensor_j=np.zeros((l,self.lu,self.lv,3))
        for k in tqdm(range(l)):
            logging.debug('computation of Chi_K : {}/{})'.format(k,l))
            zk=np.zeros(l)
            zk[k]=1
            C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,phisize)
            tensor_j[k]=self.get_full_j((0,0,C,S))
        #A_KT=np.reshape(aux,(l,3*(self.lu-1)*(self.lv-1)))
        #b_K=np.reshape(j0,(3*(self.lu-1)*(self.lv-1)))
        return tensor_j[:,:-1,:-1,:],-1*j0[:-1,:-1,:]
    def get_full_j(self,coeff):
        G,I,C,S=coeff
        return get_full_j(C,S,self.surf.dpsidu,self.surf.dpsidv,self.surf.dS,self.surf.grid,G,I)
    def get_gradj(self,coeff):
        dS=self.surf.dS
        gradj=np.zeros((2,3,self.lu,self.lv))
        G,I,C,S=coeff
        Cu,Su,Cv,Sv=dPhi(C,S)
        dphi_u=evaluate(Cu,Su,self.surf.grid[0],self.surf.grid[1])
        dphi_v=evaluate(Cv,Sv,self.surf.grid[0],self.surf.grid[1])#lu*lv
        #second derivative
        Cuu,Suu,Cuv,Suv=dPhi(Cu,Su)
        Cvu,Svu,Cvv,Svv=dPhi(Cv,Sv)
        np.testing.assert_almost_equal(Cuv,Cvu)
        np.testing.assert_almost_equal(Suv,Svu)
        dphi_uu=evaluate(Cuu,Suu,self.surf.grid[0],self.surf.grid[1])
        dphi_uv=evaluate(Cuv,Suv,self.surf.grid[0],self.surf.grid[1])
        dphi_vv=evaluate(Cvv,Svv,self.surf.grid[0],self.surf.grid[1])
        #we start the derivation
        gradj[0,:,:,:]=(dphi_uu[np.newaxis,:,:]*self.surf.dpsidv-dphi_uv[np.newaxis,:,:]*self.surf.dpsidu)/dS#derivative of phi
        gradj[0,:,:,:]+=(dphi_u[np.newaxis,:,:]*self.surf.dpsi_uv-dphi_v[np.newaxis,:,:]*self.surf.dpsi_uu)/dS#derivative of psi
        gradj[0,:,:,:]+=-1*(dphi_u*self.surf.dpsidv-dphi_v*self.surf.dpsidu)*self.surf.dS_u/(dS**2)#derivative of dS
        gradj[1,:,:,:]=(dphi_uv[np.newaxis,:,:]*self.surf.dpsidv-dphi_vv[np.newaxis,:,:]*self.surf.dpsidu)/dS#derivative of phi
        gradj[1,:,:,:]+=(dphi_u[np.newaxis,:,:]*self.surf.dpsi_vv-dphi_v[np.newaxis,:,:]*self.surf.dpsi_uv)/dS#derivative of psi
        gradj[1,:,:,:]+=-1*(dphi_u*self.surf.dpsidv-dphi_v*self.surf.dpsidu)*self.surf.dS_v/(dS**2)#derivative of dS
        # The multi-valued part
        gradj[0,:,:,:]+=(G*self.surf.dpsi_uu-I*self.surf.dpsi_uv)/dS#derivative of psi
        gradj[0,:,:,:]+=-1*(G*self.surf.dpsidu-I*self.surf.dpsidv)*self.surf.dS_u/(dS**2)#derivative of dS
        gradj[1,:,:,:]+=(G*self.surf.dpsi_uv-I*self.surf.dpsi_vv)/dS#derivative of psi
        gradj[1,:,:,:]+=-1*(G*self.surf.dpsidu-I*self.surf.dpsidv)*self.surf.dS_v/(dS**2)#derivative of dS
        return np.moveaxis(gradj,1,3)
    def get_matrix_gradj(self,phisize,G,I):
        m,n=phisize
        lc=2* (m*(2*n+1)+n)# nb of degree of freedom
        matrix_gradj=np.zeros((lc+1,2,self.lu,self.lv,3))
        for k in tqdm(range(lc)):
            zk=np.zeros(lc)
            zk[k]=1
            CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,phisize)
            coeff=(0,0,CC,CS)
            matrix_gradj[k]=self.get_gradj(coeff)
        coeff=(G,I,0*CC,0*CS)
        matrix_gradj[lc]=self.get_gradj(coeff)
        return matrix_gradj

        
def get_full_j(C,S,dpsidu,dpsidv,dS,grids,G,I):
    return get_j(C,S,dpsidu,dpsidv,dS,grids)+np.moveaxis((G*dpsidu-I*dpsidv)/dS,0,2)

@numba.njit(parallel=True,cache=True)
def compute_B_from_j(pos,j,P,cwsdS):
    """compute B on pos (Nx3 array) on 1 section of the cws"""
    lum,lvm,_=j.shape#lu-1,lv-1
    Npos=pos.shape[0]
    #cross product
    B=np.zeros((Npos,3))
    for o in prange(Npos):
        T=pos[o,:]-P#/(np.linalg.norm(P[:,k,l]-pos[:,o])**3)
        normT=np.sum(T*T,axis=2)
        coeffT=1/(normT*np.sqrt(normT))
        T[:,:,0]=T[:,:,0]*coeffT
        T[:,:,1]=T[:,:,1]*coeffT
        T[:,:,2]=T[:,:,2]*coeffT
        B[o,:]+=np.sum(np.sum(cwsdS*np.cross(j,T),axis=0),axis=0)
#        for k in range(lum):
#            for l in range(lvm):
#                B[o,:]+=cwsdS[k,l]*np.cross(j[k,l,:],P[k,l,:]-pos[o,:])#/(np.linalg.norm(P[:,k,l]-pos[:,o])**3)
    return B*scipy.constants.mu_0/(4*np.pi*lum*lvm)

def get_djdc_naif(Phisize,dpsidu,dpsidv,dS,grids):
    """return the lx luxlv x3 array dj/dc"""
    n,m=Phisize
    l=2* (m*(2*n+1)+n)# nb of degree of freedom
    lu,lv=grids[0].shape
    djdc=np.zeros((l,lu,lv,3))
    for k in tqdm(range(l)):
        zk=np.zeros(l)
        zk[k]=1
        CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,Phisize)
        djdc[k]=get_full_j(CC,CS,dpsidu,dpsidv,dS,grids,0,0)
    return djdc
def get_norm_H1p(Phisize,g_upper,dS,grids,matrix_gradj):
    """return the (l+1) x x lx array <gradj_i,gradj_j>, ='"""
    n,m=Phisize
    l=2* (m*(2*n+1)+n)# nb of degree of freedom
    mat_H1p=np.zeros((l+1,l+1))
    dj=np.swapaxes(matrix_gradj,0,1)
    for i in tqdm(range(l+1)):
        for k in range(l+1):
            # a lu x lv x 3 tensor
            dji_scalar_djk=dj[0,i]*dj[0,k]*g_upper[0,0,:,:,np.newaxis]+dj[1,i]*dj[0,k]*g_upper[1,0,:,:,np.newaxis]+dj[0,i]*dj[1,k]*g_upper[0,1,:,:,np.newaxis]+dj[1,i]*dj[1,k]*g_upper[1,1,:,:,np.newaxis]
            # we sum over x,y,z
            tmp=np.sum(dji_scalar_djk,axis=2)#<dj_i scal dj_k
            mat_H1p[i,k]=np.mean(tmp[:-1,:-1]*dS[:-1,:-1])
    return mat_H1p
def get_norm_H1p_naif(Phisize,g_upper,dS,grids,djdc,j_GI):
    """return the (l+1) x x lx array <gradj_i,gradj_j>, j_GI is the current from the 'multivalued-part'"""
    n,m=Phisize
    l=2* (m*(2*n+1)+n)# nb of degree of freedom
    lu,lv=grids[0].shape
    mat_H1p=np.zeros((l+1,l+1))
    dj=np.zeros((2,l+1,lu,lv,3))
    for k in tqdm(range(l)):
        for d in range(3):
            grad_tmp=np.gradient(djdc[k,:,:,d],1/(lu-1),1/(lv-1))
            dj[0,k,:,:,d]=grad_tmp[0]
            dj[1,k,:,:,d]=grad_tmp[1]
    for d in range(3):
        grad_tmp=np.gradient(j_GI[:,:,d],1/(lu-1),1/(lv-1))
        dj[0,l,:,:,d]=grad_tmp[0]
        dj[1,l,:,:,d]=grad_tmp[1]
    for i in tqdm(range(l+1)):
        for k in range(l+1):
            # a lu x lv x 3 tensor
            dji_scalar_djk=dj[0,i]*dj[0,k]*g_upper[0,0,:,:,np.newaxis]+dj[1,i]*dj[0,k]*g_upper[1,0,:,:,np.newaxis]+dj[0,i]*dj[1,k]*g_upper[0,1,:,:,np.newaxis]+dj[1,i]*dj[1,k]*g_upper[1,1,:,:,np.newaxis]
            # we sum over x,y,z
            tmp=np.sum(dji_scalar_djk,axis=2)#<dj_i scal dj_k
            mat_H1p[i,k]=np.mean(tmp[:-1,:-1]*dS[:-1,:-1])
    return mat_H1p
def get_djdc_partial(Phisize,dpsidu,dpsidv,dS,grids,lst_k):
    """return the len(lst_k) x lu x lv x3 array dj/dc"""
    n,m=Phisize
    l=2* (m*(2*n+1)+n)# nb of degree of freedom
    lu,lv=grids[0].shape
    djdc=np.zeros((len(lst_k),lu,lv,3))
    for flag in range(len(lst_k)):
        k=lst_k[flag]
        zk=np.zeros(l)
        zk[k]=1
        CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,Phisize)
        djdc[flag]=get_full_j(CC,CS,dpsidu,dpsidv,dS,grids,0,0)
    return djdc
def dPhi(C,S):
    """return the fourier component of the gradient of C,S"""
    m,N=C.shape
    assert(N%2==1)
    n=N//2
    #we start with d/du
    mtensor=np.repeat( (2*np.pi*np.arange(m))[:,np.newaxis],2*n+1,axis=1)
    Su=-mtensor*C#dcos=-sin
    Cu=mtensor*S
    #d/dv
    ntensor=np.repeat( (2*np.pi*np.arange(-n,n+1))[np.newaxis,:],m,axis=0)# m*(2n+1)
    Sv=-ntensor*C#dcos=-sin
    Cv=ntensor*S
    return Cu,Su,Cv,Sv

@numba.njit(cache=True,parallel=True)
def evaluate_para(C,S,U,V):
    """compute the function given by C, S on the points given by the arrays U and V"""
    m,N=C.shape
    n=N//2
    res=np.zeros(U.shape)
    for i in prange(m):
        for j in prange(2*n+1):
            res+= C[i,j]*np.cos(2*np.pi*(i*U+(j-n)*V))+S[i,j]*np.sin(2*np.pi*(i*U+(j-n)*V))
    return res
@numba.njit(cache=True)
def evaluate(C,S,U,V):
    """compute the function given by C, S on the points given by the arrays U and V"""
    m,N=C.shape
    n=N//2
    res=np.zeros(U.shape)
    for i in range(m):
        for j in range(2*n+1):
            if C[i,j]!=0 or S[i,j]!=0:
                res+= C[i,j]*np.cos(2*np.pi*(i*U+(j-n)*V))+S[i,j]*np.sin(2*np.pi*(i*U+(j-n)*V))
    return res

@numba.njit(parallel=True)
def evaluate_double_para(C1,S1,C2,S2,U,V):
    """compute the function given by C1, S1 and C2,S2 on the points given by the arrays U and V"""
    m,N=C1.shape
    n=N//2
    res1=np.zeros(U.shape)
    res2=np.zeros(U.shape)
    for i in range(m):
        for j in range(2*n+1):
            cos=np.cos(2*np.pi*(i*U+(j-n)*V))
            sin=np.sin(2*np.pi*(i*U+(j-n)*V))
            res1+= C1[i,j]*cos+S1[i,j]*sin
            res2+= C2[i,j]*cos+S2[i,j]*sin
    return res1,res2
def get_j(C,S,dpsidu,dpsidv,dS,grids,para=True):
    # do not use para for C and S with a lot of 0
    Cu,Su,Cv,Sv=dPhi(C,S)
    #dphi_u=evaluate_para(Cu,Su,grids[0],grids[1])
    #dphi_v=evaluate_para(Cv,Sv,grids[0],grids[1])#lu*lv
    if para:
        dphi_u,dphi_v=evaluate_double_para(Cu,Su,Cv,Sv,grids[0],grids[1])
    else :
        dphi_u=evaluate(Cu,Su,grids[0],grids[1])
        dphi_v=evaluate(Cv,Sv,grids[0],grids[1])#lu*lv
    #we duplicate 3 times on the first axis to do the tensor multiplication
    lu,lv=grids[0].shape
    j_phi=np.zeros((lu,lv,3))
    for k in range(3):
        j_phi[:,:,k]=(dphi_u*dpsidv[k]-dphi_v*dpsidu[k])/dS# lu x lv x 3
    return j_phi
def get_full_j(C,S,dpsidu,dpsidv,dS,grids,G,I):
    return get_j(C,S,dpsidu,dpsidv,dS,grids)+np.moveaxis((G*dpsidu-I*dpsidv)/dS,0,2)

if __name__=='__main__':
    pass
