from vector_field_on_TS import *
# from periodize import *
from tqdm import *
import logging
import mytimer
import numba
from numba import prange
Par=False
fastmathbool=True
class Avg_laplace_force(Div_free_vector_field_on_TS):
    def __init__(self,surf):
        super().__init__(surf)
    
    
    def cartesian_to_torus(self,array):
        """from an array lu x lv x 3 such that array dot normal =0, return lu x lv x 2 corresponding to the u and v component of array
        i.e. this is the inverse operation of the push forward"""
        dpsi=np.array([self.surf.dpsidu,self.surf.dpsidv])
        dpsi_reshaped=np.moveaxis(np.moveaxis(dpsi,0,3),0,3)#lu x lv x 2 x 3
        Y=np.matmul(dpsi_reshaped,array[:,:,:,np.newaxis])[:,:,:,0]#lu x lv x 2 (x 1)
        res=np.einsum('ij...,...j->...i', self.surf.g_upper,Y)
        return res

    def pi_x(self,f):
        """return the projection of the vector field f : lu x lv x 3 on the tangent bundle of self.surf"""
        return  f - np.repeat(np.sum(f*self.surf.normal,axis=2)[:,:,np.newaxis],3,axis=2)*self.surf.normal
    
    def get_pi_x(self):
        """return the projector on the tangent bundle of self.surf : lu x lv x 3x3"""
        pi_x=np.tile(np.array(np.eye(3)),(self.lu,self.lv,1,1))
        pi_x-=np.einsum('...i,...j->...ij',self.surf.normal,self.surf.normal)
        return pi_x
    
    def vector_field_pull_back_old(self,f):
        """inverse of the push forward for vector field, f has to be already tangent to S """
        #highly not optimal
        #TODO improve
        X=np.zeros((self.lu,self.lv,2))
        dpsi=np.array([self.surf.dpsidu,self.surf.dpsidv])
        for u in range(self.lu):
            for v in range(self.lv):
                A=np.dot(dpsi[:,:,u,v],np.transpose(dpsi[:,:,u,v]))
                b=np.dot(dpsi[:,:,u,v],f[u,v,:])
                X[u,v,:]=np.linalg.solve(A,b)
        return X
    def vector_field_pull_back(self,f):
        """inverse of the push forward for vector field, f has to be already tangent to S """
        #still not optimal
        #TODO improve
        X=np.zeros((self.lu,self.lv,2))
        dpsi=np.array([self.surf.dpsidu,self.surf.dpsidv])
        A=np.einsum('ji...,ki...->jk...',dpsi,dpsi)
        b=np.einsum('ij...,...j->i...',dpsi,f)
        AA=np.moveaxis(A,[0,1,2,3],[2,3,0,1])
        bb=np.moveaxis(b,[0,1,2],[2,0,1])
        X=np.linalg.solve(AA,bb)
        return X
    def get_vector_field_pull_back(self):
        """linear operator associated with the inverse of the push forward for vector field """
        dpsi=np.array([self.surf.dpsidu,self.surf.dpsidv])
        A=np.einsum('ji...,ki...->jk...',dpsi,dpsi)
        AA=np.moveaxis(A,[0,1,2,3],[2,3,0,1])
        Ainv=np.linalg.pinv(AA)
        return np.einsum('...ij,jk...->...ik',Ainv,dpsi)
    def div(self,f):
        """return the divergence of f : lu x lv x 2 """
        res=np.zeros((f.shape[0],f.shape[1]))
        for i in range(2):#both coefficients
            aux=self.surf.dS*f[:,:,i]
            d1,d2=(1/(self.lu-1),1/(self.lv-1))
            daux=np.gradient(aux,d1,d2)[i]
            #periodize_function(daux)
            res+=daux/self.surf.dS
        return res
    def div_array(self,f):
        """return the divergence of f : lu x lv x lc x 2 """
        res=np.zeros((f.shape[0],f.shape[1],f.shape[2]))
        for i in range(2):#both coefficients
            aux=self.surf.dS[:,:,np.newaxis]*f[:,:,:,i]
            d1,d2=(1/(self.lu-1),1/(self.lv-1))
            daux=np.gradient(aux,d1,d2,axis = (0,1))[i]
            #periodize_function(daux)
            res+=daux/self.surf.dS[:,:,np.newaxis]
        return res
    def iymx(self,i,j):
        """return 1/|y-x| with y=self.P[i,j]"""
        d2=(self.surf.X[i,j]-self.surf.X)**2+(self.surf.Y[i,j]-self.surf.Y)**2+(self.surf.Z[i,j]-self.surf.Z)**2
        d2[i,j]=1# for the division
        res=1/np.sqrt(d2)
        res[i,j]=0 # by convention
        return res
    #@mytimer.timeit
    def L_eps_optimized(self,coeff,epsilon,lst_theta=None,lst_zeta=None):
        """compute L_eps on a grid"""
        if lst_theta is None:
            ntheta=self.lu-1
            lst_theta=range(ntheta)
        if lst_zeta is None:
            nzeta=self.lv-1
            lst_zeta=range(nzeta)
        Pm,Pp=[],[]
        for theta in lst_theta:
            for zeta in lst_zeta:
                Pm.append(self.P[theta,zeta,:]+epsilon*self.surf.normal[theta,zeta,:]) #inside points
                Pp.append(self.P[theta,zeta,:]-epsilon*self.surf.normal[theta,zeta,:]) #outside points
        Bm=self.compute_B(np.array(Pm),coeff)
        Bp=self.compute_B(np.array(Pp),coeff)
        j1=self.get_full_j(coeff)
        res=np.zeros((len(lst_theta),len(lst_zeta),3))
        flag=0
        for theta in lst_theta:
            for zeta in lst_zeta:
                res[flag//len(lst_zeta),flag%len(lst_zeta),:]=0.5*(np.cross(j1[theta,zeta,:],Bm[:,flag]+Bp[:,flag]))
                flag+=1
        return res#,Bp,Bm,j1
    def L_eps(self,i,j,epsilon,coeff):
        """compute L_eps in self.P[i,j] with a distance"""
        assert i<=self.lu-1 and j<=self.lv-1
        Pm=self.P[i,j,:]+epsilon*self.surf.normal[i,j,:] #inside points
        Pp=self.P[i,j,:]-epsilon*self.surf.normal[i,j,:] #outside points
        Bm=self.compute_B(Pm[np.newaxis,:],coeff)
        Bp=self.compute_B(Pp[np.newaxis,:],coeff)
        j1=self.get_full_j(coeff)[i,j]
        #j1=np.array([0,0,1])
        res=0.5*(np.cross(j1,Bm.flatten())+np.cross(j1,Bp.flatten()))
        return res#,Bp,Bm,j1
    @mytimer.timeit
    def f_laplace(self,i,j,coeff1,coeff2):
        res1=np.zeros(3)
        res2=np.zeros(3)
        res3=np.zeros(3)
        res4=np.zeros(3)
        res5=np.zeros(3)
        res6=np.zeros(3)
        res=np.zeros(3)
        Np=self.surf.Np
        lu,lv=self.lu,self.lv
        d1,d2=(1/(self.lu-1),1/(self.lv-1))
        j2=self.get_full_j(coeff2)
        j1=self.get_full_j(coeff1)[i,j,:]
        rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
        lst_rot=np.array([np.linalg.matrix_power(rot,l) for l in range(Np+1)])
        for l in range(Np):
            #we rotate j1 of rot^l
            rot_l=lst_rot[l]
            j1y=np.matmul(lst_rot[l],j1)
            j1ya=np.tile(j1y,(lu,lv,1))# lu x lv x 3
            newY=np.dot(rot_l,np.array([self.surf.X[i,j],self.surf.Y[i,j],self.surf.Z[i,j]]))
            d2=(newY[0]-self.surf.X)**2+(newY[1]-self.surf.Y)**2+(newY[2]-self.surf.Z)**2
            if l==0:
                d2[i,j]=1# for the division
            elif j==0 and l==1:
                d2[i,-1]=1
                if i==0: d2[-1,-1]=1
            iymx=1/np.sqrt(d2)
            if l==0:
                iymx[i,j]=0#by convention
            pi_xjy=self.pi_x(j1ya)# lu x lv x 3
            pi_xjy_uv=self.vector_field_pull_back(pi_xjy)# lu x lv x 2        
        # -1/(y-x) *(div pi_x j_1(y))*j_2(x)
            P1=-1*(iymx*self.div(pi_xjy_uv))[:,:,np.newaxis]*j2
            print('P1')
            #-1/(y-x) *(pi_x j_1(y) \dot \nabla) j_2(x)
            dj2=self.get_gradj(coeff2)
            dj2x,dj2y,dj2z=dj2[:,:,:,0],dj2[:,:,:,1],dj2[:,:,:,2]
            P2=np.zeros((lu,lv,3))
            for k in range(2):
                P2[:,:,0]+=pi_xjy_uv[:,:,k]*dj2x[k]
                P2[:,:,1]+=pi_xjy_uv[:,:,k]*dj2y[k]
                P2[:,:,2]+=pi_xjy_uv[:,:,k]*dj2z[k]
            P2*=-1*iymx[:,:,np.newaxis]
            print('P2')
            #1/(y-x) \nabla <j1(y) j2(x) >
            f=np.sum(j1ya*j2,axis=2)
            #error is here !!!!!
            #df=0*np.array(np.gradient(f,1/(lu-1),1/(lv-1)))
            
            df=np.sum(j1ya[np.newaxis]*dj2,axis=3)
            #df=
            #dj2du,dj2dv=
            gradf=np.einsum('ij...,j...->i...',self.surf.g_upper,df)
            #periodize_function(gradf[0])
            #periodize_function(gradf[1])
            P3=np.zeros((lu,lv,3))
            for k in range(3):
                P3[:,:,k]+=iymx*(gradf[0,:,:]*self.surf.dpsidu[k]+gradf[1,:,:]*self.surf.dpsidv[k])# lu x lv x 3
            # 1/(y-x) <j1(y) j2(x) > div \pi_x
            self.df=gradf
            self.P3=P3
            print('P3')
            #highly not optimal
            e1=np.tile(np.array([1,0,0]),(lu,lv,1))
            e2=np.tile(np.array([0,1,0]),(lu,lv,1))
            e3=np.tile(np.array([0,0,1]),(lu,lv,1))
            pi_xe1=self.pi_x(e1)# lu x lv x 3
            pi_xe1_uv=self.vector_field_pull_back(pi_xe1)# lu x lv x 2
            pi_xe2=self.pi_x(e2)# lu x lv x 3
            pi_xe2_uv=self.vector_field_pull_back(pi_xe2)# lu x lv x 2
            pi_xe3=self.pi_x(e3)# lu x lv x 3
            pi_xe3_uv=self.vector_field_pull_back(pi_xe3)# lu x lv x 2
            P6=np.zeros((lu,lv,3))
            P6[:,:,0]=iymx*f*self.div(pi_xe1_uv)
            P6[:,:,1]=iymx*f*self.div(pi_xe2_uv)
            P6[:,:,2]=iymx*f*self.div(pi_xe3_uv)
            print('P6')
            # <j1(y) n(x) > <y-x,n(x)> /|y-x|^3 j2(x)
            ymx=np.zeros((3,lu,lv))
            ymx=np.array([newY[0]-self.surf.X, newY[1]-self.surf.Y, newY[2]-self.surf.Z])#y-x
            K=iymx**3*np.sum(ymx*self.surf.n,axis=0)# K=<y-x,n(x)> /|y-x|^3
            c=K*np.sum(j1ya*self.surf.normal,axis=2)
            P4=np.einsum('...,...k->...k',c,j2)
            print('P4')
            #-<j1(y) j2(x)> <y−x,n(x)>/|y-x|^3 n(x)dx
            P5=-1*np.einsum('...,...k->...k',f*K,self.surf.normal)
            print('P5')
            #Integration
            #S=self.surf.dS[:-1,:-1,np.newaxis]*(P1+P2+P3+P4+P5)[:-1,:-1,:]
            S1=self.surf.dS[:-1,:-1,np.newaxis]*P1[:-1,:-1,:]
            S2=self.surf.dS[:-1,:-1,np.newaxis]*P2[:-1,:-1,:]
            S3=self.surf.dS[:-1,:-1,np.newaxis]*P3[:-1,:-1,:]
            S4=self.surf.dS[:-1,:-1,np.newaxis]*P4[:-1,:-1,:]
            S5=self.surf.dS[:-1,:-1,np.newaxis]*P5[:-1,:-1,:]
            S6=self.surf.dS[:-1,:-1,np.newaxis]*P6[:-1,:-1,:]
            self.S3=S3
            self.S6=S6
            res1+=np.matmul(lst_rot[Np-l],1e-7*np.sum(S1,axis=(0,1))/((lu-1)*(lv-1))) # mu0/4pi
            res2+=np.matmul(lst_rot[Np-l],1e-7*np.sum(S2,axis=(0,1))/((lu-1)*(lv-1))) # mu0/4pi
            res3+=np.matmul(lst_rot[Np-l],1e-7*np.sum(S3,axis=(0,1))/((lu-1)*(lv-1))) # mu0/4pi
            res4+=np.matmul(lst_rot[Np-l],1e-7*np.sum(S4,axis=(0,1))/((lu-1)*(lv-1))) # mu0/4pi
            res5+=np.matmul(lst_rot[Np-l],1e-7*np.sum(S5,axis=(0,1))/((lu-1)*(lv-1))) # mu0/4pi
            res6+=np.matmul(lst_rot[Np-l],1e-7*np.sum(S6,axis=(0,1))/((lu-1)*(lv-1))) # mu0/4pi
            if False :# for debug
                eps=1e-3#1e-3
                tmp=np.array([self.surf.X[i,j]-self.surf.X+eps*self.surf.n[0,:,:], self.surf.Y[i,j]-self.surf.Y+eps*self.surf.n[1,:,:], self.surf.Z[i,j]-self.surf.Z+eps*self.surf.n[2,:,:]])
                iymxeps=np.sum(tmp**2,axis=0)
                #iymxeps[i,j]=1# for the division
                iymxeps=1/np.sqrt(iymxeps)
                #iymxeps[i,j]=0 # by convention
                j1y_ymx=np.sum(j1ya*np.moveaxis(tmp,0,2),axis=2)*iymx**3
                cc=j1y_ymx[:,:,np.newaxis]*j2
                c=1e-7*np.sum(self.surf.dS[:-1,:-1,np.newaxis]*(cc)[:-1,:-1,:],axis=(0,1))/((lu-1)*(lv-1))
                j1y_j2x=np.sum(j1ya*j2,axis=2)*iymxeps**3
                dd=j1y_j2x[np.newaxis,:,:]*np.array([self.surf.X[i,j]-self.surf.X, self.surf.Y[i,j]-self.surf.Y, self.surf.Z[i,j]-self.surf.Z])
                ddn=np.einsum('...,k...->k...',np.sum(self.surf.n*dd,axis=0),self.surf.n)
                d=1e-7*np.sum(self.surf.dS[np.newaxis,:-1,:-1]*(dd)[:,:-1,:-1],axis=(1,2))/((lu-1)*(lv-1))
                dn=1e-7*np.sum(self.surf.dS[np.newaxis,:-1,:-1]*(ddn)[:,:-1,:-1],axis=(1,2))/((lu-1)*(lv-1))
        res=res1+res2+res3+res4+res5+res6
        return res,res1,res2,res3,res4,res5,res6#,c-d

    # FRANK: HERE!!! 
    # Seems to calulate laplace force
    def f_laplace_optimized(self,ntheta,nzeta,coeff1,coeff2,lst_theta=None,lst_zeta=None):
        #general computation
        Np=self.surf.Np # Num field period
        rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
        lst_rot=np.array([np.linalg.matrix_power(rot,l) for l in range(Np+1)])
        lu,lv=self.lu,self.lv
        d1,d2=(1/(self.lu-1),1/(self.lv-1))
        param=(lu,lv,d1,d2)
        j1=self.get_full_j(coeff1)
        j2=self.get_full_j(coeff2)
        e1=np.tile(np.array([1,0,0]),(lu,lv,1))
        e2=np.tile(np.array([0,1,0]),(lu,lv,1))
        e3=np.tile(np.array([0,0,1]),(lu,lv,1))
        pi_x=self.get_pi_x()
        pi_xe1=np.einsum('...ij,...j->...i',pi_x,e1)# lu x lv x 3
        pi_xe2=np.einsum('...ij,...j->...i',pi_x,e2)# lu x lv x 3
        pi_xe3=np.einsum('...ij,...j->...i',pi_x,e3)# lu x lv x 3
        pull_back=self.get_vector_field_pull_back()
        pi_xe1_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe1)
        pi_xe2_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe2)
        pi_xe3_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe3)
        #pi_xe1_uv=self.vector_field_pull_back(pi_xe1)# lu x lv x 2
        #pi_xe2_uv=self.vector_field_pull_back(pi_xe2)# lu x lv x 2
        #pi_xe3_uv=self.vector_field_pull_back(pi_xe3)# lu x lv x 2
        #np.testing.assert_almost_equal(pi_xe1_uv,np.einsum('...ij,...j->...i',pull_back,pi_xe1))# lu x lv x 3)
        div_xe_uv=np.array([self.div(pi_xe1_uv),self.div(pi_xe2_uv),self.div(pi_xe3_uv)])
        #TODO : explicit computation of
        #dj2x,dj2y,dj2z= toroidal_surface.Toroidal_surface.grad(j2[:,:,0],j2[:,:,1],j2[:,:,2],self.surf.Np,circ=True)#periodic gradient
        #dj2=np.array([dj2x,dj2y,dj2z])
        
        #dj2x,dj2y,dj2z= toroidal_surface.Toroidal_surface.grad(j2[:,:,0],j2[:,:,1],j2[:,:,2],self.surf.Np,circ=True)#periodic gradient
        #dj2=np.array([dj2x,dj2y,dj2z])
        dj2=np.moveaxis(self.get_gradj(coeff2),3,0)#3 x 2 x lu x lv
        
        if lst_theta is None:
            lst_theta=range(ntheta)
        if lst_zeta is None:
            lst_zeta=range(nzeta)
        laplace_array_full=np.zeros((Np,len(lst_theta),len(lst_zeta),3))
        #specific computation:
        #for i in tqdm(range(len(lst_theta))):
        for i in range(len(lst_theta)):
            theta=lst_theta[i]
            for j in range(len(lst_zeta)):
                zeta=lst_zeta[j]
                for l in range(Np):
                    #we rotate j1 of rot^l
                    j1y=np.matmul(lst_rot[l],j1[theta,zeta,:])
                    j1ya=np.tile(j1y,(lu,lv,1))# lu x lv x 3
                    pi_xjy=np.einsum('...ij,...j->...i',pi_x,j1ya)# lu x lv x 3
                    pi_xjy_uv=np.einsum('...ij,...j->...i',pull_back,pi_xjy)
                    #pi_xjy_uv=self.vector_field_pull_back(pi_xjy)# lu x lv x 2
                    div_pi_xjy_uv=self.div(pi_xjy_uv)
                    res=aux(lst_rot[l],self.surf.X,self.surf.Y,self.surf.Z,theta,zeta,l,div_pi_xjy_uv, div_xe_uv, self.surf.g_upper,self.surf.dpsidu,self.surf.dpsidv,self.surf.n,self.surf.normal, self.surf.dS,j2,param,pi_xjy_uv,dj2,j1ya)
                    laplace_array_full[l,i,j,:]=np.matmul(lst_rot[Np-l],res)#lst_rot[Np-l] is the inverse of lst_rot[l]
                    
        return np.sum(laplace_array_full,axis=0)# we sum all contribution from the Np branches
    def compute_matrix_cost_L_eps(self,G,I,Phisize,epsilon):
        """compute the matrix associated to the linear a
        Phisize : m,n
        """
        from tqdm import tqdm
        n,m=Phisize
        l=2* (m*(2*n+1)+n)# nb of degree of freedom
        logging.info('Computation of the matrix elements for L_eps.cws size : {}x{}, eps={} nb Fourier components {} (m={},n={})'.format(self.surf.X.shape[0],self.surf.X.shape[1],epsilon ,l,m,n))
        A=np.zeros((cws.dim,l,3))
        CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(np.zeros(l),Phisize)
        b=self.L_eps_optimized((G,I,CC,CS),epsilon)
        for k in tqdm(range(l)):
            logging.debug('computation of L_eps : {}/{})'.format(k,l))
            zk=np.zeros(l)
            zk[k]=1
            CC,CS=Div_free_vector_field_on_TS.array_coeff_to_CS(zk,Phisize)
            res_tmp=self.L_eps_optimized((0,0,CC,CS),epsilon)
            A[:,k,:]=np.reshape(res_tmp)
        return (A,-b)
    def grad_1_f_laplace(self,ntheta,nzeta,coeff,djdc,lst_theta=None,lst_zeta=None):
        #general computation
        lc=len(djdc)#dim of j
        Np=self.surf.Np
        rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
        lst_rot=np.array([np.linalg.matrix_power(rot,l) for l in range(Np+1)])
        lu,lv=self.lu,self.lv
        d1,d2=(1/(self.lu-1),1/(self.lv-1))
        param=(lu,lv,lc,d1,d2)
        j1=self.get_full_j(coeff)
        j2=j1.copy()
        e1=np.tile(np.array([1,0,0]),(lu,lv,1))
        e2=np.tile(np.array([0,1,0]),(lu,lv,1))
        e3=np.tile(np.array([0,0,1]),(lu,lv,1))
        pi_x=self.get_pi_x() # lu x lv x 3 x 3
        pi_xe1=np.einsum('...ij,...j->...i',pi_x,e1)# lu x lv x 3
        pi_xe2=np.einsum('...ij,...j->...i',pi_x,e2)# lu x lv x 3
        pi_xe3=np.einsum('...ij,...j->...i',pi_x,e3)# lu x lv x 3
        pull_back=self.get_vector_field_pull_back() # lu x lv x 2 x 3
        pi_xe1_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe1)
        pi_xe2_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe2)
        pi_xe3_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe3)
        div_xe_uv=np.array([self.div(pi_xe1_uv),self.div(pi_xe2_uv),self.div(pi_xe3_uv)])

        #dj2x,dj2y,dj2z= toroidal_surface.Toroidal_surface.grad(j2[:,:,0],j2[:,:,1],j2[:,:,2],self.surf.Np,circ=True)#periodic gradient
        #dj2=np.array([dj2x,dj2y,dj2z])
        dj2=np.moveaxis(self.get_gradj(coeff),3,0)#3 x 2 x lu x lv
        # projection + pullback
        full_pull_back=np.einsum('...ij,...jk->...ik',pull_back,pi_x)# lu x lv x 2 x 3
        if lst_theta is None:
            lst_theta=range(ntheta)
        if lst_zeta is None:
            lst_zeta=range(nzeta)      
        return aux_grad_extended_j1(lst_rot,self.surf.X,self.surf.Y,self.surf.Z,div_xe_uv, self.surf.g_upper,self.surf.dpsidu,self.surf.dpsidv,self.surf.n,self.surf.normal, self.surf.dS,j2,param,dj2,j1,djdc,full_pull_back,np.array(lst_theta),np.array(lst_zeta),Np )

    def grad_2_f_laplace(self,ntheta,nzeta,coeff,djdc,matrix_gradj,lst_theta=None,lst_zeta=None):
        #general computation
        lc=len(djdc)#dim of j
        Np=self.surf.Np
        rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
        lst_rot=np.array([np.linalg.matrix_power(rot,l) for l in range(Np+1)])
        lu,lv=self.lu,self.lv
        d1,d2=(1/(self.lu-1),1/(self.lv-1))
        param=(lu,lv,lc,d1,d2)
        j1=self.get_full_j(coeff)
        j2=j1.copy()
        e1=np.tile(np.array([1,0,0]),(lu,lv,1))
        e2=np.tile(np.array([0,1,0]),(lu,lv,1))
        e3=np.tile(np.array([0,0,1]),(lu,lv,1))
        pi_x=self.get_pi_x()
        pi_xe1=np.einsum('...ij,...j->...i',pi_x,e1)# lu x lv x 3
        pi_xe2=np.einsum('...ij,...j->...i',pi_x,e2)# lu x lv x 3
        pi_xe3=np.einsum('...ij,...j->...i',pi_x,e3)# lu x lv x 3
        pull_back=self.get_vector_field_pull_back()
        pi_xe1_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe1)
        pi_xe2_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe2)
        pi_xe3_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe3)
        div_xe_uv=np.array([self.div(pi_xe1_uv),self.div(pi_xe2_uv),self.div(pi_xe3_uv)])
        #to improve
        #dj2_=np.zeros((3,2,ntheta+1,nzeta+1,lc))
        #for flag in range(lc):
        #    dj2_[0,:,:,:,flag],dj2_[1,:,:,:,flag],dj2_[2,:,:,:,flag]= toroidal_surface.Toroidal_surface.grad(djdc[flag,:,:,0],djdc[flag,:,:,1],djdc[flag,:,:,2],self.surf.Np,circ=True)#periodic gradient
        dj2_=np.swapaxes(matrix_gradj[:-1],0,4)
        if lst_theta is None:
            lst_theta=range(ntheta)
        if lst_zeta is None:
            lst_zeta=range(nzeta)
        laplace_array_full=np.zeros((Np,len(lst_theta),len(lst_zeta),lc,3))
        djdc_reshape=np.moveaxis(djdc,0,2)
        #specific computation:
        #for i in tqdm(range(len(lst_theta))):
        for i in range(len(lst_theta)):
            theta=lst_theta[i]
            for j in range(len(lst_zeta)):
                zeta=lst_zeta[j]
                for l in range(Np):
                    j1y=np.matmul(lst_rot[l],j1[theta,zeta,:])
                    j1ya=np.tile(j1y,(lu,lv,1))# lu x lv x 3
                    pi_xjy=np.einsum('...ij,...j->...i',pi_x,j1ya)# lu x lv x 3
                    pi_xjy_uv=np.einsum('...ij,...j->...i',pull_back,pi_xjy)
                    #pi_xjy_uv=self.vector_field_pull_back(pi_xjy)# lu x lv x 2
                    div_pi_xjy_uv=self.div(pi_xjy_uv)
                    res=aux_grad_j2(lst_rot[l],self.surf.X,self.surf.Y,self.surf.Z,theta,zeta,l,div_pi_xjy_uv, div_xe_uv, self.surf.g_upper,self.surf.dpsidu,self.surf.dpsidv,self.surf.n,self.surf.normal, self.surf.dS,djdc_reshape,param,pi_xjy_uv,dj2_,j1ya)
                    laplace_array_full[l,i,j,:,:]=np.einsum('ij,...j->...i',lst_rot[Np-l],res)#lst_rot[Np-l] is the inverse of lst_rot[l]
        return np.sum(laplace_array_full,axis=0)# we sum all contribution from the Np branches)

    def grad_f_laplace(self,ntheta,nzeta,coeff,djdc,matrix_gradj,lst_theta=None,lst_zeta=None):
        #general computation
        lc=len(djdc)#dim of j
        Np=self.surf.Np
        rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
        lst_rot=np.array([np.linalg.matrix_power(rot,l) for l in range(Np+1)])
        lu,lv=self.lu,self.lv
        d1,d2=(1/(self.lu-1),1/(self.lv-1))
        param=(lu,lv,lc,d1,d2)
        j1=self.get_full_j(coeff)
        j2=j1.copy()
        e1=np.tile(np.array([1,0,0]),(lu,lv,1))
        e2=np.tile(np.array([0,1,0]),(lu,lv,1))
        e3=np.tile(np.array([0,0,1]),(lu,lv,1))
        pi_x=self.get_pi_x()
        pi_xe1=np.einsum('...ij,...j->...i',pi_x,e1)# lu x lv x 3
        pi_xe2=np.einsum('...ij,...j->...i',pi_x,e2)# lu x lv x 3
        pi_xe3=np.einsum('...ij,...j->...i',pi_x,e3)# lu x lv x 3
        pull_back=self.get_vector_field_pull_back()
        pi_xe1_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe1)
        pi_xe2_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe2)
        pi_xe3_uv=np.einsum('...ij,...j->...i',pull_back,pi_xe3)
        div_xe_uv=np.array([self.div(pi_xe1_uv),self.div(pi_xe2_uv),self.div(pi_xe3_uv)])
        #to improve
        #dj2_=np.zeros((3,2,ntheta+1,nzeta+1,lc))
        #for flag in range(lc):
        #    dj2_[0,:,:,:,flag],dj2_[1,:,:,:,flag],dj2_[2,:,:,:,flag]= toroidal_surface.Toroidal_surface.grad(djdc[flag,:,:,0],djdc[flag,:,:,1],djdc[flag,:,:,2],self.surf.Np,circ=True)#periodic gradient
        dj2_=np.swapaxes(matrix_gradj[:-1],0,4) #(lc+1,2,self.lu,self.lv,3)
        #dj2x,dj2y,dj2z= toroidal_surface.Toroidal_surface.grad(j2[:,:,0],j2[:,:,1],j2[:,:,2],self.surf.Np,circ=True)#periodic gradient
        #dj2=np.array([dj2x,dj2y,dj2z])
        dj2=np.moveaxis(self.get_gradj(coeff),3,0)#3 x 2 x lu x lv
        full_pull_back=np.einsum('...ij,...jk->...ik',pull_back,pi_x)# lu x lv x 2 x 3
        if lst_theta is None:
            lst_theta=range(ntheta)
        if lst_zeta is None:
            lst_zeta=range(nzeta)

        return aux_grad_extended(lst_rot,self.surf.X,self.surf.Y,self.surf.Z,div_xe_uv, self.surf.g_upper,self.surf.dpsidu,self.surf.dpsidv,self.surf.n,self.surf.normal, self.surf.dS,j2,param,dj2,j1,djdc,full_pull_back,np.array(lst_theta),np.array(lst_zeta),dj2_,Np )

def aux_grad_extended(lst_rot,Px,Py,Pz,div_xe_uv,g_upper,dpsidu,dpsidv,surf_n,surf_normal,dS,j2,param,dj2,j1,djdc,full_pull_back,lst_theta,lst_zeta,dj2_,Np):
    (lu,lv,lc,d1,d2)=param
    lz=len(lst_zeta)
    djdc_r=np.moveaxis(djdc,0,2)
    laplace_array_full=np.zeros((Np,len(lst_theta),len(lst_zeta),lc,3))
    #initizlization of the arrays :
    j1y_=np.zeros((lc,3))
    pi_xjy_uv_=np.zeros((lu,lv,lc,2))
    div_pi_xjy_uv_=np.zeros((lu,lv,lc))
    aux=np.empty((lu,lv,lc))
    aux2=np.empty((lu,lv))
    daux=np.zeros((lu, lv, lc))
    daux2=np.zeros((lu, lv))
    P1=np.zeros((lu,lv,lc,3))
    P2=np.zeros((lu,lv,lc,3))
    P3=np.zeros((lu,lv,lc,3))
    P4=np.zeros((lu,lv,lc,3))
    P5=np.zeros((lu,lv,lc,3))
    P6=np.zeros((lu,lv,lc,3))
    sP1=np.zeros((lu,lv,lc,3))#(P1+P2+P3+P4+P5+P6)
    f1=np.empty((lu,lv,lc))
    f2=np.empty((lu,lv,lc))
    df=np.zeros((2,lu,lv,lc))
    gradf=np.zeros((2,lu,lv,lc))
    dS_3=np.expand_dims(dS,axis=2)
    div_pi_xjy_uv=np.zeros((lu,lv))
    for i in range(len(lst_theta)):
        #print(i)
        theta=lst_theta[i]
        for j in range(len(lst_zeta)):
            zeta=lst_zeta[j]
            for l in range(Np):
                rot_l=lst_rot[l]

                j1y=np.matmul(lst_rot[l],j1[theta,zeta,:])
                j1ya=np.tile(j1y,(lu,lv,1))# lu x lv x 3

                pi_xjy_uv=np.einsum('...ij,...j->...i',full_pull_back,j1ya)
                for flag in range(lc):
                    j1y_[flag,:]=np.dot(rot_l,djdc[flag,theta,zeta,:])

                j1ya_=np.expand_dims(np.expand_dims(j1y_,axis=0),axis=0)# 1 x 1 x lc x 3
                pi_xjy_uv_*=0
                compute_pi_xjy_uv_(pi_xjy_uv_,full_pull_back,j1y_,lu,lv,lc)

                div_pi_xjy_uv_*=0
                div_pi_xjy_uv*=0
                compute_div_pi_xjy_uv(div_pi_xjy_uv,div_pi_xjy_uv_,pi_xjy_uv,pi_xjy_uv_,dS,dS_3,aux,aux2,daux,daux2,lu,lv,lc)
                #
                #res2=aux_grad_j2(rot_l,Px,Py,Pz,theta,zeta,l,div_pi_xjy_uv, div_xe_uv, g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,np.moveaxis(djdc,0,2),param,pi_xjy_uv,dj2_,j1ya)
                #(lst_rot,Px,Py,Pz,div_xe_uv,g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,j2,param,dj2,j1,djdc,full_pull_back,lst_theta,lst_zeta,Np)
                #P1r,P2r,P3r,P4r,P5r,P6r=aux_grad_j2_debug(rot_l,Px,Py,Pz,theta,zeta,l,div_pi_xjy_uv, div_xe_uv, g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,np.moveaxis(djdc,0,2),param,pi_xjy_uv,dj2_,j1ya)
                newY=np.dot(rot_l,np.array([Px[theta,zeta],Py[theta,zeta],Pz[theta,zeta]]))
                d2=(newY[0]-Px)**2+(newY[1]-Py)**2+(newY[2]-Pz)**2
                if l==0:
                    d2[theta,zeta]=1# for the division
                elif zeta==0 and l==1:
                    d2[theta,-1]=1
                    if theta==0: d2[-1,-1]=1
                iymx=1/np.sqrt(d2)
                if l==0:
                    iymx[theta,zeta]=0#by convention

                aux_P1(P1,iymx,div_pi_xjy_uv_,j2,div_pi_xjy_uv,djdc_r,lu,lv,lc)
      
                #P2*=0
                aux_P2_full(P2,pi_xjy_uv,pi_xjy_uv_,dj2,dj2_,iymx,lu,lv,lc)
                
                compute_f1_f2(f1,f2,j1ya_,j1ya,j2,djdc_r,lu,lv,lc)
                #f1=np.sum(j1ya_*np.expand_dims(j2,axis=2),axis=3)
                #f2=np.sum(np.expand_dims(j1ya,axis=2)*djdc_r,axis=3)
                f=f1+f2
                #df*=0
                #TODO : use explicit expression
                #df*=0
                #for flag in range(3):
                #    df+=dj2[flag,:,:,:,np.newaxis]*j1y_[np.newaxis,np.newaxis,np.newaxis,:,flag]
                #    df+=dj2_[flag]*j1ya[np.newaxis,:,:,np.newaxis,flag]
                compute_df_optimized(df,dj2,j1y_,dj2_,j1ya,lu,lv,lc)
                #gradf=np.einsum('ij...,j...->i...',g_upper,df)
                #gradf*=0
                compute_gradf(gradf,df,g_upper,lu,lv,lc)
                #P3*=0
                aux_P3(P3,iymx,gradf,dpsidu,dpsidv,lu,lv,lc)

                ymx=np.zeros((3,lu,lv))
                ymx[0]=newY[0]-Px
                ymx[1]=newY[1]-Py
                ymx[2]=newY[2]-Pz
                K=iymx**3*np.sum(ymx*surf_n,axis=0)# K=<y-x,n(x)> /|y-x|^3
                aux_P4(P4,K,j1ya_,j1ya,surf_normal,j2,djdc_r,lu,lv,lc)
                #P5=-1*np.einsum('...,...k->...k',f*K,surf_normal)
                aux_P5(P5,f,K,surf_normal,lu,lv,lc)

                aux_P6(P6,f,iymx,div_xe_uv,lu,lv,lc)
                #reduction
                aux_sP(sP1,P1,P2,P3,P4,P5,P6,lu,lv,lc)
                #sP1=P1+P2+P3+P4+P5+P6
                sP=sP1[:-1,:-1,:,:]
                res=1e-7/((lu-1)*(lv-1))*np.einsum('ij,ij...->...',dS[:-1,:-1],sP)
                laplace_array_full[l,i,j,:,:]=np.sum(np.expand_dims(lst_rot[Np-l],axis=0)*np.expand_dims(res,axis=1),axis=2)
    return  np.sum(laplace_array_full,axis=0)# we sum all contribution from the Np branches)

                #    df+=dj2[flag,:,:,:,np.newaxis]*j1y_[np.newaxis,np.newaxis,np.newaxis,:,flag]
                #    df+=dj2_[flag]*j1ya[np.newaxis,:,:,np.newaxis,flag]
@numba.njit(fastmath=fastmathbool,cache=True,parallel=Par)
def compute_df_optimized(df,dj2,j1y_,dj2_,j1ya,lu,lv,lc):
    #dj2 3 x 2 x lu x lv
    #df (2,lu,lv,lc)
    for i in range(2):
        for j in range(lu):
            for k in range(lv):
                for l in range(lc):
                    df[i,j,k,l]=0
                    for m in range(3):
                        df[i,j,k,l]+=dj2[m,i,j,k]*j1y_[l,m]
                        df[i,j,k,l]+=dj2_[m,i,j,k,l]*j1ya[j,k,m]
@numba.njit(fastmath=fastmathbool,cache=True)
def compute_div_pi_xjy_uv(div_pi_xjy_uv,div_pi_xjy_uv_,pi_xjy_uv,pi_xjy_uv_,dS,dS_3,aux,aux2,daux,daux2,lu,lv,lc):
    d1,d2=(1/(lu-1),1/(lv-1))
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                #part/du
                aux[i,j,k]=dS[i,j]*pi_xjy_uv_[i,j,k,0]
            aux2[i,j]=dS[i,j]*pi_xjy_uv[i,j,0]
    for ii in range(1,lu-1):
        for j in range(lv):
            for k in range(lc):
                daux[ii,j,k]=(aux[ii+1,j,k]-aux[ii-1,j,k])/(2*d1)
            daux2[ii,j]=(aux2[ii+1,j]-aux2[ii-1,j])/(2*d1)
    for j in range(lv):
        for k in range(lc):
            daux[0,j,k]=(aux[1,j,k]-aux[0,j,k])/(d1)
            daux[-1,j,k]=(aux[-2,j,k]-aux[-1,j,k])/(d1)
        daux2[0,j]=(aux2[1,j]-aux2[0,j])/(d1)
        daux2[-1,j]=(aux2[-2,j]-aux2[-1,j])/(d1)
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                div_pi_xjy_uv_[i,j,k]=daux[i,j,k]/dS[i,j]
            div_pi_xjy_uv[i,j]=daux2[i,j]/dS[i,j]
    #part/dv
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                #part/du
                aux[i,j,k]=dS[i,j]*pi_xjy_uv_[i,j,k,1]
            aux2[i,j]=dS[i,j]*pi_xjy_uv[i,j,1]
    for jj in range(1,lv-1):
        for i in range(lu):
            for k in range(lc):
                daux[i,jj,k]=(aux[i,jj+1,k]-aux[i,jj-1,k])/(2*d2)
            daux2[i,jj]=(aux2[i,jj+1]-aux2[i,jj-1])/(2*d2)
    for i in range(lu):
        for k in range(lc):
            daux[i,0,k]=(aux[i,1,k]-aux[i,0,k])/(d2)
            daux[i,-1,k]=(aux[i,-2,k]-aux[i,-1,k])/(d2)
        daux2[i,0]=(aux2[i,1]-aux2[i,0])/(d2)
        daux2[i,-1]=(aux2[i,-2]-aux2[i,-1])/(d2)
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                div_pi_xjy_uv_[i,j,k]+=daux[i,j,k]/dS[i,j]
            div_pi_xjy_uv[i,j]+=daux2[i,j]/dS[i,j]
@numba.njit(fastmath=fastmathbool,cache=True,parallel=Par)
def aux_P1(P1,iymx,div_pi_xjy_uv_,j2,div_pi_xjy_uv,djdc_r,lu,lv,lc):
    for i in prange(lu):
        for j in range(lv):
            for k in range(lc):
                for m in range(3):
                    P1[i,j,k,m]=-1*iymx[i,j]*div_pi_xjy_uv_[i,j,k]*j2[i,j,m]
                    P1[i,j,k,m]-=iymx[i,j]*div_pi_xjy_uv[i,j]*djdc_r[i,j,k,m]
@numba.njit(fastmath=fastmathbool,cache=True,parallel=Par)
def aux_P5(P5,f,K,surf_normal,lu,lv,lc):
    for i in prange(lu):
        for j in range(lv):
            for k in range(lc):
                for m in range(3):
                    P5[i,j,k,m]=-1*f[i,j,k]*K[i,j]*surf_normal[i,j,m]
@numba.njit(fastmath=fastmathbool,cache=True,parallel=Par)
def aux_P4(P4,K,j1ya_,j1ya,surf_normal,j2,djdc_r,lu,lv,lc):
    for i in prange(lu):
        for j in range(lv):
            for k in range(lc):
                for m in range(3):
                    P4[i,j,k,m]=0
                    for n in range(3):
                        P4[i,j,k,m]+=K[i,j]*surf_normal[i,j,n]*j1ya_[0,0,k,n]*j2[i,j,m]
                        P4[i,j,k,m]+=K[i,j]*surf_normal[i,j,n]*j1ya[i,j,n]*djdc_r[i,j,k,m]

                #c=np.expand_dims(K,axis=2)*np.sum(j1ya_*np.expand_dims(surf_normal,axis=2),axis=3)
                #c2=K*np.sum(j1ya*surf_normal,axis=2)
                #P4=np.expand_dims(c,axis=3)*np.expand_dims(j2,axis=2)
                #P4+=np.expand_dims(np.expand_dims(c2,axis=2),axis=3)*djdc_r
@numba.njit(fastmath=fastmathbool,cache=True,parallel=Par)
def aux_P6(P6,f,iymx,div_xe_uv,lu,lv,lc):
    for i in prange(lu):
        for j in range(lv):
            for k in range(lc):
                for m in range(3):
                    P6[i,j,k,m]=f[i,j,k]*iymx[i,j]*div_xe_uv[m,i,j]
                #P6[:,:,:,0]=f*np.expand_dims(iymx*div_xe_uv[0],axis=2)
                #P6[:,:,:,1]=f*np.expand_dims(iymx*div_xe_uv[1],axis=2)
                #P6[:,:,:,2]=f*np.expand_dims(iymx*div_xe_uv[2],axis=2)
@numba.njit(fastmath=fastmathbool,cache=True)
def compute_gradf(gradf,df,g_upper,lu,lv,lc):
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                gradf[0,i,j,k]=g_upper[0,0,i,j]*df[0,i,j,k]+g_upper[0,1,i,j]*df[1,i,j,k]
                gradf[1,i,j,k]=g_upper[1,0,i,j]*df[0,i,j,k]+g_upper[1,1,i,j]*df[1,i,j,k]
@numba.njit(fastmath=fastmathbool,cache=True,parallel=Par)
def compute_f1_f2(f1,f2,j1ya_,j1ya,j2,djdc_r,lu,lv,lc):
    for i in prange(lu):
        for j in range(lv):
            for k in range(lc):
                f1[i,j,k]=0
                f2[i,j,k]=0
                for l in range(3):
                    f1[i,j,k]+=j1ya_[0,0,k,l]*j2[i,j,l]
                    f2[i,j,k]+=j1ya[i,j,l]*djdc_r[i,j,k,l]
                #np.sum(j1ya_*np.expand_dims(j2,axis=2),axis=3)
                #1/(y-x) \nabla <j1(y) j2(x) >
                #f2=np.sum(np.expand_dims(j1ya,axis=2)*djdc_r,axis=3)
#@numba.njit(fastmath=fastmathbool,cache=True)  
def aux_pi_xjy_uv_(full_pull_back,j1ya_):
    #lu x lv x 2 x 3 times lu x lv x lc x 3 > lu x lv x lc x 2
    #print(full_pull_back.shape,j1ya_.shape)
    return np.sum(np.expand_dims(full_pull_back,axis=2)*np.expand_dims(j1ya_,axis=3),axis=4)
    
@numba.njit(fastmath=fastmathbool,cache=True)
def compute_df(df,f,lu,lv):
    for jj in range(lv):
        #print(jj,df[0,0,jj],(f[1,jj]-f[0,jj])*(lu-1))
        df[0,0,jj,:]=(f[1,jj,:]-f[0,jj,:])*(lu-1)
        df[0,-1,jj,:]=(f[-1,jj,:]-f[-2,jj,:])*(lu-1)
    for ii in range(1,lu-1):
        for jj in range(lv):
            df[0,ii,jj,:]=(f[ii+1,jj,:]-f[ii-1,jj,:])*(lu-1)/2
    for ii in range(lu):
        df[1,ii,0,:]=(f[ii,1,:]-f[ii,0,:])*(lv-1)
        df[1,ii,-1,:]=(f[ii,-2,:]-f[ii,-1,:])*(lv-1)
    for jj in range(1,lv-1):
        for ii in range(lu):
            df[1,ii,jj,:]=(f[ii,jj+1,:]-f[ii,jj-1,:])*(lv-1)/2

@numba.njit(fastmath=fastmathbool,cache=True)
def compute_pi_xjy_uv_(pi_xjy_uv_,full_pull_back,j1y_,lu,lv,lc):
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                for l in range(3):
                    pi_xjy_uv_[i,j,k,0]+=full_pull_back[i,j,0,l]*j1y_[k,l]
                    pi_xjy_uv_[i,j,k,1]+=full_pull_back[i,j,1,l]*j1y_[k,l]
    
@numba.njit(fastmath=fastmathbool,cache=True)
def compute_f(f,j1y_,j2,lc,lu,lv):
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                for l in range(3):
                    f[i,j,k]+=j1y_[k,l]*j2[i,j,l]
@numba.njit(fastmath=fastmathbool,cache=True)
def aux_reduce(res,dS,sP,lu,lv,lc):
    for i in range(lu-1):
        for j in range(lv-1):
            for k in range(lc):
                for l in range(3):
                    res[k,l]+=(dS[i,j]*sP[i,j,k,l])
@numba.njit(fastmath=fastmathbool,cache=True)
def aux_sP(sP,P1,P2,P3,P4,P5,P6,lu,lv,lc):
    for i in range(lu-1):
        for j in range(lv-1):
            for k in range(lc):
                for l in range(3):
                    sP[i,j,k,l]=P1[i,j,k,l]+P2[i,j,k,l]+P3[i,j,k,l]+P4[i,j,k,l]+P5[i,j,k,l]+P6[i,j,k,l]
    #S=np.expand_dims(np.expand_dims(dS[:-1,:-1],axis=2),axis=3)*sP[:-1,:-1,:,:]
    #res_p=1e-7*np.sum(S,axis=1)/((lu-1)*(lv-1))
    #res=np.sum(res_p,axis=0)
@numba.njit(fastmath=fastmathbool,cache=True,parallel=Par)
def aux_P2_full(P2,pi_xjy_uv,pi_xjy_uv_,dj2,dj2_,iymx,lu,lv,lc):
    for i in prange(lu):
        for j in range(lv):
            for k in range(lc):
                for m in range(3):
                    P2[i,j,k,m]=0
                    for l in range(2):
                        P2[i,j,k,m]-=pi_xjy_uv_[i,j,k,l]*dj2[m,l,i,j]*iymx[i,j]
                        P2[i,j,k,m]-=pi_xjy_uv[i,j,l]*dj2_[m,l,i,j,k]*iymx[i,j]
                    #for k in range(2):
                #    P2[:,:,:,0]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[0][k]
                #    P2[:,:,:,1]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[1][k]
                #    P2[:,:,:,2]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[2][k]
                #P2*=-1*np.expand_dims(np.expand_dims(iymx,axis=2),axis=3)
@numba.njit(fastmath=fastmathbool,cache=True)
def aux_P2(P2,pi_xjy_uv_,dj2,iymx,lu,lv,lc):
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                for l in range(2):
                    for m in range(3):
                        P2[i,j,k,m]-=pi_xjy_uv_[i,j,k,l]*dj2[m,l,i,j]*iymx[i,j]
@numba.njit(fastmath=fastmathbool,cache=True)
def aux_P3(P3,iymx,gradf,dpsidu,dpsidv,lu,lv,lc):
    for i in range(lu):
        for j in range(lv):
            for k in range(lc):
                for m in range(3):
                    P3[i,j,k,m]=(gradf[0,i,j,k]*dpsidu[m,i,j]+gradf[1,i,j,k]*dpsidv[m,i,j])*iymx[i,j]
                    #                    for k in range(3):
                    #P3[:,:,:,k]+=np.expand_dims(iymx,axis=2)*(gradf[0,:,:,:]*np.expand_dims(dpsidu[k],axis=2)+gradf[1,:,:,:]*np.expand_dims(dpsidv[k],axis=2))# lu x lv x 3

def aux_grad_extended_j1(lst_rot,Px,Py,Pz,div_xe_uv,g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,j2,param,dj2,j1y,djdc,full_pull_back,lst_theta,lst_zeta,Np):
    (lu,lv,lc,d1,d2)=param
    laplace_array_full=np.zeros((Np,len(lst_theta),len(lst_zeta),lc,3))
    #initizlization of the arrays :
    j1y_=np.zeros((lc,3))
    pi_xjy_uv_=np.zeros((lu,lv,lc,2))
    div_pi_xjy_uv_=np.zeros((lu,lv,lc))
    daux=np.zeros((lu, lv, lc))
    P2=np.zeros((lu,lv,lc,3))
    P3=np.zeros((lu,lv,lc,3))
    P4=np.zeros((lu,lv,lc,3))
    sP1=np.zeros((lu,lv,lc,3))#(P1+P2+P3+P4+P5+P6)
    df=np.zeros((2,lu,lv,lc))
    gradf=np.zeros((2,lu,lv,lc))
    for i in range(len(lst_theta)):
        #print(i)
        theta=lst_theta[i]
        for j in range(len(lst_zeta)):
            zeta=lst_zeta[j]
            for l in range(Np):
                rot_l=lst_rot[l]
                #rotation of j1
                #j1y_=np.einsum('ij,...j->...i',lst_rot[l],djdc[:,theta,zeta,:])
                for flag in range(lc):
                    j1y_[flag,:]=np.dot(rot_l,djdc[flag,theta,zeta,:])
                #j1ya_=np.tile(j1y_,(lu,lv,1,1))# lu x lv x lc x 3
                j1ya_=np.expand_dims(np.expand_dims(j1y_,axis=0),axis=0)# 1 x 1 x lc x 3
                pi_xjy_uv_*=0
                compute_pi_xjy_uv_(pi_xjy_uv_,full_pull_back,j1y_,lu,lv,lc)
                #pi_xjy_uv_=np.sum(np.expand_dims(full_pull_back,axis=2)*np.expand_dims(j1ya_,axis=3),axis=4)
                #div array : return the divergence of pi_xjy_uv_ : lu x lv x lc x 2
                div_pi_xjy_uv_*=0
                d1,d2=(1/(lu-1),1/(lv-1))
                #part/du
                aux=np.expand_dims(dS,axis=2)*pi_xjy_uv_[:,:,:,0]
                for ii in range(1,lu-1):
                    daux[ii]=(aux[ii+1]-aux[ii-1])/(2*d1)
                daux[0]=(aux[1]-aux[0])/(d1)
                daux[-1]=(aux[-2]-aux[-1])/(d1)
                div_pi_xjy_uv_+=daux/np.expand_dims(dS,axis=2)
                #part/dv
                aux=np.expand_dims(dS,axis=2)*pi_xjy_uv_[:,:,:,1]
                for jj in range(1,lv-1):
                    daux[:,jj]=(aux[:,jj+1]-aux[:,jj-1])/(2*d2)
                daux[:,0]=(aux[:,1]-aux[:,0])/(d2)
                daux[:,-1]=(aux[:,-2]-aux[:,-1])/(d2)
                div_pi_xjy_uv_+=daux/np.expand_dims(dS,axis=2)

                newY=np.dot(rot_l,np.array([Px[theta,zeta],Py[theta,zeta],Pz[theta,zeta]]))
                d2=(newY[0]-Px)**2+(newY[1]-Py)**2+(newY[2]-Pz)**2
                if l==0:
                    d2[theta,zeta]=1# for the division
                elif zeta==0 and l==1:
                    d2[theta,-1]=1
                    if theta==0: d2[-1,-1]=1
                iymx=1/np.sqrt(d2)
                if l==0:
                    iymx[theta,zeta]=0#by convention
                P1=-1*np.expand_dims(np.expand_dims(iymx,axis=2)*div_pi_xjy_uv_,axis=3)*np.expand_dims(j2,axis=2)

                P2*=0
                aux_P2(P2,pi_xjy_uv_,dj2,iymx,lu,lv,lc)

                #1/(y-x) \nabla <j1(y) j2(x) >
                #f=np.empty((lu,lv,lc))
                #compute_f(f,j1y_,j2,lc,lu,lv)
                f=np.sum(j1ya_*np.expand_dims(j2,axis=2),axis=3)
                df*=0
                for flag in range(3):
                    df+=dj2[flag,:,:,:,np.newaxis]*j1y_[np.newaxis,np.newaxis,np.newaxis,:,flag]
                #compute_df(df,f,lu,lv)
                #gradf=np.einsum('ij...,j...->i...',g_upper,df)
                gradf*=0
                gradf[0]=np.expand_dims(g_upper[0,0],axis=2)*df[0]+np.expand_dims(g_upper[0,1],axis=2)*df[1]
                gradf[1]=np.expand_dims(g_upper[1,0],axis=2)*df[0]+np.expand_dims(g_upper[1,1],axis=2)*df[1]
                P3*=0
                aux_P3(P3,iymx,gradf,dpsidu,dpsidv,lu,lv,lc)
                P6=np.zeros((lu,lv,lc,3))
                P6[:,:,:,0]=f*np.expand_dims(iymx*div_xe_uv[0],axis=2)
                P6[:,:,:,1]=f*np.expand_dims(iymx*div_xe_uv[1],axis=2)
                P6[:,:,:,2]=f*np.expand_dims(iymx*div_xe_uv[2],axis=2)
                ymx=np.zeros((3,lu,lv))
                ymx[0]=newY[0]-Px
                ymx[1]=newY[1]-Py
                ymx[2]=newY[2]-Pz
                K=iymx**3*np.sum(ymx*surf_n,axis=0)# K=<y-x,n(x)> /|y-x|^3
                c=np.expand_dims(K,axis=2)*np.sum(j1ya_*np.expand_dims(surf_normal,axis=2),axis=3)
                #P4=np.einsum('...,...k->...k',c,j2)
                P4=np.expand_dims(c,axis=3)*np.expand_dims(j2,axis=2)
                #P5=-1*np.einsum('...,...k->...k',f*K,surf_normal)
                P5=-1*np.expand_dims(f*np.expand_dims(K,axis=2),axis=3)*np.expand_dims(surf_normal,axis=2)
                #res=np.zeros((lc,3))
                #aux_reduce(res,dS,P1+P2+P3+P4+P5+P6,lu,lv,lc)
                aux_sP(sP1,P1,P2,P3,P4,P5,P6,lu,lv,lc)
                sP=sP1[:-1,:-1,:,:]
                res=1e-7/((lu-1)*(lv-1))*np.einsum('ij,ij...->...',dS[:-1,:-1],sP)
                laplace_array_full[l,i,j,:,:]=np.sum(np.expand_dims(lst_rot[Np-l],axis=0)*np.expand_dims(res,axis=1),axis=2)
    return  np.sum(laplace_array_full,axis=0)# we sum all contribution from the Np branches)
@numba.njit(fastmath=fastmathbool,cache=True)
def aux_grad_j1(rot_l,Px,Py,Pz,theta,zeta,l,div_pi_xjy_uv,div_xe_uv,g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,j2,param,pi_xjy_uv,dj2,j1ya):
    (lu,lv,lc,d1,d2)=param
    newY=np.dot(rot_l,np.array([Px[theta,zeta],Py[theta,zeta],Pz[theta,zeta]]))
    d2=(newY[0]-Px)**2+(newY[1]-Py)**2+(newY[2]-Pz)**2
    if l==0:
        d2[theta,zeta]=1# for the division
    elif zeta==0 and l==1:
        d2[theta,-1]=1
        if theta==0: d2[-1,-1]=1
    iymx=1/np.sqrt(d2)
    if l==0:
        iymx[theta,zeta]=0#by convention
    P1=-1*np.expand_dims(np.expand_dims(iymx,axis=2)*div_pi_xjy_uv,axis=3)*np.expand_dims(j2,axis=2)
    P2=np.zeros((lu,lv,lc,3))
    for k in range(2):
        P2[:,:,:,0]+=pi_xjy_uv[:,:,:,k]*np.expand_dims(dj2[0][k],axis=2)
        P2[:,:,:,1]+=pi_xjy_uv[:,:,:,k]*np.expand_dims(dj2[1][k],axis=2)
        P2[:,:,:,2]+=pi_xjy_uv[:,:,:,k]*np.expand_dims(dj2[2][k],axis=2)
    P2*=-1*np.expand_dims(np.expand_dims(iymx,axis=2),axis=3)
    #1/(y-x) \nabla <j1(y) j2(x) >
    f=np.sum(j1ya*np.expand_dims(j2,axis=2),axis=3)
    df=np.zeros((2,lu,lv,lc))
    for jj in range(lv):
        #print(jj,df[0,0,jj],(f[1,jj]-f[0,jj])*(lu-1))
        df[0,0,jj,:]=(f[1,jj,:]-f[0,jj,:])*(lu-1)
        df[0,-1,jj,:]=(f[-1,jj,:]-f[-2,jj,:])*(lu-1)
    for ii in range(1,lu-1):
        for jj in range(lv):
            df[0,ii,jj,:]=(f[ii+1,jj,:]-f[ii-1,jj,:])*(lu-1)/2
    for ii in range(lu):
        df[1,ii,0,:]=(f[ii,1,:]-f[ii,0,:])*(lv-1)
        df[1,ii,-1,:]=(f[ii,-2,:]-f[ii,-1,:])*(lv-1)
    for jj in range(1,lv-1):
        for ii in range(lu):
            df[1,ii,jj,:]=(f[ii,jj+1,:]-f[ii,jj-1,:])*(lv-1)/2
            
    #gradf=np.einsum('ij...,j...->i...',g_upper,df)
    gradf=np.zeros((2,lu,lv,lc))
    gradf[0]=np.expand_dims(g_upper[0,0],axis=2)*df[0]+np.expand_dims(g_upper[0,1],axis=2)*df[1]
    gradf[1]=np.expand_dims(g_upper[1,0],axis=2)*df[0]+np.expand_dims(g_upper[1,1],axis=2)*df[1]
    P3=np.zeros((lu,lv,lc,3))
    for k in range(3):
        P3[:,:,:,k]+=np.expand_dims(iymx,axis=2)*(gradf[0,:,:,:]*np.expand_dims(dpsidu[k],axis=2)+gradf[1,:,:,:]*np.expand_dims(dpsidv[k],axis=2))# lu x lv x 3
    P6=np.zeros((lu,lv,lc,3))
    P6[:,:,:,0]=f*np.expand_dims(iymx*div_xe_uv[0],axis=2)
    P6[:,:,:,1]=f*np.expand_dims(iymx*div_xe_uv[1],axis=2)
    P6[:,:,:,2]=f*np.expand_dims(iymx*div_xe_uv[2],axis=2)
    ymx=np.zeros((3,lu,lv))
    ymx[0]=newY[0]-Px
    ymx[1]=newY[1]-Py
    ymx[2]=newY[2]-Pz
    K=iymx**3*np.sum(ymx*surf_n,axis=0)# K=<y-x,n(x)> /|y-x|^3
    c=np.expand_dims(K,axis=2)*np.sum(j1ya*np.expand_dims(surf_normal,axis=2),axis=3)
    #P4=np.einsum('...,...k->...k',c,j2)
    P4=np.expand_dims(c,axis=3)*np.expand_dims(j2,axis=2)
    #P5=-1*np.einsum('...,...k->...k',f*K,surf_normal)
    P5=-1*np.expand_dims(f*np.expand_dims(K,axis=2),axis=3)*np.expand_dims(surf_normal,axis=2)
    S=np.expand_dims(np.expand_dims(dS[:-1,:-1],axis=2),axis=3)*(P1+P2+P3+P4+P5+P6)[:-1,:-1,:,:]
    res_p=1e-7*np.sum(S,axis=1)/((lu-1)*(lv-1))
    return np.sum(res_p,axis=0)

#@numba.njit(fastmath=fastmathbool,cache=True)
def aux_grad_j2(rot_l,Px,Py,Pz,theta,zeta,l,div_pi_xjy_uv,div_xe_uv,g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,djdc_r,param,pi_xjy_uv,dj2_,j1ya):
    (lu,lv,lc,d1,d2)=param
    newY=np.dot(rot_l,np.array([Px[theta,zeta],Py[theta,zeta],Pz[theta,zeta]]))
    d2=(newY[0]-Px)**2+(newY[1]-Py)**2+(newY[2]-Pz)**2
    if l==0:
        d2[theta,zeta]=1# for the division
    elif zeta==0 and l==1:
        d2[theta,-1]=1
        if theta==0: d2[-1,-1]=1
    iymx=1/np.sqrt(d2)
    if l==0:
        iymx[theta,zeta]=0#by convention
    P1=-1*np.expand_dims(np.expand_dims(iymx*div_pi_xjy_uv,axis=2),axis=3)*djdc_r
    P2=np.zeros((lu,lv,lc,3))
    for k in range(2):
        P2[:,:,:,0]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[0][k]
        P2[:,:,:,1]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[1][k]
        P2[:,:,:,2]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[2][k]
    P2*=-1*np.expand_dims(np.expand_dims(iymx,axis=2),axis=3)
    #1/(y-x) \nabla <j1(y) j2(x) >
    f=np.sum(np.expand_dims(j1ya,axis=2)*djdc_r,axis=3)
    df=np.zeros((2,lu,lv,lc))
    for flag in range(3):
        df+=dj2_[flag]*j1ya[np.newaxis,:,:,np.newaxis,flag]
    #for jj in range(lv):
        #print(jj,df[0,0,jj],(f[1,jj]-f[0,jj])*(lu-1))
    #    df[0,0,jj,:]=(f[1,jj,:]-f[0,jj,:])*(lu-1)
    #    df[0,-1,jj,:]=(f[-1,jj,:]-f[-2,jj,:])*(lu-1)
    #for ii in range(1,lu-1):
    #    for jj in range(lv):
    #        df[0,ii,jj,:]=(f[ii+1,jj,:]-f[ii-1,jj,:])*(lu-1)/2
    #for ii in range(lu):
    #    df[1,ii,0,:]=(f[ii,1,:]-f[ii,0,:])*(lv-1)
    #    df[1,ii,-1,:]=(f[ii,-2,:]-f[ii,-1,:])*(lv-1)
    #for jj in range(1,lv-1):
    #    for ii in range(lu):
    #        df[1,ii,jj,:]=(f[ii,jj+1,:]-f[ii,jj-1,:])*(lv-1)/2
            
    #gradf=np.einsum('ij...,j...->i...',g_upper,df)
    gradf=np.zeros((2,lu,lv,lc))
    gradf[0]=np.expand_dims(g_upper[0,0],axis=2)*df[0]+np.expand_dims(g_upper[0,1],axis=2)*df[1]
    gradf[1]=np.expand_dims(g_upper[1,0],axis=2)*df[0]+np.expand_dims(g_upper[1,1],axis=2)*df[1]
    P3=np.zeros((lu,lv,lc,3))
    for k in range(3):
        P3[:,:,:,k]+=np.expand_dims(iymx,axis=2)*(gradf[0,:,:,:]*np.expand_dims(dpsidu[k],axis=2)+gradf[1,:,:,:]*np.expand_dims(dpsidv[k],axis=2))# lu x lv x 3
    P6=np.zeros((lu,lv,lc,3))
    P6[:,:,:,0]=f*np.expand_dims(iymx*div_xe_uv[0],axis=2)
    P6[:,:,:,1]=f*np.expand_dims(iymx*div_xe_uv[1],axis=2)
    P6[:,:,:,2]=f*np.expand_dims(iymx*div_xe_uv[2],axis=2)
    ymx=np.zeros((3,lu,lv))
    ymx[0]=newY[0]-Px
    ymx[1]=newY[1]-Py
    ymx[2]=newY[2]-Pz
    K=iymx**3*np.sum(ymx*surf_n,axis=0)# K=<y-x,n(x)> /|y-x|^3
    c=K*np.sum(j1ya*surf_normal,axis=2)
    #P4=np.einsum('...,...k->...k',c,j2)
    P4=np.expand_dims(np.expand_dims(c,axis=2),axis=3)*djdc_r
    #P5=-1*np.einsum('...,...k->...k',f*K,surf_normal)
    P5=-1*np.expand_dims(f*np.expand_dims(K,axis=2),axis=3)*np.expand_dims(surf_normal,axis=2)
    S=np.expand_dims(np.expand_dims(dS[:-1,:-1],axis=2),axis=3)*(P1+P2+P3+P4+P5+P6)[:-1,:-1,:,:]
    res_p=1e-7*np.sum(S,axis=1)/((lu-1)*(lv-1))
    return np.sum(res_p,axis=0)
def aux_grad_j2_debug(rot_l,Px,Py,Pz,theta,zeta,l,div_pi_xjy_uv,div_xe_uv,g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,djdc_r,param,pi_xjy_uv,dj2_,j1ya):
    (lu,lv,lc,d1,d2)=param
    newY=np.dot(rot_l,np.array([Px[theta,zeta],Py[theta,zeta],Pz[theta,zeta]]))
    d2=(newY[0]-Px)**2+(newY[1]-Py)**2+(newY[2]-Pz)**2
    if l==0:
        d2[theta,zeta]=1# for the division
    elif zeta==0 and l==1:
        d2[theta,-1]=1
        if theta==0: d2[-1,-1]=1
    iymx=1/np.sqrt(d2)
    if l==0:
        iymx[theta,zeta]=0#by convention
    P1=-1*np.expand_dims(np.expand_dims(iymx*div_pi_xjy_uv,axis=2),axis=3)*djdc_r
    P2=np.zeros((lu,lv,lc,3))
    for k in range(2):
        P2[:,:,:,0]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[0][k]
        P2[:,:,:,1]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[1][k]
        P2[:,:,:,2]+=np.expand_dims(pi_xjy_uv[:,:,k],axis=2)*dj2_[2][k]
    P2*=-1*np.expand_dims(np.expand_dims(iymx,axis=2),axis=3)
    #1/(y-x) \nabla <j1(y) j2(x) >
    f=np.sum(np.expand_dims(j1ya,axis=2)*djdc_r,axis=3)
    df=np.zeros((2,lu,lv,lc))
    for jj in range(lv):
        #print(jj,df[0,0,jj],(f[1,jj]-f[0,jj])*(lu-1))
        df[0,0,jj,:]=(f[1,jj,:]-f[0,jj,:])*(lu-1)
        df[0,-1,jj,:]=(f[-1,jj,:]-f[-2,jj,:])*(lu-1)
    for ii in range(1,lu-1):
        for jj in range(lv):
            df[0,ii,jj,:]=(f[ii+1,jj,:]-f[ii-1,jj,:])*(lu-1)/2
    for ii in range(lu):
        df[1,ii,0,:]=(f[ii,1,:]-f[ii,0,:])*(lv-1)
        df[1,ii,-1,:]=(f[ii,-2,:]-f[ii,-1,:])*(lv-1)
    for jj in range(1,lv-1):
        for ii in range(lu):
            df[1,ii,jj,:]=(f[ii,jj+1,:]-f[ii,jj-1,:])*(lv-1)/2
            
    #gradf=np.einsum('ij...,j...->i...',g_upper,df)
    gradf=np.zeros((2,lu,lv,lc))
    gradf[0]=np.expand_dims(g_upper[0,0],axis=2)*df[0]+np.expand_dims(g_upper[0,1],axis=2)*df[1]
    gradf[1]=np.expand_dims(g_upper[1,0],axis=2)*df[0]+np.expand_dims(g_upper[1,1],axis=2)*df[1]
    P3=np.zeros((lu,lv,lc,3))
    for k in range(3):
        P3[:,:,:,k]+=np.expand_dims(iymx,axis=2)*(gradf[0,:,:,:]*np.expand_dims(dpsidu[k],axis=2)+gradf[1,:,:,:]*np.expand_dims(dpsidv[k],axis=2))# lu x lv x 3
    P6=np.zeros((lu,lv,lc,3))
    P6[:,:,:,0]=f*np.expand_dims(iymx*div_xe_uv[0],axis=2)
    P6[:,:,:,1]=f*np.expand_dims(iymx*div_xe_uv[1],axis=2)
    P6[:,:,:,2]=f*np.expand_dims(iymx*div_xe_uv[2],axis=2)
    ymx=np.zeros((3,lu,lv))
    ymx[0]=newY[0]-Px
    ymx[1]=newY[1]-Py
    ymx[2]=newY[2]-Pz
    K=iymx**3*np.sum(ymx*surf_n,axis=0)# K=<y-x,n(x)> /|y-x|^3
    c=K*np.sum(j1ya*surf_normal,axis=2)
    #P4=np.einsum('...,...k->...k',c,j2)
    P4=np.expand_dims(np.expand_dims(c,axis=2),axis=3)*djdc_r
    #P5=-1*np.einsum('...,...k->...k',f*K,surf_normal)
    P5=-1*np.expand_dims(f*np.expand_dims(K,axis=2),axis=3)*np.expand_dims(surf_normal,axis=2)
    S=np.expand_dims(np.expand_dims(dS[:-1,:-1],axis=2),axis=3)*(P1+P2+P3+P4+P5+P6)[:-1,:-1,:,:]
    res_p=1e-7*np.sum(S,axis=1)/((lu-1)*(lv-1))
    return P1,P2,P3,P4,P5,P6

# FRANK: HERE!!!
@numba.njit(fastmath=fastmathbool,cache=True)
def aux(rot_l,Px,Py,Pz,theta,zeta,l,div_pi_xjy_uv,div_xe_uv,g_upper,dpsidu,dpsidv,surf_n,surf_normal, dS,j2,param,pi_xjy_uv,dj2,j1ya):
    (lu,lv,d1,d2)=param
    newY=np.dot(rot_l,np.array([Px[theta,zeta],Py[theta,zeta],Pz[theta,zeta]]))
    d2=(newY[0]-Px)**2+(newY[1]-Py)**2+(newY[2]-Pz)**2
    if l==0:
        d2[theta,zeta]=1# for the division
    elif zeta==0 and l==1:
        d2[theta,-1]=1
        if theta==0: d2[-1,-1]=1
    iymx=1/np.sqrt(d2)
    if l==0:
        iymx[theta,zeta]=0#by convention
    P1=-1*np.expand_dims(iymx*div_pi_xjy_uv,axis=2)*j2
    P2=np.zeros((lu,lv,3))
    for k in range(2):
        P2[:,:,0]+=pi_xjy_uv[:,:,k]*dj2[0][k]
        P2[:,:,1]+=pi_xjy_uv[:,:,k]*dj2[1][k]
        P2[:,:,2]+=pi_xjy_uv[:,:,k]*dj2[2][k]
    P2*=-1*np.expand_dims(iymx,axis=2)
    #1/(y-x) \nabla <j1(y) j2(x) >
    f=np.sum(j1ya*j2,axis=2)
    df=np.zeros((2,lu,lv))
    for k in range(2):
        for l in range(3):
            df[k]+=j1ya[:,:,l]*dj2[l,k,:,:]
    #df=0*df
    #gradf=np.einsum('ij...,j...->i...',g_upper,df)
    gradf=np.zeros((2,lu,lv))
    gradf[0]=g_upper[0,0]*df[0]+g_upper[0,1]*df[1]
    gradf[1]=g_upper[1,0]*df[0]+g_upper[1,1]*df[1]
    P3=np.zeros((lu,lv,3))
    for k in range(3):
        P3[:,:,k]+=iymx*(gradf[0,:,:]*dpsidu[k]+gradf[1,:,:]*dpsidv[k])# lu x lv x 3
    P6=np.zeros((lu,lv,3))
    P6[:,:,0]=iymx*f*div_xe_uv[0]
    P6[:,:,1]=iymx*f*div_xe_uv[1]
    P6[:,:,2]=iymx*f*div_xe_uv[2]
    ymx=np.zeros((3,lu,lv))
    ymx[0]=newY[0]-Px
    ymx[1]=newY[1]-Py
    ymx[2]=newY[2]-Pz
    K=iymx**3*np.sum(ymx*surf_n,axis=0)# K=<y-x,n(x)> /|y-x|^3
    c=K*np.sum(j1ya*surf_normal,axis=2)
    #P4=np.einsum('...,...k->...k',c,j2)
    P4=np.expand_dims(c,axis=2)*j2
    #P5=-1*np.einsum('...,...k->...k',f*K,surf_normal)
    P5=-1*np.expand_dims(f*K,axis=2)*surf_normal
    S=np.expand_dims(dS[:-1,:-1],axis=2)*(P1+P2+P3+P4+P5+P6)[:-1,:-1,:]
    res_p=1e-7*np.sum(S,axis=1)/((lu-1)*(lv-1))
    return np.sum(res_p,axis=0)
if __name__=='__main__':
    lu,lv=64+1,64+1
    path_plasma='code/li383/plasma_surf.txt'
    path_cws='code/li383/cws.txt'
    path_bnorm='code/li383/bnorm.txt'
    m,n=4,4
    Phisize=(m,n)
    cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(lu,lv),Np=3)
    #avg5=avg_laplace_force.Avg_laplace_force(cws5)
    djdc=get_djdc_naif(Phisize,cws.dpsidu,cws.dpsidv,cws.dS,cws.grid)
    avg=Avg_laplace_force(cws)

    I,G=1e6,2e5
    np.random.seed(987)
    l=2* (m*(2*n+1)+n)
    lst_coeff=1e3*(2*np.random.random(l)-1)/(np.arange(1,l+1)**2)
    C,S=Div_free_vector_field_on_TS.array_coeff_to_CS(lst_coeff,(m,n))
    coeff=(G,I,C,S)
    avg.grad_2_f_laplace(lu-1,lv-1,coeff,djdc)
