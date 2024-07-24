import numpy as np
#from mayavi import mlab
import time
import logging
import mytimer
class Toroidal_surface():
    sat=1e8
    @mytimer.timeit
    def __init__(self,W7x_pathfile=None,nbpts=(16,16),pos=None,radii=None,flat_torus=False,Np=5):
        """pathfile : path to W7X datafile
        pos : X,Y,Z of the toroidal surface
        radii : 2-tuple R,r"""
        self.Np=Np
        grad=None
        if not W7x_pathfile is None:
            logging.debug('creation of a Toroidal surface with data from W7X')
            grad=self.load_W7X(W7x_pathfile,nbpts)
        elif not pos is None:
            logging.debug('creation of a Toroidal surface with explicit points location')
            self.X,self.Y,self.Z=pos
        elif not radii is None:
            logging.debug('creation of a Toroidal surface as a Torus')
            grad=self.generate_torus(radii,nbpts)
        elif flat_torus:
            self.generate_flat_torus(nbpts)
        else:
            raise Exception('Not implemented')
        
        #We initialize the grid if it has not already been done
        u,v=np.linspace(0, 1, self.X.shape[0]),np.linspace(0, 1, self.X.shape[1])
        self.grid=np.meshgrid(u,v,indexing='ij')
        #We compute the normal vector field
        self.compute_grad_and_normal(grad)
        self.lu,self.lv=self.X.shape
        self.dim=len(self.X[:-1,:-1].flatten())# nb of degree of freedom
        logging.debug('creation of a Toroidal surface successfull')

    def load_W7X(self,pathfile,nbpts):
        #load the file in the format given by W7X
        lu,lv=nbpts
        data=[]
        with open(pathfile,'r') as f:
            for line in f:
                #print(line)
                #print(str.split(line))
                data.append(str.split(line))
        adata=np.array(data,dtype='float64')
        m,n,Rmn,Zmn=adata[:,0],adata[:,1],adata[:,2],adata[:,3]
        #m,n,Rmn,Zmn=adata[:30,0],adata[:30,1],adata[:30,2],adata[:30,3]
        #nmax,mmax=int(max(n)),int(max(m))
        u,v=np.linspace(0, 1, nbpts[0]),np.linspace(0, 1, nbpts[1])
        ugrid,vgrid=np.meshgrid(u,v,indexing='ij')
        R=np.zeros(ugrid.shape)
        Z=np.zeros(ugrid.shape)
        #first derivative
        dpsi_u=np.zeros((3,lu,lv))
        dpsi_v=np.zeros((3,lu,lv))
        #second derivative
        dpsi_uu=np.zeros((3,lu,lv))
        dpsi_uv=np.zeros((3,lu,lv))
        dpsi_vv=np.zeros((3,lu,lv))
        dRdu=np.zeros((lu,lv))
        dRdv=np.zeros((lu,lv))

        phi = 2*np.pi/self.Np*vgrid
#        for i in range(len(u)):
#            for j in range(len(v)):
#                for k in range(len(n)):
#                    R[j,i]+=Rmn[k]*np.cos(2*np.pi*(m[k]*u[i]-n[k]*v[j]))
#                    Z[j,i]+=Zmn[k]*np.sin(2*np.pi*(m[k]*u[i]+n[k]*v[j]))
        for sa in np.array_split(np.arange(len(m)), max(int(len(u)*len(v)*len(m)/Toroidal_surface.sat),1)): # to avoid memory saturation
            tmp=np.tensordot(m[sa],ugrid,0)+np.tensordot(n[sa],vgrid,0)# m*u+n*v#            
            R+=np.tensordot(Rmn[sa],np.cos(2*np.pi*tmp),1)#sum_n,m(Rmn*np.cos(2*pi*(m*u+n*v))
            Z+=np.tensordot(Zmn[sa],np.sin(2*np.pi*tmp),1)#sum_n,m(Zmn*np.sin(2*pi*(m*u+n*v))
            #first derivative
            dpsi_u[0,:,:]+=np.tensordot(m[sa]*Rmn[sa],-2*np.pi*np.sin(2*np.pi*tmp),1)*np.cos(phi)#dR/du *cos(phi)=dX/du
            dpsi_u[1,:,:]+=np.tensordot(m[sa]*Rmn[sa],-2*np.pi*np.sin(2*np.pi*tmp),1)*np.sin(phi)#dR/du *sin(phi)=dY/du
            dpsi_u[2,:,:]+=np.tensordot(m[sa]*Zmn[sa],2*np.pi*np.cos(2*np.pi*tmp),1)#dZ/du
            dpsi_v[0,:,:]+=np.tensordot(n[sa]*Rmn[sa],-2*np.pi*np.sin(2*np.pi*tmp),1)*np.cos(phi)#dR/dv *cos(phi)
            dpsi_v[1,:,:]+=np.tensordot(n[sa]*Rmn[sa],-2*np.pi*np.sin(2*np.pi*tmp),1)*np.sin(phi)#dR/dv *sin(phi)
            dpsi_v[2,:,:]+=np.tensordot(n[sa]*Zmn[sa],2*np.pi*np.cos(2*np.pi*tmp),1)#dZ/dv
            #second derivative
            dpsi_uu[0,:,:]+=np.tensordot(m[sa]**2*Rmn[sa],-(2*np.pi)**2*np.cos(2*np.pi*tmp),1)*np.cos(phi)#d^2R/du^2 *cos(phi)=dX/du
            dpsi_uu[1,:,:]+=np.tensordot(m[sa]**2*Rmn[sa],-(2*np.pi)**2*np.cos(2*np.pi*tmp),1)*np.sin(phi)#d^2R/du^2 *sin(phi)=dY/du
            dpsi_uu[2,:,:]+=np.tensordot(m[sa]**2*Zmn[sa],-(2*np.pi)**2*np.sin(2*np.pi*tmp),1)#d^2Z/du^2
            dpsi_uv[0,:,:]+=np.tensordot(m[sa]*n[sa]*Rmn[sa],-(2*np.pi)**2*np.cos(2*np.pi*tmp),1)*np.cos(phi)#d^2R/dudv *cos(phi)
            dpsi_uv[1,:,:]+=np.tensordot(m[sa]*n[sa]*Rmn[sa],-(2*np.pi)**2*np.cos(2*np.pi*tmp),1)*np.sin(phi)#d^2R/dudv *sin(phi)
            dpsi_uv[2,:,:]+=np.tensordot(m[sa]*n[sa]*Zmn[sa],-(2*np.pi)**2*np.sin(2*np.pi*tmp),1)#d^2Z/dudv
            dpsi_vv[0,:,:]+=np.tensordot(n[sa]**2*Rmn[sa],-(2*np.pi)**2*np.cos(2*np.pi*tmp),1)*np.cos(phi)#d^2R/dv^2 *cos(phi)=dX/du
            dpsi_vv[1,:,:]+=np.tensordot(n[sa]**2*Rmn[sa],-(2*np.pi)**2*np.cos(2*np.pi*tmp),1)*np.sin(phi)#d^2R/dv^2 *sin(phi)=dY/du
            dpsi_vv[2,:,:]+=np.tensordot(n[sa]**2*Zmn[sa],-(2*np.pi)**2*np.sin(2*np.pi*tmp),1)#d^2Z/dv^2
            #other stuff
            dRdu+=np.tensordot(m[sa]*Rmn[sa],-2*np.pi*np.sin(2*np.pi*tmp),1)
            dRdv+=np.tensordot(n[sa]*Rmn[sa],-2*np.pi*np.sin(2*np.pi*tmp),1)
        self.R=R
        self.Z=Z
        # we generate X and Y
        self.X=R*np.cos(phi)
        self.Y=R*np.sin(phi)

        dpsi_v[0,:,:]+= -R*2*np.pi/self.Np*np.sin(phi)#R dcos(phi)/dv
        dpsi_v[1,:,:]+= R*2*np.pi/self.Np*np.cos(phi)#R dsin(phi)/dv
        dpsi_uv[0,:,:]+= -dRdu*2*np.pi/self.Np*np.sin(phi)#R dcos(phi)/dv
        dpsi_uv[1,:,:]+= dRdu*2*np.pi/self.Np*np.cos(phi)#R dsin(phi)/dv
        dpsi_vv[0,:,:]+= -R*(2*np.pi/self.Np)**2*np.cos(phi)#R d^2cos(phi)/dv^2
        dpsi_vv[1,:,:]+= -R*(2*np.pi/self.Np)**2*np.sin(phi)#R d^2sin(phi)/dv^2
        dpsi_vv[0,:,:]+= -2* dRdv*2*np.pi/self.Np*np.sin(phi)#R dcos(phi)/dv
        dpsi_vv[1,:,:]+= 2* dRdv*2*np.pi/self.Np*np.cos(phi)#R dsin(phi)/dv

        self.grid=(ugrid,vgrid) #save for expend_coarse_magnetic_axis=
        self.dpsi_uu=dpsi_uu
        self.dpsi_uv=dpsi_uv
        self.dpsi_vv=dpsi_vv
        #We also compute dS_u and dS_v:
        N=np.cross(dpsi_u,dpsi_v,0,0,0)
        dS=np.linalg.norm(N,axis=0)
        dNdu=np.cross(dpsi_uu,dpsi_v,0,0,0)+np.cross(dpsi_u,dpsi_uv,0,0,0)
        dNdv=np.cross(dpsi_uv,dpsi_v,0,0,0)+np.cross(dpsi_u,dpsi_vv,0,0,0)
        self.dS_u=np.sum(dNdu*N,axis=0)/dS
        self.dS_v=np.sum(dNdv*N,axis=0)/dS
        return np.array([[dpsi_u[0,:,:],dpsi_v[0,:,:]],[dpsi_u[1,:,:],dpsi_v[1,:,:]],[dpsi_u[2,:,:],dpsi_v[2,:,:]]])
    def generate_torus(self,radii,nbpts):
        R,r=radii
        Ntheta,Nphi=nbpts
        phi=np.linspace(0,2*np.pi/self.Np,Nphi)
        theta=np.linspace(0,2*np.pi,Ntheta)
        theta_grid,phi_grid=np.meshgrid(theta,phi,indexing='ij')
        self.X=(R+r*np.cos(theta_grid))*np.cos(phi_grid)
        self.Y=(R+r*np.cos(theta_grid))*np.sin(phi_grid)
        self.Z=r*np.sin(theta_grid)
        dpsi_u=np.array([-2*np.pi*r*np.sin(theta_grid)*np.cos(phi_grid),-2*np.pi*r*np.sin(theta_grid)*np.sin(phi_grid),r*2*np.pi*np.cos(theta_grid)])
        dpsi_v=np.array([(R+r*np.cos(theta_grid))*-2*np.pi/self.Np *np.sin(phi_grid),(R+r*np.cos(theta_grid))*2*np.pi/self.Np *np.cos(phi_grid),np.zeros(theta_grid.shape)])
        return np.array([[dpsi_u[0,:,:],dpsi_v[0,:,:]],[dpsi_u[1,:,:],dpsi_v[1,:,:]],[dpsi_u[2,:,:],dpsi_v[2,:,:]]])
    def generate_flat_torus(self,nbpts):
        lu,lv=nbpts
        X=np.linspace(-1,1,lu)
        Y=np.linspace(-1,1,lv)
        self.X,self.Y=np.meshgrid(X,Y,indexing='ij')
        self.Z=0.*self.X
        self.Np=1
    def grad(X,Y,Z,Np,circ=True):
        """implement numerical gradient with periodic boundary correction"""
        d1,d2=(1/(X.shape[0]-1),1/(X.shape[1]-1))# to normalize the gradient

        dx=np.gradient(X,d1,d2)
        dy=np.gradient(Y,d1,d2)
        dz=np.gradient(Z,d1,d2)
        # as the last and first component are identical up to a rotation, we correct the gradient
        if circ:
            rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])
            for i in range(2):
                vec=np.array([dx[i],dy[i],dz[i]])
                back_rotate=np.tensordot(rot,vec,axes=(0,0))
                direct_rotate=np.tensordot(rot,vec,axes=(1,0))
                for k in range(3):
                    d =([dx[i],dy[i],dz[i]])[k]
                    d[:,0],d[:,-1]= (d[:,0]+back_rotate[k,:,-1])/2 , (direct_rotate[k,:,0]+d[:,-1])/2
                    d[0,:],d[-1,:]= (d[0,:]+d[-1,:])/2 , (d[0,:]+d[-1,:])/2
                    #d[:,0],d[:,-1]= (d[:,0]+back_rotate[k,:,-1])/2 , (direct_rotate[k,:,0]+d[:,-1])/2
        return(dx,dy,dz)
    def compute_grad_and_normal(self,grad=None):
        """Compute the normal vector field"""
        if grad is None:
            dx,dy,dz=Toroidal_surface.grad(self.X,self.Y,self.Z,self.Np)
        else :
            dx,dy,dz=grad
        nn=1*np.array([dy[0]*dz[1]-dy[1]*dz[0],dz[0]*dx[1]-dz[1]*dx[0],dx[0]*dy[1]-dx[1]*dy[0]])
        normn=np.linalg.norm(nn,axis=0)
        self.dS=normn
        self.dpsidu=np.array([dx[0],dy[0],dz[0]])
        self.dpsidv=np.array([dx[1],dy[1],dz[1]])
        self.g_lower=np.array([[np.sum(self.dpsidu*self.dpsidu,0),np.sum(self.dpsidu*self.dpsidv,0)],[np.sum(self.dpsidu*self.dpsidv,0),np.sum(self.dpsidv*self.dpsidv,0)]])# the metric g_{i,j} to use against vector field
        self.g_upper=np.zeros((2,2,self.X.shape[0],self.X.shape[1]))# the metric g^{i,j}
        tmp_lower=np.moveaxis(self.g_lower,(0,1,2,3),(2,3,0,1))
        tmp_upper =np.linalg.inv(tmp_lower)
        #for i in range(self.X.shape[0]):
        #    for j in range(self.X.shape[1]):
        #        self.g_upper[:,:,i,j]=np.linalg.inv(self.g_lower[:,:,i,j])
        self.g_upper=np.moveaxis(tmp_upper,(0,1,2,3),(2,3,0,1))
        self.n=nn/normn # 3 x lu x lv
        self.normal=np.moveaxis(self.n,0,2) # lu x lv x3
    
    def plot_3Dvector_field(self,vf,c=1,step=1,half=False):
        """plot a 3D vector field on the surface"""
        from mayavi import mlab
        if half:
            l=len(self.X[0]-1)
            qv=mlab.quiver3d(self.X[:-1,:l//2], self.Y[:-1,:l//2], self.Z[:-1,:l//2], vf[0,:,:l//2], vf[1,:,:l//2], vf[2,:,:l//2], line_width=3, scale_factor=1)
        else:
            qv=mlab.quiver3d(self.X[:-1,:-1], self.Y[:-1,:-1], self.Z[:-1,:-1], vf[0,:,:], vf[1,:,:], vf[2,:,:], line_width=3, scale_factor=1)
        return(qv)
    def plot_surface(self,clm='Wistia',half=False):
        """Plot the surface given by X,Y,Z"""
        from mayavi import mlab
        if half:
            l=len(self.X[0])
            s = mlab.mesh(self.X[:,:l//2],self.Y[:,:l//2],self.Z[:,:l//2],representation='mesh',colormap=clm)
        else:
            s = mlab.mesh(self.X,self.Y,self.Z,representation='mesh',colormap=clm)
        return(s)
    def plot_function_on_surface(self,f,half=False):
        from mayavi import mlab
        """Plot f the surface given by self.X,self.Y,self.Z"""
        fc=np.concatenate((f,[f[0]]),axis=0)
        fc2=np.concatenate((fc[:,0:1],fc),axis=1)
        if half:
            l=len(self.X[0])
            s = mlab.mesh(self.X[:,:l//2],self.Y[:,:l//2],self.Z[:,:l//2],representation='mesh',scalars=fc2[:,:l//2])
        else:
            s = mlab.mesh(self.X,self.Y,self.Z,representation='mesh',scalars=fc2)
        mlab.colorbar(s,nb_labels=4,label_fmt='%.1E',orientation='vertical')
        return(s)
    def expend_surface(self,d):
        """create a new surface S+d*n"""
        Pos=(self.X-d*self.n[0,:,:],self.Y-d*self.n[1,:,:],self.Z-d*self.n[2,:,:])
        return Toroidal_surface(pos=Pos)
    def coarse_magnetic_axis(self):
        """compute a coarse approximation of the magnetic axis position as a function phi -> R,z"""
        new_R=np.average(self.R,1)
        new_Z=np.average(self.Z,1)
        return(new_R,new_Z)
    def expend_coarse_magnetic_axis(self,d):
        """return a cws at distance d of the coarse_magnetic_axis"""
        R,Z=self.coarse_magnetic_axis()
        (ugrid,vgrid)=self.grid
        ntheta=ugrid.shape[1]
        RR=np.tensordot(R,np.ones(ntheta),0)
        ZZ=np.tensordot(Z,np.ones(ntheta),0)
        X=(RR+d*np.cos(2*np.pi*ugrid))*np.cos(2*np.pi*vgrid/self.Np)
        Y=(RR+d*np.cos(2*np.pi*ugrid))*np.sin(2*np.pi*vgrid/self.Np)
        Z=ZZ+np.sin(2*np.pi*ugrid)
        return Toroidal_surface(pos=(X,Y,Z))

def expend_surface_smooth(surf,d,step=10):
    for i in range(step):
        surf=surf.expend_surface(d/step)# we do small step in order to avoid the destruction of the mesh structure
    return surf
if __name__=='__main__':
    from mayavi import mlab
    #T=Toroidal_surface(flat_torus=True)
    T=Toroidal_surface(W7x_pathfile='code/Wendelstein 7-X data/fourier.dat',nbpts=(34,31))
    T.plot_surface()
    T.plot_3Dvector_field(-1*T.n)
    mlab.show()