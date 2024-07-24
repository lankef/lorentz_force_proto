import numpy as np
from mayavi import mlab
import toroidal_surface
import logging
class Magnetic_field():
    def __init__(self,W7x_pathfile=None,data=None,pB=None):
        if not W7x_pathfile is None:
            logging.info('initialization of the magnetic field from W7x data')
            self.load_W7x(W7x_pathfile)
        elif not data is None:
            logging.info('initialization of a magnetic field from explicit data')
            self.X,self.Y,self.Z=data[0,:],data[1,:],data[2,:]
            self.Bx,self.By,self.Bz=data[3,:],data[4,:],data[5,:]
        elif not pB is None:
            pos,B=pB
            self.X,self.Y,self.Z=pos[:,0],pos[:,1],pos[:,2]
            self.Bx,self.By,self.Bz=B[0,:],B[1,:],B[2,:]

    def load_W7x(self,pathfile):
        """load the file pathfile in the format given by W7X"""
        data=[]
        with open(pathfile,'r') as f:
            for line in f:
                data.append(str.split(line))
        data=np.array(data[2:],dtype=float)# we delete the first 2 lines of the file
        self.R,self.phi=data[:,0],data[:,1]
        self.X,self.Y,self.Z=data[:,2],data[:,3],data[:,4]
        self.Bx,self.By,self.Bz=data[:,5],data[:,6],data[:,7]
    def select_close_to_axis(self,Rg,Zg,d):
        """select only the magnetic field component at a distance <d to the Rg Zg component"""
        l=np.linspace(0,2*np.pi/5,len(Rg))
        Rgi=np.interp(self.phi,l,Rg)#we interpolate Rg on the self.phi scale
        Zgi=np.interp(self.phi,l,Zg)#we interpolate Zg on the self.phi scale
        sel= ((Rgi-self.R)**2+(Zgi-self.Z)**2<d)
        return sel
    def plot(self,sel=None,npts=4e5):
        if sel==None:
            sel=np.ones(len(self.X),dtype=bool)#no selection
        X,Y,Z=self.X[sel],self.Y[sel],self.Z[sel]
        Bx,By,Bz= self.Bx[sel],self.By[sel],self.Bz[sel]
        print('number of magnetic components :'+str(len(Bx)))
        qv=mlab.quiver3d(X,Y,Z,Bx,By,Bz,scalars=np.sqrt(Bx**2+By**2+Bz**2),mask_points=max(len(Bx)//npts,1))
        mlab.colorbar(qv)
        return(qv)
    def select_magnetic_field(self,sel):
        X,Y,Z=self.X[sel],self.Y[sel],self.Z[sel]
        Bx,By,Bz= self.Bx[sel],self.By[sel],self.Bz[sel]
        data=np.array([X,Y,Z,Bx,By,Bz])
        return Magnetic_field(data=data)

