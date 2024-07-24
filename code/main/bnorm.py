import numpy as np
def get_bnorm(pathfile,plasma):
        data=[]
        with open(pathfile,'r') as f:
            for line in f:
                data.append(str.split(line))
        adata=np.array(data,dtype='float64')
        m,n,bmn=adata[:,0],adata[:,1],adata[:,2]
        bnorm=np.zeros((plasma.grid[0]).shape)
        for i in range(len(m)):
            bnorm+=bmn[i]*np.sin(2*np.pi*m[i]*plasma.grid[0]+2*np.pi*n[i]*plasma.grid[1])
        return bnorm
if __name__=='__main__':
    from toroidal_surface import *
    plasma=Toroidal_surface(W7x_pathfile='code/li383/plasma_surf.txt',nbpts=(34,31))
    pathfile='code/li383/bnorm.txt'
    b=get_bnorm(pathfile,plasma)