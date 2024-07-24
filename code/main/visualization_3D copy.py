from mayavi import mlab
from regcoil import *
import avg_laplace_force
from tqdm import tqdm
from vector_field_on_TS import *
import pickle

f=open("Output/fe.tmp",'rb')
res=pickle.load(f)
f.close()

Np=3
ntheta_plasma = 64+1
ntheta_coil   = 64+1
nzeta_plasma = 64+1
nzeta_coil   = 64+1
mpol_coil  = 12 #probably not enough
ntor_coil  = 12 #probably not enough
net_poloidal_current_Amperes = 11884578.094260072/Np
net_toroidal_current_Amperes = 0.
lamb1=1.2e-14
lamb2=2.5e-16
lamb3=5.1e-19
Phisize=(ntor_coil,mpol_coil)

sol_C,sol_S=Div_free_vector_field_on_TS.array_coeff_to_CS(res,Phisize)

coeff=(net_poloidal_current_Amperes,net_toroidal_current_Amperes,sol_C,sol_S)

G,I=net_poloidal_current_Amperes,net_toroidal_current_Amperes
path_plasma='code/li383/plasma_surf.txt'
path_cws='code/li383/cws.txt'
path_bnorm='code/li383/bnorm.txt'
plasma_surf=toroidal_surface.Toroidal_surface(W7x_pathfile=path_plasma,nbpts=(ntheta_plasma,nzeta_plasma),Np=3)
cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(ntheta_plasma,nzeta_plasma),Np=3)
div_free=vector_field_on_TS.Div_free_vector_field_on_TS(cws)

eps=1e-4

#avg=avg_laplace_force.Avg_laplace_force(cws)

#array_eps=avg.L_eps_optimized(coeff,eps)
#laplace_array=avg.f_laplace_optimized(ntheta_coil-1,nzeta_coil-1,coeff,coeff)
#array=laplace_array
#print(np.max(np.linalg.norm(array,axis=2)))
#cr=4
#array_renorm=array/(cr*np.max(np.linalg.norm(array,axis=2)))
mlab.figure(bgcolor = (1,1,1))
cws.plot_surface(clm='Greens',half=True)
plasma_surf.plot_surface(half=True)
#cws.plot_3Dvector_field(np.moveaxis(array_renorm,2,0))
#cws.plot_function_on_surface(np.linalg.norm(array,axis=2))
mlab.show()