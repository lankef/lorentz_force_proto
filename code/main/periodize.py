import numpy as np
def periodize_function(array):
    """from a function :lu x lv, average the boundary"""
    array[:,0],array[:,-1]= (array[:,0]+array[:,-1])/2 , (array[:,0]+array[:,-1])/2#for periodic boundary
    array[0,:],array[-1,:]= (array[0,:]+array[-1,:])/2 , (array[0,:]+array[-1,:])/2#for periodic boundary
def periodize_vectorfield(array,Np):
    """from a vectorfield with Np-periodicity along the z axis, average the boundary vectors"""
    
    rot=np.array([[np.cos(2*np.pi/Np),-np.sin(2*np.pi/Np),0],[np.sin(2*np.pi/Np),np.cos(2*np.pi/Np),0],[0,0,1]])#rotation matrix

    array[:,0,:],array[:,-1,:]= (array[:,0,:]+np.tensordot(array[:,-1,:],rot ,axes=(1,0)))/2 , (np.tensordot(array[:,0,:],rot ,axes=(1,1))+array[:,-1,:])/2#for the v part, we have periodicity up to a rotation
    array[0,:,:],array[-1,:,:]= (array[0,:,:]+array[-1,:,:])/2 , (array[0,:,:]+array[-1,:,:])/2#for the u part, we have exact periodicity