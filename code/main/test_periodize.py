import unittest
import numpy as np
from periodize import *
import toroidal_surface
class Test_periodize(unittest.TestCase):
    def test_periodize_vectorfield(self):
        lu,lv=17,19
        torus=toroidal_surface.Toroidal_surface(radii=(1,5),nbpts=(lu,lv))
        X,Y,Z=torus.X,torus.Y,torus.Z
        d1,d2=(1/(X.shape[0]-1),1/(X.shape[1]-1))# to normalize the gradient
        dx=np.gradient(X,d1,d2)
        dy=np.gradient(Y,d1,d2)
        dz=np.gradient(Z,d1,d2)
        dpsiu=np.zeros((lu,lv,3))
        dpsiu[:,:,0],dpsiu[:,:,1],dpsiu[:,:,2]=dx[0],dy[0],dz[0]
        periodize_vectorfield(dpsiu,torus.Np)
        dpsiv=np.zeros((lu,lv,3))
        dpsiv[:,:,0],dpsiv[:,:,1],dpsiv[:,:,2]=dx[1],dy[1],dz[1]
        periodize_vectorfield(dpsiv,torus.Np)
        np.testing.assert_almost_equal(dpsiu,np.moveaxis(torus.dpsidu,0,2))
        np.testing.assert_almost_equal(dpsiv,np.moveaxis(torus.dpsidv,0,2))

if __name__=='__main__':
    unittest.main()