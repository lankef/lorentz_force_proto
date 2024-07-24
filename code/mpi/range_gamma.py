import subprocess
import numpy as np
import configparser
import shlex

ngamma=7
gamma_max=13 #10^(-gamma)
gamma_min=19
lst_gamma=10**(-np.linspace(gamma_min,gamma_max,ngamma))
config = configparser.ConfigParser()
config.read('code/mpi/config.ini')
for k in range(ngamma):
    config['other']['path_output']='Output/output_high{}'.format(k)
    config['other']['lamb_init']=str(lst_gamma[k])
    config['other']['gamma']=str(lst_gamma[k])
    config_file='code/mpi/config{}.ini'.format(k)
    with open(config_file, 'w') as configfile:
       config.write(configfile)
    t=shlex.split('mpiexec -np 65 python3 code/mpi/full_optimization_mpi.py {}'.format(config_file,k))
    print(t)
    #subprocess.run('mpiexec -np 65 python3 code/mpi/full_optimization_mpi.py {} > fichier_sortie_{}.log 2>&1'.format(config_file,k))
    log = open('log{}'.format(k), "w")
    err = open('err{}'.format(k), "w")
    subprocess.run(t,stdout=log, stderr=err)
    log.close()
    err.close()
