import matplotlib.pyplot as plt
import numpy as np

def get_f(c0,c1,lu,lv,lc):
    def f_e(x):
        if 0 <=x and x <=c1:
            return np.max((x-c0,0))**2/(1- np.max((x-c0,0))/(c1-c0))
        else:
            raise Exception('infinite cost')
    def grad_f_e(x):
        if 0 <=x and x <=c1:
            if x<c0:
                return 0
            else:
                c=c1-c0
                y=x-c0
                return (c*y*(2*c - y))/((c - y)**2)
        else:
            raise Exception('infinite cost in gradient')
    def full_fe(X,dS):
        res=0
        for i in range(lu-1):
            for j in range(lv-1):
                res+=f_e(X[i,j])*dS[i,j]
        return res/((lu-1)*(lv-1))
    def full_grad(X,dX,dS):
        """from X : lu x lv  and dX : lu x lv x lc
        return df_e_X(dX)"""
        res=np.zeros(lc)
        for i in range(lu-1):
            for j in range(lv-1):
                for k in range(lc):
                    res[k]+=grad_f_e(X[i,j])*dX[i,j,k]*dS[i,j]
        return res/((lu-1)*(lv-1))
    return f_e,grad_f_e
if __name__ == "__main__":
    c0=1
    c1=2
    f,df=get_f(c0,c1,2,2,1)
    X=np.linspace(0,1.90,30)
    print(X[0])
    Y=[f(x) for x in X]
    dY=[df(x) for x in X]
    dYn=np.gradient(Y,X[1]-X[0])
    print(X,Y)
    plt.rcParams.update({'font.size': 16})
    ax=plt.axes()
    
    ax.plot(X,Y,label='$f_e$')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels([0,'$c_0$','$c_1$'])
    ax.legend()
    plt.show()
    ax=plt.axes()
    ax.plot(X,dY)
    ax.plot(X,dYn)
    plt.show()