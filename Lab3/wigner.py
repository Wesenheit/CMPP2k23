import numpy as np
import matplotlib.pyplot as plt

def zad1(N,n):
    matrix=np.random.randn(n,N,N)
    matrix=(matrix+np.swapaxes(matrix,1,2))/2
    fig=plt.figure(figsize=(12,8))
    ax=plt.axes()
    eigen,_=np.linalg.eig(matrix)
    ax.hist(eigen.flatten(),bins=100,density=True,label="simulation")
    X=np.linspace(np.min(eigen.flatten()),np.max(eigen.flatten()),1000)
    ax.set_xlabel("$E$")
    R=np.sqrt(2*N)
    Y=2/(np.pi*R**2)*np.sqrt(R**2-X**2)
    ax.plot(X,Y,label="semicircle law")
    ax.set_title(r"$N={}$,$n_{{sample}}={}$".format(N,n))
    ax.legend()
    plt.savefig("zad1_{}_{}.png".format(N,n),dpi=500)

def zad2(N,n,kind="goe"):
    if kind=="goe":
        matrix=np.random.randn(n,N,N)
        matrix=(matrix+np.swapaxes(matrix,1,2))/2
    else:
        X=np.random.randn(n,N,N)
        Y=np.random.randn(n,N,N)
        matrix=(X+1j*Y)/np.sqrt(2)
        matrix=(matrix+np.conjugate(np.swapaxes(matrix,1,2)))/2
    eigen,_=np.linalg.eig(matrix)
    eigen=np.sort(eigen)
    if N>10:
        eigen=eigen[:,N//4:N//4*3]
    else:
        eien=eigen[:,N//2-1:N//2]
    diff=np.diff(eigen)
    diff/=np.mean(diff)
    fig=plt.figure(figsize=(12,8))
    ax=plt.axes()
    ax.hist(diff.flatten(),density=True,label="simulation",bins=100)
    X=np.linspace(np.min(diff),np.max(diff),1000)
    if kind=="goe":
        Y=np.pi/2*X*np.exp(-np.pi/4*X**2)
    else:
        Y=32/np.pi**2*X**2*np.exp(-4/np.pi*X**2)
    ax.plot(X,Y,label="theory")
    ax.legend()
    ax.set_title("{}: $N={}$,$n_{{sample}}={}$".format(kind,N,n))
    ax.set_xlabel("$s$")
    ax.set_ylabel("$P(s)$")
    plt.savefig("zad2_{}_{}_{}.png".format(kind,N,n),dpi=500)

for s in ((6,20000),(20,10000),(200,500)):
    zad1(*s)
zad2(200,500)
zad2(200,500,"gue")
zad2(8,10000)
zad2(8,10000,"gue")
