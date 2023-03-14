import numpy as np
import matplotlib.pyplot as plt
import sys
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

def zad3(M,n,K):
    hbar=2*np.pi/M
    base=np.arange(0,M)*2*np.pi/M
    base=np.random.rand(n).reshape(-1,1)*2*np.pi+base
    #print(base.shape)
    V=np.exp(-1j/(hbar)*K*np.cos(base))
    P=np.exp(-1j/(2*hbar)*base**2)
    print(V.shape,P.shape)
    H=np.zeros([n,M,M],dtype=np.complex64)
    for i in range(M):
        fun=np.zeros([n,M],dtype=np.complex64)
        fun[:,i]=1
        fun*=V
        funp=np.fft.fft(fun)
        #funp/=np.linalg.norm(funp)
        funp*=P
        fun=np.fft.ifft(funp)
        #print(np.linalg.norm(fun))
        fun/=np.linalg.norm(fun)
        H[:,i,:]=fun
    #print(np.linalg.norm(H))
    eigen,_=np.linalg.eig(H)
    eigen=np.sort(eigen)
    eigen=eigen[:,M//4:M//4*3]
    diff=np.diff(eigen)
    diff/=np.mean(diff)
    fig=plt.figure(figsize=(12,8))
    ax=plt.axes()
    ax.hist(diff.flatten(),bins=50,density=True)
    ax.set_title("$K={}$,$M={}$,$n={}$".format(K,M,n))
    plt.savefig("zad3_{}_{}_{}.png".format(M,n,K))

zad3(100,20,5)
zad3(100,20,1)
sys.exit()
for s in ((6,20000),(20,10000),(200,500)):
    zad1(*s)
zad2(200,500)
zad2(200,500,"gue")
zad2(8,10000)
zad2(8,10000,"gue")
