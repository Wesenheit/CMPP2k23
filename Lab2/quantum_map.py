import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba

def evolve(position,K):
    position[1]=position[1]+K*np.sin(position[0])
    position[0]=np.mod(np.sum(position),2*np.pi)
@numba.jit
def get_wave(base,x0,p0,d,M):
    hbar=2*np.pi/M
    base=np.arange(0,M)*2*np.pi/M
    psi=np.zeros(M,dtype=np.complex64)
    for dp in range(-d,d):
        psi+=np.exp(1j*p0*base/hbar)*np.exp(-(base-x0+2*np.pi*dp)**2/(2*hbar))
    psi=psi/np.linalg.norm(psi)
    return psi

def zad1(M,x0,p0,d=10):
    hbar=2*np.pi/M
    base=np.arange(0,M)*2*np.pi/M
    psi=np.zeros(M,dtype=np.complex64)
    print(d)
    for dp in range(-d,d):
        psi+=np.exp(1j*p0*base/hbar)*np.exp(-(base-x0+2*np.pi*dp)**2/(2*hbar))
    psi=psi/np.linalg.norm(psi)
    fig,[ax1,ax2]=plt.subplots(2,1,figsize=(10,8))
    arr_1=np.conjugate(psi)*psi
    ax1.plot(base,arr_1,label="wavepacket")
    ax1.vlines(x=x0,ymin=np.min(arr_1),ymax=np.max(arr_1),label="$x_0$")   
    psit=np.fft.fft(psi)
    psit=psit/np.linalg.norm(psit)
    #print(psit)
    arr_2=np.conjugate(psit)*psit
    ax2.plot(base,arr_2,label="wavepacket")
    ax2.vlines(x=p0,ymin=np.min(arr_2),ymax=np.max(arr_2),label="$p_0$")  
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$\psi$")
    ax2.set_xlabel("$p$")
    ax2.set_ylabel("$\tilde{\psi}$")
    ax1.set_title("Space image for gaussian packet $x_0={}$, $p_0={}$".format(x0,p0))
    ax2.set_title("Momentum image")
    plt.tight_layout()
    plt.savefig("zad1.png")
#    plt.show()

def zad2(M,N,K,x0,p0,d=10):
    hbar=2*np.pi/M
    base=np.arange(0,M)*2*np.pi/M
    psi=np.zeros(M,dtype=np.complex64)
    print(d)
    for dp in range(-d,d):
        psi+=np.exp(1j*p0*base/hbar)*np.exp(-(base-x0+2*np.pi*dp)**2/(2*hbar))
    psi=psi/np.linalg.norm(psi)
    store=np.zeros([N,M,2])
    V=np.exp(-1j/(hbar)*K*np.cos(base))
    #print(V)
    P=np.exp(-1j/(2*hbar)*base**2)
    #print(P)
    #print(psi)
    position=np.array([x0,p0])
    clasical=np.zeros([N,2])
    for j in range(N):
        print(np.linalg.norm(psi))
        psi=psi*V
        psit=np.fft.fft(psi)
        psit=psit*P
        psi=np.fft.ifft(psit)
        psi/=np.linalg.norm(psi)
        psit/=np.linalg.norm(psit)
        store[j,:,0]=np.conjugate(psi)*psi
        store[j,:,1]=np.conjugate(psit)*psit
        evolve(position,K)
        clasical[j,:]=position.copy()
    fig,[ax1,ax2]=plt.subplots(2,1,figsize=(10,8))
   
    def callable(i):
        print(i)
        ax1.clear()
        ax2.clear()
        ax1.plot(base,store[i,:,0],label="quantum")
        ax2.plot(base,store[i,:,1],label="quantum")
        ax1.vlines(clasical[i,0],0,0.2,label="classical")
        ax2.vlines(clasical[i,1],0,0.2,label="classical")
        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$\psi$")
        ax2.set_xlabel("$p$")
        ax2.set_ylabel("$\tilde{\psi}$")
        ax1.set_title("Space image for gaussian packet $x_0={}$, $p_0={}$,$K={}$".format(x0,p0,K))
        ax2.set_title("Momentum image")
        ax1.set_ylim(0,0.2)
        ax2.set_ylim(0,0.2)
        ax1.set_xlim(0,2*np.pi)
        ax2.set_xlim(0,2*np.pi)

    ani=FuncAnimation(fig,func=callable,frames=N)
    ani.save("zad2_{}.mp4".format(K))

def zad3(M,N,K,x0,p0,d=10):
    hbar=2*np.pi/M
    base=np.arange(0,M)*2*np.pi/M
    psi=np.zeros(M,dtype=np.complex64)
    print(d)
    for dp in range(-d,d):
        psi+=np.exp(1j*p0*base/hbar)*np.exp(-(base-x0+2*np.pi*dp)**2/(2*hbar))
    psi=psi/np.linalg.norm(psi)
    store=np.zeros([N,M,2],dtype=np.complex64)
    V=np.exp(-1j/(hbar)*K*np.cos(base))
    P=np.exp(-1j/(2*hbar)*base**2)
    for j in range(N):
        psi=psi*V
        psit=np.fft.fft(psi)
        psit=psit*P
        psi=np.fft.ifft(psit)
        psi/=np.linalg.norm(psi)
        psit/=np.linalg.norm(psit)
        store[j,:,0]=psi.copy()
        store[j,:,1]=psit.copy()

    fig=plt.figure(figsize=(10,8))
    ax=plt.gca()
   
    def callable(i):
        print(i)
        Q=np.zeros([M,M])
        for a in range(M):
            for b in range(M):
                Q[a,b]=np.abs(np.sum(np.conjugate(get_wave(base,base[a],base[b],d=10,M=M))*store[i,:,0]))
        ax.imshow(Q.T)
        ax.set_title("$Q(x,p)$ $x_0={}$, $p_0={}$, $K={}$".format(x0,p0,K))

    ani=FuncAnimation(fig,func=callable,frames=np.arange(0,N))
    ani.save("zad3_{}.mp4".format(K))
#zad1(100,0.1,2*np.pi/100*40)
#zad2(100,200,5,3,2*np.pi/100*3)
zad3(100,20,2.5,1.1,2*np.pi/100*3)