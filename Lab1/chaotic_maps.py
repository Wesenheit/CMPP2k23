import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import *
import numba

def asSpherical(xyz):
    x       = xyz[:,0]
    y       = xyz[:,1]
    z       = xyz[:,2]
    r       =  np.sqrt(x**2 + y**2 + z**2)
    theta   =  np.arccos(z/r)#*180/ np.pi 
    phi     =  np.arctan2(y,x)#*180/ np.pi
    return [r,theta,phi]

def evolve(position,K):
    position[1]=position[1]+K*np.sin(position[0])
    position[0]=np.mod(np.sum(position),2*np.pi)

def evolve_para(position,K):
    position[:,1]=position[:,1]+K*np.sin(position[:,0])
    position[:,0]=np.mod(np.sum(position,axis=1),2*np.pi)

@numba.jit
def evolve_para_extra(t,position,k):
    odp=np.zeros(position.shape)
    odp[:,0]=position[:,2]*np.cos(position[:,0]*k)+position[:,1]*np.sin(k*position[:,0])
    odp[:,1]=position[:,1]*np.cos(position[:,0]*k)-position[:,2]*np.sin(k*position[:,0])
    odp[:,2]=-position[:,0]
    return odp

@numba.jit
def evolve_extra(t,position,k):
    odp=np.zeros(position.shape)
    odp[0]=position[2]*np.cos(position[0]*k)+position[1]*np.sin(k*position[0])
    odp[1]=position[1]*np.cos(position[0]*k)-position[2]*np.sin(k*position[0])
    odp[2]=-position[0]
    return odp
def zad1a(K,N):
    position1=np.array([3,1.9])
    position2=np.array([3,1.8999])
    array1=np.zeros([N,2])
    array2=np.zeros([N,2])
    for i in range(N):
        array1[i]=position1
        array2[i]=position2
        evolve(position1,K)
        evolve(position2,K)
    fig,[ax1,ax2]=plt.subplots(2,1,figsize=(10,8))
    ax1.plot(array1[:,0],label="$x$",marker=".")
    ax1.plot(array1[:,1],label="$p$",marker=".")
    ax1.set_title("$x_0=3$ $p_0=1.9$")
    ax2.plot(array2[:,0],label="$x$",marker=".")
    ax2.plot(array2[:,1],label="$p$",marker=".")
    ax2.set_title("$x_0=3$ $p_0=1.8999$")
    ax1.set_xlabel(r"$n$")
    ax2.set_xlabel(r"$n$")
    ax1.set_ylabel(r"Amplitude")
    ax2.set_ylabel(r"Amplitude")
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.suptitle("Standard map for $K={}$".format(K))
    plt.tight_layout()
    plt.savefig("plot_1.png")


def zad2(K,N,num):
    positions=np.random.rand(num,2)*2*np.pi
    store=np.zeros([N,num,2])
    rgb=np.random.rand(num,3) 
    for i in range(N):
        store[i]=positions
        evolve_para(positions,K)
    fig=plt.figure(figsize=(10,8))
    ax=plt.gca()
    for j in range(num):
        ax.scatter(store[:,j,0],np.mod(store[:,j,1],2*np.pi),color=rgb[j],marker=".")
    ax.grid()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p$")
    plt.suptitle("Standard map for $K={}$".format(K))
    plt.savefig("zad2_{}.png".format(K))

@numba.jit
def integrate(position,K,N,store,dt,ile=100):
    for i in range(N):
        for _ in range(ile):
            der=evolve_para_extra(0,position,K)
            position+=der*dt
        store[i,:,:]=position
def zad3(k,N,num):
    positions=np.random.rand(num,2)*2*np.pi
    positions=np.concatenate(((np.sin(positions[:,0])*np.cos(positions[:,1])).reshape(-1,1),(np.sin(positions[:,0])*np.sin(positions[:,1])).reshape(-1,1),np.cos(positions[:,0]).reshape(-1,1)),axis=1)
    store=np.zeros([N,num,3])
    rgb=np.random.rand(num,3) 
    #print(positions.shape)
    #print(positions)
    integrate(positions,K,N,store,0.001)
    fig=plt.figure(figsize=(10,8))
    ax=plt.gca()
    #print(positions)
    for j in range(num):
        r,theta,phi=asSpherical(store[:,j,:])
        print(r)
        ax.scatter(phi,theta,color=rgb[j],marker=".")
    ax.grid()
    plt.suptitle("Kicked top map for $K={}$".format(K))
    plt.savefig("zad3_{}.png".format(K))


zad1a(1.2,50)
for K in (1.2,2.1,5.5):
    zad2(K,1000,100)

#for K in (1,2,3,6):
#    zad3(K,10000,100)