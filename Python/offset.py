import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import fsolve, minimize_scalar
from scipy.linalg import expm

def bisect(func_,a_,b_,iter_= 1000):
    NN = 10
    xa = a_
    xb = b_
    x = np.linspace(xa,xb,NN)
    for it in range(iter_):
        fx = np.array([func_(xx) for xx in x])
        # now find minimum value and location
        fminimum = np.min(fx)
        iminimum = np.argmin(fx)
        if iminimum == 0 or iminimum == NN-1:
            break
        x = np.linspace(x[iminimum-1],x[iminimum+1],NN)
            
    return x[iminimum]

def readKappa():
    Nx = 20
    kappa = np.zeros((Nx,Nx))
    with open("perylene.yml", 'r') as file:
        meta_yml = yaml.safe_load(file)
        adj = meta_yml['adjacency']
        hop = meta_yml['hopping']
        kappa = np.zeros((Nx,Nx))
        for x in range(len(adj)):
            y,z = adj[x]
            kappa[y,z] = hop
        kappa += kappa.T
    return kappa

def computeTangentPlane() -> float:
    r"""
        minimizes: S(phi_NLO) for phi_NLO
    """
    with open("twoSite.yml", 'r') as file:
        meta_yml = yaml.safe_load(file)

        Nt = meta_yml['system']['Nt']
        beta = meta_yml['system']['beta']
        U = meta_yml['system']['U']
        mu = -meta_yml['system']['mu']
        Nx = meta_yml['system']['nions']
        Nconf = meta_yml['HMC']['Nconf']

        adj = meta_yml['system']['adjacency']
        hop = meta_yml['system']['hopping']
        kappa = np.zeros((Nx,Nx))

        for x in range(len(adj)):
            y,z = adj[x]

            kappa[y,z] = hop
        kappa += kappa.T

    V = Nx*Nt
    delta = beta/Nt
    deltaU = U * delta
    deltaMu = mu * delta

    if deltaMu == 0:
        return 0

    epsilon, _ = np.linalg.eigh(kappa)


    def f(phi):
        LHS = -np.mean(
            np.tanh([.5*beta*(epsilon[i]+mu+phi) for i in range(Nx)])
        ) 

        return LHS - phi/U

    phi_cr = float(fsolve(f,0.0)[0] * delta)
    # Return the (imaginary) shift for the configuration
    return phi_cr 

def computeNLO():#(lat, params, action) -> float:
    r"""
        minimizes: S(phi_NLO) + 0.5 log det H_S(phi_NLO) for phi_NLO
    """

    with open("twoSite.yml", 'r') as file:
        meta_yml = yaml.safe_load(file)

        Nt = meta_yml['system']['Nt']
        beta = meta_yml['system']['beta']
        U = meta_yml['system']['U']
        mu = meta_yml['system']['mu']
        Nx = meta_yml['system']['nions']
        Nconf = meta_yml['HMC']['Nconf']

        adj = meta_yml['system']['adjacency']
        hop = meta_yml['system']['hopping']
        kappa = np.zeros((Nx,Nx))

        for x in range(len(adj)):
            y,z = adj[x]

            kappa[y,z] = hop
        kappa += kappa.T

    V = Nx*Nt
    delta = beta/Nt
    deltaU = U * delta
    deltaMu = mu * delta

    if deltaMu == 0:
        return 0

    epsilon, U = np.linalg.eigh(kappa)

    Udag = U
    U = np.transpose(U.copy())
    
    deltaEpsilon = epsilon*delta

    expKp = np.exp( deltaEpsilon)
    expKm = np.exp(-deltaEpsilon)

    def wn(n_):
        return 2 * np.pi * (n_+.5) / Nt

    def MM(phi):
        cosh2 = -Udag @ np.diag( 2*np.cosh(deltaEpsilon - 1j*phi + deltaMu) ) @ U
         
        answer = np.eye(V,k=0) + np.eye(V,k=-Nx*2) - np.eye(V,k=V-Nx*2)+0j

        for t in range(Nt-1):
            for x1 in range(Nx):
                xt1 = Nx*(t+1)+x1
                for x2 in range(Nx):
                    xt2 = Nx*t+x2

                    answer[xt1][xt2] = cosh2[x1][x2]

        # anti-periodic term
        for x1 in range(Nx):
            xt1 = x1
            for x2 in range(Nx):
                xt2 = Nx*(Nt-1)+x2

                answer[xt1][xt2]=-cosh2[x1][x2]

        return answer

    def trMm2(phi,pm_):

        NN=[]
        for n in range(-int(Nt/2),int(Nt/2)):
            if pm_==1:
                dd = np.array([expKp[lam]*np.exp(1j*wn(n)-1j*phi+deltaMu)/(1-expKp[lam]*np.exp(1j*wn(n)-1j*phi+deltaMu)) for lam in range(Nx)])
            else:
                dd = np.array([expKm[lam]*np.exp(1j*wn(n)+1j*phi-deltaMu)/(1-expKm[lam]*np.exp(1j*wn(n)+1j*phi-deltaMu)) for lam in range(Nx)])
            NN.append(Udag @ np.diag(dd) @ U)
        NN = np.array(NN)
        
        answer = np.zeros((V,V))+0j
        for t1 in range(Nt):
            for t2 in range(Nt):
                blockij = np.zeros((Nx,Nx))+0j
                blockji = np.zeros((Nx,Nx))+0j
                for n in range(-int(Nt/2),int(Nt/2)):
                    blockij += np.exp(1j*wn(n)*(t2-t1))*NN[n+int(Nt/2)]
                    blockji += np.exp(1j*wn(n)*(t1-t2))*NN[n+int(Nt/2)]
                for x1 in range(Nx):
                    for x2 in range(Nx):
                        answer[Nx*t1+x1][Nx*t2+x2]=blockji[x2][x1]*blockij[x1][x2]/Nt/Nt
        return answer

    def hessian(phi):
        answer = (1./deltaU+ 1.) * np.eye(V,k=0)+0j
        
        answer-= trMm2(phi,1) + trMm2(phi,-1)

        sgn,ldH = np.linalg.slogdet(  answer.real )
        
        return ldH
    
    def S(phi):
        sgn, ldMM = np.linalg.slogdet( MM(phi) )
        return (V*phi*phi/(2*deltaU) - ldMM).real

    def Seff(phi):
        if isinstance(phi,np.ndarray):
            phi = phi[0]

        act = S(1j*phi)
        ldH = hessian(1j*phi)

        seff = -(act + 0.5 * ldH)/V
        print( f"Seff({phi}) = {seff}" )
        return seff


    # use fsolve to find the root of Seff. phi_cr is a single number
    res = minimize_scalar( Seff )
    phi_cr = res.x

    # fsovle returns a np.core.multiarray.scalar which causes errors in isle params
    # hence convert to float

    # Return the (imaginary) shift for the configuration
    return float(phi_cr)

if __name__ == "__main__":
    print(f"Offset = {computeNLO()}")
