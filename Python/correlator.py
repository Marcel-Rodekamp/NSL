import numpy as np 
import h5py as h5
import itertools
import lsqfit 
import gvar as gv
import yaml 
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)


@dataclass
class CorrelatorData:
    Nt: int
    beta: float
    U: float
    mu: float
    Nx: int
    Nconf: int

    # diagonalization routine
    # this is a function that recieves C = np.array (Nbst,Nt,Nx,Nx) and returns the diagonalized 
    # coorellator C_diag = np.array (Nbst,Nt,Nx)
    diagonalize: callable

    Nbst: int = 100
    delta: float = None

    # bootstrap sample IDs (Nbst, Nconf)
    # We can generate bst samples of an observable using:
    # data.C_sp[data.k_bst].mean(axis=1) 
    # which has the shape (Nbst,*data.shape[1:])
    k_bst: np.ndarray = None
    
    # S, per configuration (Nconf, )
    actVals: np.ndarray = None

    # weights for the reweighting
    weights: np.ndarray = None
    weights_bst: np.ndarray = None
    weights_est: np.ndarray = None
    weights_err: np.ndarray = None
    
    # p p^+, per configuration (Nconf,Nt,Nx)
    C_sp: np.ndarray = None
    C_sp_bst: np.ndarray = None
    C_sp_est: np.ndarray = None
    C_sp_err: np.ndarray = None
    C_sp_cov: np.ndarray = None

    # h h^+, per configuration (Nconf,Nt,Nx)
    C_sh: np.ndarray = None
    C_sh_bst: np.ndarray = None
    C_sh_est: np.ndarray = None
    C_sh_err: np.ndarray = None
    C_sh_cov: np.ndarray = None

    # p p^+ symmetriezed, per configuration (Nconf,Nt,Nx)
    Cs_sp_bst: np.ndarray = None
    Cs_sp_est: np.ndarray = None
    Cs_sp_err: np.ndarray = None
    Cs_sp_cov: np.ndarray = None

    # h h^+, per configuration (Nconf,Nt,Nx)
    Cs_sh_bst: np.ndarray = None
    Cs_sh_est: np.ndarray = None
    Cs_sh_err: np.ndarray = None
    Cs_sh_cov: np.ndarray = None

    def serialize(self, h5f):
        logger.info("Serializing CorrelatorData...")

        def write(grp, key, val):
            if isinstance(val,dict):
                # if the value is a dictionary we recursively call write on each key-value pair
                for subkey,subval in val.items():
                    write(grp, subkey, subval)
            elif isinstance(val, nonlinear_fit):
                #gv.dumps pickles the lsqfit.nonlinear_fit object and converts it to a byte string
                # casting it to np.void allows the string to be stored into the h5 file.
                grp.create_dataset(key, data=np.void(gv.dumps(val)))
            elif isinstance(val, Path):
                # if the value is a Path we store the string representation
                grp.create_dataset(key, data=str(val))
            else:
                # otherwise we store numbers and numpy arrays which have a native representation
                # in HDF5
                grp.create_dataset(key, data=val)

        # loop over the different entries in the dataclass
        for name, obj in self.__dict__.items():
            write(h5f.create_group(name),name,obj)

    def deserialize(self, h5f):
        logger.info(f"Reading Cached Correlator Data...")

        def read(grp,key,name):
            if key == "":
                currentH5Key = f'/{name}'
            else:
                currentH5Key = f'/{name}/{key}'

            if isinstance(grp[currentH5Key],h5.Group):
                # if the entry is a group we recursively call read on each key-value pair
                for subkey in grp[currentH5Key].keys():
                    read(grp, f'{key}/{subkey}', name)
            elif isinstance(grp[currentH5Key],h5.Dataset):
                # if the entry is a dataset we read the data and store it in the dataclass

                # The lsqfit fir results are stored in name = fit_results and ending key = res
                if name == "fit_results" and key[-3:] == "res":
                    # we use gv.loads to unpickle the object lsqfit.nonlinear_fit
                    self.__dict__[name][key[1:]] = gv.loads(grp[currentH5Key][()])
                elif type(self.__dict__[name]) is dict:
                    # if we are reading a dictionary
                    self.__dict__[name][key[1:]] = grp[currentH5Key][()]
                else:
                    # otherwise we can just read the numbers, numpy arrays
                    self.__dict__[name] = grp[currentH5Key][()]
        
        # loop over the different entries in the dataclass
        for name in self.__dict__.keys():
            if name not in h5f.keys():
                logger.info( f"\t Didn't find dataset {name}" )
                continue
            read(h5f, "", name)

    def reweighting_weights(self):
        logger.info("Measuring reweighting weights...")

        self.weights = np.exp(-1j*self.actVals.imag) # shape = (Nconf,)
        self.weights_bst = self.weights[self.k_bst].mean(axis=1) # shape = (Nbst)
        self.weights_est = self.weights_bst.mean(axis=0) # shape = (,)
        self.weights_err = self.weights_bst.std (axis=0) # shape = (,)

    def measure_statistical_power(self):
        r"""
            We calculate the statistical power 
            
            |Σ| = < exp(-i Im S) > |_{Ncut}

            For various cuts of the number of configurations Nconf.
            Ncut = 0.1*Nconf, 0.2*Nconf, ..., Nconf

            The relevant value is at Ncut = Nconf. The other values are used for the convergence plot in simulation statistics.
        """
        logger.info("Measuring statistical power...")
        ImS_bst = self.actVals.imag[self.k_bst]

        self.NconfCuts = np.arange(int(0.1*self.Nconf),self.Nconf, int(0.1*self.Nconf) )

        self.statistical_power_bst = np.zeros( (self.Nbst, len(self.NconfCuts)) )
        for i,N in enumerate(self.NconfCuts):
            self.statistical_power_bst[:,i] = np.abs(np.exp(-1j*ImS_bst[:,:N]).mean(axis=1))

        self.statistical_power_est = self.statistical_power_bst.mean(axis=0)
        self.statistical_power_err = self.statistical_power_bst.std(axis=0)
    
    def measure_correlator(self):
        def measure( C ):
            r"""
                We measure the raw correlator by:
                0. Hermitizing C(t)_{x,y} = (C(t)_{y,x}^* + C(t)_{x,y})/2
                1. Bootstrapping
                2. Reweighting with exp(-i Im S)
                3. Diagonalizing the correlator (Givens-Rotation, Jacobi method)
                4. Symmetrizing (Cs(t) = 0.5*(C(t) + C(Nt-t)))
                5. Covariance mean, standard deviation & Covariance for C and Cs over bootstrap
            """

            # The correlators are hermitian, we make sure that is true per configuration
            logger.info("\t 0. Hermitizing...")
            C += C.transpose(0,1,3,2).conj()
            C *= 0.5

            # 1. bootstrap the data
            logger.info("\t 1. Bootstraping...")
            # Note, this may be reduced to a loop over all bootstrap samples separately to reduce the memory footprint
            # for nbst in range(self.Nbst)
            C_bst = C[self.k_bst].mean(1) # shape = (Nbst,Nt,Nx,Nx)

            # 2. Reweight with exp(-i Im S)
            logger.info("\t 2. Reweighting...")
            C = C * self.weights[:,None,None,None] / self.weights_est # shape = (Nconf,Nt,Nx,Nx)
            C_bst = C_bst * self.weights_bst[:,None,None,None] / self.weights_est  # shape = (Nbst,Nt,Nx,Nx)
            
            # symmetrize
            logger.info("\t 4. Symmetrizing with time-reverse...")
            Cs_bst = 0.5*(C_bst[:,1:,:,:] + C_bst[:,1:,:,:][:,::-1,:,:]) # shape = (Nbst,Nt-1,Nx

            # diagonalize the correlator and compute the bootstrap mean and standard deviation
            # C_bst.shape = (Nbst,Nt,Nx)
            logger.info("\t 3. Diagonalizing...")
            C_bst = self.diagonalize( C_bst )
            Cs_bst = self.diagonalize( Cs_bst )

            # end for nbst in range(self.Nbst)

            # finally average and estimate uncertainty/covariance
            logger.info("\t 5. Bootstrap Average, Deviation, Covariance...")
            C_est = C_bst.mean(axis=0) # shape = (Nt,Nx)
            C_err = C_bst.std(axis=0)  # shape = (Nt,Nx)
            Cs_est = Cs_bst.mean(axis=0) # shape = (Nt,Nx)
            Cs_err = Cs_bst.std(axis=0)  # shape = (Nt,Nx)

            # estimate covariance for the fitting
            Cs_cov = np.zeros( (self.Nt-1,self.Nt-1,self.Nx), dtype=complex )
            C_cov = np.zeros( (self.Nt,self.Nt,self.Nx), dtype=complex )
            for k in range(self.Nx):
                Cs_cov[:,:,k] = np.cov(Cs_bst[:,:,k], rowvar=False)
                C_cov[:,:,k]  = np.cov(C_bst[:,:,k], rowvar=False)

            return C.real, C_bst.real, C_est.real, C_err, C_cov.real, \
                           Cs_bst.real, Cs_est.real, Cs_err, Cs_cov.real
        # measure

        logger.info("Measuring C_sp...")
        self.C_sp, self.C_sp_bst, self.C_sp_est, self.C_sp_err, self.C_sp_cov, \
            self.Cs_sp_bst, self.Cs_sp_est, self.Cs_sp_err, self.Cs_sp_cov = measure(self.C_sp)

        logger.info("Measuring C_sh...")
        self.C_sh, self.C_sh_bst, self.C_sh_est, self.C_sh_err, self.C_sh_cov, \
            self.Cs_sh_bst, self.Cs_sh_est, self.Cs_sh_err, self.Cs_sh_cov = measure(self.C_sh)

    def __post_init__(self):
        if self.k_bst is None:
            self.k_bst = np.random.randint(0,self.Nconf, size=(self.Nbst, self.Nconf))
        else:
            Nbst_,Nconf_ = self.k_bst.shape
            if Nbst_ != self.Nbst or Nconf != self.Nconf:
                raise RuntimeError( "Passed k_bst does not align with Nbst: {Nbst_} (k_bst) != {self.Nbst} (given) or Nconf: {Nconf_} (given) != {self.Nconf} " )
        
        self.delta = self.beta/self.Nt
    
        self.reweighting_weights()

        self.measure_statistical_power()

        self.measure_correlator()


def diagonalize_givens_rotation( C, Nt, Nx, Nbst, amplitudes ):
    r"""
        This diagonalization is based on the jacobi method
        where the largest off diagonal element is rotated into the diagonal iteratively
    """ 
    # based on the algorithm implementation in 
    # https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Lectures/lectures2015.pdf

    def diagonalizeBlock(block):
        blockSize = block.stop-block.start

        # find the row/column index sets that reference the upper triangular matrix starting at the k=1 off diagonal
        r,c = np.triu_indices(blockSize,k=1)
        
        # create a copy of the data within the block
        C_bst = C_block[:,:,block,block]

        # set the precision of the algorithm
        precision = 1e-18

        # this defines a ending condition for the algorithm
        # in case convergence can not be found
        maxIter = blockSize**3

        def findMaxOffDiag(nbst,tau):        
            r"""
                This function finds the maximal off diagonal element in the upper triangular matrix.
                The correlator is hermitian, thus we only need to look at the upper triangular matrix.
            """
            # Identify the largest element in the upper triangular matrix
            idx = np.abs(C_bst[nbst,tau,r,c]).argmax()
            
            # return the result
            return np.abs(C_bst[nbst,tau,r[idx],c[idx]]),  r[idx], c[idx]

        def rotate(nbst,tau, k, l):
            r"""
                Given the index (k,l) of the maximal element in the upper triangular matrix
                We can rotate it into the diagonal. Maintaining the structure of the original matrix.

                For details on the roatation consider lecture notes
                https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Lectures/lectures2015.pdf

            """

            # determine rotation
            c = 1 # cos(Theta)
            s = 0 # sin(Theta)

            if C_bst[nbst,tau,k,l] != 0:
                T = (C_bst[nbst,tau,l,l]-C_bst[nbst,tau,k,k])/(2*C_bst[nbst,tau,k,l]);

                if T > 0:
                    t = 1./(T + np.sqrt(1+T**2))
                else:
                    t =-1./(-T + np.sqrt(1+T**2))

                c = 1./np.sqrt(1+t**2)
                s = c*t
            else:
                # Do no rotation and set the largest elements to 0
                C_bst[nbst,tau,k,l], C_bst[nbst,tau,l,k] = 0,0

                return

            # actually rotate
            C_kk = C_bst[nbst,tau,k,k]
            C_ll = C_bst[nbst,tau,l,l]

            C_bst[nbst,tau,k,k] = c**2 * C_kk - 2*c*s*C_bst[nbst,tau,k,l] + s**2*C_ll
            C_bst[nbst,tau,l,l] = s**2 * C_kk + 2*c*s*C_bst[nbst,tau,k,l] + c**2*C_ll
            C_bst[nbst,tau,k,l], C_bst[nbst,tau,l,k] = 0,0
        
            for i in range(blockSize):
                if i != k and i != l:
                    C_ik = C_bst[nbst,tau,i,k];
                    C_il = C_bst[nbst,tau,i,l];
                    C_bst[nbst,tau,i,k] = c*C_ik - s*C_il;
                    C_bst[nbst,tau,k,i] = C_bst[nbst,tau,i,k];
                    C_bst[nbst,tau,i,l] = c*C_il + s*C_ik;
                    C_bst[nbst,tau,l,i] = C_bst[nbst,tau,i,l];

        def jacobi_method( nbst, tau ):
            r"""
                This function implements the jacobi method to diagonalize the correlator
                It iteratively finds the largest off diagonal element and rotates it into the diagonal.
            """
            # find largest off diagonal element 
            maxOffDiag, maxOffDiag_r,maxOffDiag_c = findMaxOffDiag(nbst,tau)

            iter_ = 0
            while( maxOffDiag > precision and iter_ < maxIter ):
                # find largest off diagonal element
                maxOffDiag, maxOffDiag_r,maxOffDiag_c = findMaxOffDiag(nbst,tau)
                
                # rotate that element into the diagonal
                rotate(nbst,tau, maxOffDiag_r, maxOffDiag_c)
                
                # prepare next iteration
                iter_+=1
            # end while

            if iter_ >= maxIter:
                logger.info(f"Jacobi method did not converge for block {block} after {maxIter} iterations.")

        # loop over different bootstrap and time slices
        for nbst in range(Nbst):
            for tau in range(Nt):
                jacobi_method(nbst,tau)

        # we can now just extract the diagonal elements and return
        return  np.diagonal(C_bst,axis1=2,axis2=3) # shape = (Nbst,Nt,Nx)
    # end diagonalizeBlock

    # for the symmetrized correlator we have one less time slice. Hence we need to
    # extract the size of the time axis here. The incoming correlators are assumed
    # to be of shape = (Nbst,Nt,Nx,Nx)
    Nt = C.shape[1]
    
    # prepare memory to store the diagonalized correlators
    C_diag_bst = np.zeros( (Nbst,Nt,Nx), dtype=complex )

    # We start by block diagonalizing the correlator using the symmetry of the system
    C_block = amplitudes[None,None,:,:] @ C @ amplitudes.T.conj()[None,None,:,:] # shape = (Nbst,Nt,Nx,Nx)

    # to a good approximation we could simply use the diagonal elements of the block
    # diagonal version. 
    #return np.diagonal(C_block, axis1=2, axis2=3)

    # We now discard noise at the off block diagonals and go though each block individually
    # For each block we diagonalize it and store the diagonal elements (eigenvalues) in C_diag_bst
    # corresponding to the current available block.
    for block in [slice_pp,slice_mp,slice_pm,slice_mm]:
        C_diag_bst[:,:,block]  = diagonalizeBlock( block )

    # return the diagonalized correlator
    return C_diag_bst
# end diagonalize_givens_rotation

def diagonalize_unitary( C, amplitudes ):
    r"""
    """ 
    # We start by block diagonalizing the correlator using the symmetry of the system
    C_block = amplitudes[None,None,:,:] @ C @ amplitudes.T.conj()[None,None,:,:] # shape = (Nbst,Nt,Nx,Nx)

    # Assuming the block diagonal is actually diagonal:
    return np.diagonal(C_block, axis1=2, axis2=3)

# end diagonalize_givens_rotation
