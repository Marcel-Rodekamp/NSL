import numpy as np
import itertools
import lsqfit 
import gvar as gv
import yaml 
from dataclasses import dataclass,field
from tqdm import tqdm
from lsqfit import nonlinear_fit

from correlator import CorrelatorData

import logging
logger = logging.getLogger(__name__)



@dataclass
class Fitter:
    # Raw/bootstrap/symmetrized Correlator
    correlator: CorrelatorData

    NstatesMax: int = 2
    NfitExtends: int = 2
    
    # System parameters stored for convenience. 
    # These are extracted from the correlator object
    Nt: int = None 
    beta: float = None
    U: float = None
    mu: float = None 
    Nconf: int = None
    Nbst: int = None
    delta: float = None
    Nx: int = None

    # total charge
    charge_bst: np.ndarray = None
    charge_est: np.ndarray = None
    charge_err: np.ndarray = None

    # Autocorrelations
    # These need to be redone!
    #gamma_actVals_est: np.ndarray = None
    #gamma_actVals_err: np.ndarray = None
    #gamma_C_sp_est: np.ndarray = None
    #gamma_C_sp_err: np.ndarray = None

    # Statistical Power
    NconfCuts: np.ndarray = None
    statistical_power_bst: np.ndarray = None
    statistical_power_est: np.ndarray = None
    statistical_power_err: np.ndarray = None

    # Effecitve Mass
    meff_cosh_bst: np.ndarray = None
    meff_cosh_est: np.ndarray = None
    meff_cosh_err: np.ndarray = None

    meff_exp_bst: np.ndarray = None
    meff_exp_est: np.ndarray = None
    meff_exp_err: np.ndarray = None

    # Fit Results
    # keys are of the form: "{nStateFit}/{tstart}/{tend}"
    # value are of type: lsqfir.nonlinear_fit 
    fit_results: dict = field(default_factory=dict)
    # keys are of the form: "{nStateFit}/{tstart}/{tend}/{parameterLabel}"
    # parameterLabel = A0,E0,A1,E1,...,fcn,AIC
    # values are of type: np.ndarray( Nbst, Nx )
    # value|@fcn is of type: np.ndarray( Nbst, Nplot, Nx )
    fit_params_bst: dict = field(default_factory=dict)
    # value are of type: np.ndarrary( Nx )
    # value|@fcn is of type: np.ndarray( Nbst, Nplot, Nx )
    fit_params_est: dict = field(default_factory=dict)
    fit_params_err: dict = field(default_factory=dict)

    # stores the model average for each parameter on {'A0','E0',...} bootstrap sample
    # keys are of the form: {parameterLabel}
    # values are of type: np.ndarray( Nbst, Nx )
    modelAverage_bst: dict = field(default_factory=dict)
    # values are of type: np.ndarray( Nx )
    modelAverage_est: dict = field(default_factory=dict)
    modelAverage_err: dict = field(default_factory=dict)

    ignoreCache: bool = False
    Nplot = 100

    def analysisCache(self):
        return f"data/AnalysisCache/Nt{self.Nt:g}_beta{self.beta:g}_U{self.U:g}_mu{self.mu:g}.h5"

    def serialize(self):
        logger.info("Serializing Fitter...")

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
        with h5.File(self.analysisCache(),'w') as h5f:
            for name, obj in self.__dict__.items():
                if name == "correlator":
                    obj.serialize(h5f)
                else:
                    write(h5f.create_group(name),name,obj)

    def deserialize(self):
        logger.info(f"Reading Cached Analysis ({self.analysisCache()})...")

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
        with h5.File(self.analysisCache(),'r') as h5f:

            for name in self.__dict__.keys():
                if name not in h5f.keys():
                    logger.info( f"\t Didn't find dataset {name}" )
                    continue
                elif name == "correlator":
                    continue
                else:
                    read(h5f, "", name)

            self.correlator = CorrelatorData(
                Nt = self.Nt,
                beta = self.beta,
                U = self.U,
                mu = self.mu,
                Nx = self.Nx,
                Nconf = self.Nconf
            )

            self.correlator.deserialize()

    def measure_autocorrelation(self):
        def auto_corr(obs):
            shape = obs.shape[1:]

            # With the data we are generating autocorrelation is not an issue. If the zero crossing isn't found 
            # go back and increase this up to self.Nconf//2
            W = self.Nconf//2
            
            gamma_bst = np.zeros( (self.Nbst, W, *shape) , dtype=obs.dtype )

            obs_bst = obs[self.k_bst] # shape = (Nbst,Nconf,*obs.shape[1:])
            obs_est = obs_bst.mean(axis=1) # shape = (Nbst,*obs.shape[1:])

            for tau in tqdm(range(W)):
                gamma_bst[:,tau,...] = np.mean( 
                    (obs_bst - obs_est[:,None,...]) * ( np.roll(obs_bst,tau,axis=1) - obs_est[:,None,...]).conj(), 
                    axis = 1
                ).real

            gamma_bst /= gamma_bst[:,0,...][:,None,...]

            gamma_est = gamma_bst.mean(axis=0)
            gamma_err = gamma_bst.std(axis=0)

            zero_crossing = np.argmax(gamma_est<=0,axis=0)

            W = np.max(zero_crossing)

            return gamma_est[:W+1], gamma_err[:W+1]
         
        logger.info("Measuring autocorrelation of S...")
        self.gamma_actVals_est,self.gamma_actVals_err = auto_corr(self.actVals.real)
        logger.info("Measuring autocorrelation of C_sp...")
        # We have data that is almost uncorrelated, thus we save some time by just providing the first few configurations
        self.gamma_C_sp_est,self.gamma_C_sp_err = auto_corr(self.correlator.C_sp.mean(axis=(1,2,3)))
        self.gamma_C_sp_est = self.gamma_C_sp_est.real
        # ToDo: Measure autocorrelation of all observables and store the max autocorrelation 
        #       separately

    def measure_statistical_power(self):
        r"""
            We calculate the statistical power 
            
            |Σ| = < exp(-i Im S) > |_{Ncut}

            For various cuts of the number of configurations Nconf.
            Ncut = 0.1*Nconf, 0.2*Nconf, ..., Nconf

            The relevant value is at Ncut = Nconf. The other values are used for the convergence plot in simulation statistics.
        """
        logger.info("Measuring statistical power...")
        ImS_bst = self.correlator.actVals.imag[self.k_bst]

        self.NconfCuts = np.arange(int(0.1*self.Nconf),self.Nconf, int(0.1*self.Nconf) )

        self.statistical_power_bst = np.zeros( (self.Nbst, len(self.NconfCuts)) )
        for i,N in enumerate(self.NconfCuts):
            self.statistical_power_bst[:,i] = np.abs(np.exp(-1j*ImS_bst[:,:N]).mean(axis=1))

        self.statistical_power_est = self.statistical_power_bst.mean(axis=0)
        self.statistical_power_err = self.statistical_power_bst.std(axis=0)

    def measure_charge(self):
        logger.info("Measuring charge...")
        self.charge_bst = np.zeros( (self.Nbst,) )

        if self.mu != 0:
            # The correlators are already reweights
            # q_x = C_sp_xx(tau=0) - C_sh_xx(tau=0)
            self.charge_bst = np.sum(self.correlator.C_sp_bst[ :, 0, : ] - self.correlator.C_sh_bst[ :, 0, : ], axis=1) # shape = (Nbst,)

        self.charge_est = np.mean(self.charge_bst,axis=0) 
        self.charge_err = np.std (self.charge_bst,axis=0) 

    def measure_meff(self):
        logger.info("Measuring meff...")

        # compute the effective mass for each bootstrap sample
        self.meff_exp_bst = (
            np.log(np.roll(self.correlator.C_sp_bst[:,:,:],-1,axis=1)) - np.log(np.roll(self.correlator.C_sp_bst[:,:,:],1,axis=1))
        ) / (2*self.delta) # shape = (Nbst,Nt,Nx)

        self.meff_exp_est = self.meff_exp_bst.mean(axis=0)
        self.meff_exp_err = self.meff_exp_bst.std(axis=0)


        # prepare memory for the bootstrap samples
        # we discarded the tau=0 time slice for the symmetrization
        # as the effective mass required information from tau+1, tau 
        # we thus discard the last time slice.
        self.meff_cosh_bst = np.zeros( (self.Nbst, self.Nt-3, self.Nx) )

        # The abscissa for the cosh model is shifted by -1 as it discards the tau=0 time slice
        # due to a missing tau=Nt time slice.
        # We thus start at tau = 1 and end at tau = Nt-1 
        # Then the indexing requires a shift by -1 for the output effective mass 
        # (but not for the correlators as they already discarded that point)
        tau = np.arange(1,self.Nt-2) 

        frac = (self.correlator.Cs_sp_bst[:,tau+1,:] + self.correlator.Cs_sp_bst[:,tau-1,:]) / (2*self.correlator.Cs_sp_bst[:,tau,:])

        # Due to finite statistics and a degenerate eigenspace of the correlation matrix in the symmetrized form
        # this faction can become smaller then 1 yielding to a not defined arc cosh. 
        # in such a case we will find the closest (in tau) points that are larger then 1 and
        # interpolate between them.
        #for nbst in range(self.Nbst):
        #    for tau in range(0,self.Nt-3):
        #        for k in range(self.Nx):

        #            # If the fraction value drops below one
        #            if frac[nbst,tau,k] < 1:
        #                
        #                # find the closest points that are larger then 1
        #                # we know that tau-1 is > 1 as we just accepted/adjusted it
        #                # we just need to find one to the right.
        #                for n in range(tau+1,self.Nt-3):
        #                    if frac[nbst,n,k] >= 1:
        #                        frac[nbst,tau,k] = (frac[nbst,tau-1,k] + frac[nbst,n,k])/2

        self.meff_cosh_bst = np.arccosh( 
            frac
        ) / self.delta # shape = (Nbst,Nt,Nx)

        self.meff_cosh_est = np.nanmean(self.meff_cosh_bst,axis=0)
        self.meff_cosh_err = np.nanstd( self.meff_cosh_bst,axis=0) 

    def fitExtends(self):
        minExtend  = 2 # => 4 points
        maxExtend  = self.Nt//2 - 1 # => Nt-2 points

        # evenly space the fit interval sizes
        stepExtend = (maxExtend-minExtend)//self.NfitExtends 

        return np.arange(minExtend,maxExtend,stepExtend)

    def coshModel(self,tau,p,numStates):
        r"""
            We symmetries the correlators as    
                C(tau) = \sum_n A_n exp( - * E_n * tau )
            we obtain
                1/2 ( C(tau) + C(-tau) ) = \sum_n A_n cosh( E_n * (tau - beta/2) )
        """
        tau = tau-(self.Nt/2)

        # This is the zeros term which does not have a ΔE_n (single State)
        C = p["A0"] * np.cosh( p["E0"] * tau )

        # The following terms have ΔE_n (multi State)
        
        # We store the result of the telescopic sum in ΔE_n = E_n - E_{n-1} to obtain 
        # E_{n-1} = E_0 + sum_{m=1}^{n-1} ΔE_m
        E = p["E0"]

        for n in range(1,numStates):
            C += p[f"A{n}"] * np.cosh( 
                (p[f"ΔE{n}"]+E) * tau 
            )
            
            # next term in the telescopic sum
            E+= p[f"ΔE{n}"]

        return C

    def expModel(self,tau,p,numStates,sign):
        r"""

        """
        # This is the zeros term which does not have a ΔE_n (single State)
        C = p["A0"] * np.exp( -sign*p[f"E{0}"]*tau )

        # The following terms have ΔE_n (multi State)
        
        # We store the result of the telescopic sum in ΔE_n = E_n - E_{n-1} to obtain 
        # E_{n-1} = E_0 + sum_{m=1}^{n-1} ΔE_m
        E = p["E0"]

        for n in range(1,numStates):
            C += p[f"A{n}"] * np.exp( 
                -sign*(p[f"ΔE{n}"]+E)*tau
            )
            
            # next term in the telescopic sum
            E+= p[f"ΔE{n}"]

        return C

    def fit_energy(self):
        logger.info("Fitting energy...")
        def AIC(fit):
            r"""
                Compute the Akaike information criterion for a fit result
                based on the chi^2 obtained from lsqfit.
                The form can be found in 
                    https://arxiv.org/abs/2305.19417
                    https://arxiv.org/abs/2208.14983
                    https://arxiv.org/abs/2008.01069
                equation 3 in the first:
                    AIC^{perf} = -2ln L^* + 2k - 2d_K
                Here we compare
                    1. -2*ln(L^*) = chi^2 
                    2. k = number of parameters
                    3. d_K = number of points

                @1. we use the augmented chi^2 from lsqfit including the priors
                @2. we have a prior for each parameter thus each counts twice
            """
            chip = np.sum([
                (fit.prior[key].mean - fit.p[key].mean)**2 / fit.prior[key].sdev**2 for key in fit.prior.keys()
            ])
            L = fit.chi2 - chip
            k = len(fit.p) 
            d_K = len(fit.x)

            return L + 2*k - 2*d_K

        def fitWindow(extend, k):
            r"""

            """
            # determine the signal to noise ratio to impose a cutoff 
            StN = np.abs(self.correlator.C_sp_est[:,k]) / self.correlator.C_sp_err[:,k]

            # check if the signal becomes to bad 
            cut_StN = np.argwhere( StN < 1 )

            # If there is no Signal-to-Noise problem we fit the cosh model
            useCoshModel = len(cut_StN) == 0

            # we find the longest slope
            tmax = np.argmax(self.correlator.C_sp_est[:,k])
            tmin = np.argmin(self.correlator.C_sp_est[:,k])

            # determine if we are backwards or forwards propagarting
            # The assigned sign matches the form of the expModel: exp( - sign * E * tau )
            if tmax < tmin: # Interval Left: forward state -> exp( - |E| * tau )
                sign = +1
            else:           # Interval Right: backward state  -> exp( + |E| * tau )
                sign = -1

            if useCoshModel:
                # the fit window is twice extend centered around beta/2 <=> Nt/2
                # The sign is not used for the model but is stored for representation 
                # with the non-interacting energy.
                return self.Nt//2 - extend , self.Nt//2 + extend, sign, useCoshModel
            else: 
                # For the exponential model we have to find the interval on the longest 
                # slope

                # find the midpoint of that interval
                midpoint = (tmax+tmin)//2

                # determine preliminary start and end point
                ts = midpoint - extend
                te = midpoint + extend

                # If we have no signal to noise problem, we can just use the fit window 
                # as intended. If we are within 15%Nt of the turning point (tmin) we truncate
                # This is never the case as per construction of the condition on when to use coshModel
                # However, if we want to change the condition on when to use coshModel, this might be
                # helpful, thus we leave it in.
                if len(cut_StN) == 0:
                    cut = int(0.15 * self.Nt)
                    if sign > 0:
                        # Interval on the left
                        te = te if te < tmin-cut else tmin-cut
                    else:
                        # Interval on the right
                        ts = ts if ts > tmin+cut else tmin+cut
                # If we have a signal to noise problem we truncate the fit window before 
                # its first occurence
                else: 
                    if sign > 0:
                        # Interval on the left
                        te = te if te < cut_StN[0,0] else cut_StN[0,0]-1
                        ts = ts if ts > 0 else 0 
                    else:
                        # Interval on the right
                        ts = ts if ts > cut_StN[-1,0] else cut_StN[-1,0]+1
                        te = te if te < self.Nt-1 else self.Nt-1
                # Now we can return the fit window and the sign of the energy
                return ts, te, sign, useCoshModel

        def getPrior(useCoshModel, tstart, tend, numStates, k):
            r"""

            """
            prior = gv.BufferDict()
            
            scale = 10

            Q_min = 0.05

            if numStates == 1:
                if useCoshModel:
                    meff_est = self.meff_cosh_est[ts:te, k].mean(axis=0) * self.delta  
                    meff_err = scale * self.meff_cosh_err[ts:te, k].mean(axis=0) * self.delta  
                else:
                    meff_est = self.meff_exp_est[ts:te, k].mean(axis=0) * self.delta
                    meff_err = scale * self.meff_exp_err[ts:te, k].mean(axis=0) * self.delta

                if np.isnan(meff_est) or np.isnan(meff_err):
                    meff_est, meff_err = 1,1 
                    
                prior["log(E0)"] = gv.log(gv.gvar( np.abs(meff_est), meff_err ))
                prior["A0"] = gv.gvar( 1,1 )
            else:
                # We assume that lower state fits have been done. Thus we can get the priors
                # from their results

                # This fills the arrays
                # self.modelAverage_bst[f"E{i}"]
                # self.modelAverage_bst[f"A{i}"]
                # and respective _est, _err
                # which computed on the {N-1}-state fit servers as priors here
                self.modelAverage(numStates=numStates-1)

                # We start with the prior for E0
                prior["log(E0)"] = gv.log(gv.gvar(
                    self.modelAverage_est["E0"][k],
                    scale * self.modelAverage_err["E0"][k]
                ))

                # For every other state we fit the respective energy difference
                # ΔE_i = E_{i} - E_{i-1}
                for i in range(1,numStates-1):
                    prior[f"log(ΔE{i})"] = gv.log(gv.gvar(
                        self.modelAverage_bst[f"E{i}"][:,k] - self.modelAverage_bst[f"E{i-1}"][:,k],
                        scale * np.std(
                            self.modelAverage_bst[f"E{i}"][:,k] - self.modelAverage_bst[f"E{i-1}"][:,k],
                        )
                    ))

                # Next, the last state where E_{N-1} is not known we assume the ΔE to be of the order 
                # of the E_{N-2} with a large (100%) uncertainty
                prior[f"log(ΔE{numStates-1})"] = gv.log(gv.gvar(
                    self.modelAverage_est[f"E{numStates-2}"][k],
                    self.modelAverage_est[f"E{numStates-2}"][k]
                ))

                # Now repeat a similar procedure for the scales A0
                for i in range(0,numStates-1):
                    prior[f"A{i}"] = gv.gvar(
                        self.modelAverage_est[f"A{i}"][k],
                        scale * self.modelAverage_err[f"A{i}"][k]
                    )

                # Again the last one, for which no information is available, we assume it to be of the 
                # same order then the previous one with a large (100%) uncertainty
                prior[f"A{numStates-1}"] = gv.gvar(
                    self.modelAverage_est[f"A{numStates-2}"][k],
                    self.modelAverage_est[f"A{numStates-2}"][k]
                )
            # else (numStates != 1)

            return prior

        def makeDict(numStates,tstart,tend,k):
            key = f'{numStates}/{tstart}/{tend}/{k}'
            if f'{key}/res' not in self.fit_results.keys():
                # lsqfit result for central value fit
                self.fit_results[f'{key}/res'] = None
                # bool to determine which model was chosen
                self.fit_results[f'{key}/useCoshModel'] = None
                # sign of the energy
                self.fit_results[f'{key}/sign'] = None
                
                for i in range(numStates):
                    # fit parameters for bootstrap samples
                    self.fit_params_bst[f'{key}/E{i}'] = np.zeros( (self.Nbst,) )
                    self.fit_params_bst[f'{key}/A{i}'] = np.zeros( (self.Nbst,) )
                    self.fit_params_bst[f'{key}/fcn'] = np.zeros( (self.Nbst, self.Nplot) )
                    self.fit_params_bst[f'{key}/AIC'] = np.zeros( (self.Nbst, ) )

                    # fit parameters for central values
                    self.fit_params_est[f'{key}/E{i}'] = 0
                    self.fit_params_est[f'{key}/A{i}'] = 0
                    self.fit_params_est[f'{key}/fcn'] = np.zeros( (self.Nplot,) )
                    self.fit_params_est[f'{key}/AIC'] = 0

                    # fit parameter errors from bootstrap samples
                    self.fit_params_err[f'{key}/E{i}'] = 0
                    self.fit_params_err[f'{key}/A{i}'] = 0
                    self.fit_params_err[f'{key}/fcn'] = np.zeros( (self.Nplot,) )
                # end for i 

                # if created we still need to fit this dictionary
                return True

            # otherwise we already fitted this dictionary and we can continue
            return False
                
        def fit(useCoshModel, tstart, tend, sign, numStates, k, nbst = None):
            r"""
                Perform a fit to the correlator data using the model and prior provided.
                Compute the Akaike information criterion for the fit.
            """

            # organize the data and fit model
            C_est, C_cov, model = None,None,None
            if useCoshModel:
                # Cosh Model uses the symmetrized correlator
                if nbst is None: # Central Value fits
                    C_est = self.correlator.Cs_sp_est[tstart:tend,k]
                else: # bootstrap fits
                    C_est = self.correlator.Cs_sp_bst[nbst,tstart:tend,k]

                # prepare covariance
                C_cov = self.correlator.Cs_sp_cov[tstart:tend,tstart:tend,k]
                
                # prepare model
                model = lambda tau, p: self.coshModel( tau, p, numStates)

                # prepare abscissa
                # we symmetrized the correlator, which removes the tau = 0 time slice
                # The data is still zero based stored thus 
                #   Cs_sp_est[0] = (Cs_sp_est[1] + Cs_sp_est[(Nt-1)-1])/2
                # We can compensate for this by shifting the abscissa (tau) by +1 such that
                #   tau[0] = 1         <=>   Cs_sp_est(tau=1)      <=>   Cs_sp_est[0]
                #   tau[1] = 2         <=>   Cs_sp_est(tau=2)      <=>   Cs_sp_est[1]
                #   ...                          ...                        ...
                #   tau[Nt-2] = Nt-1   <=>   Cs_sp_est(tau=Nt-1)   <=>   Cs_sp_est[(Nt-1)-1]
                # Which matches the total length of len(C_sp_est[:,k]) = Nt-1
                # We do not have to change the symmetrizing shift as we are yet symmetryzing the
                # correlator with time extend beta = Nt * delta. Only our array doesn't store
                # all the points.
                tau = np.arange(tstart,tend) + 1
            else:
                # Exp Model uses the non-symmetrized correlator
                if nbst is None: # Central Value fits
                    C_est = self.correlator.C_sp_est[tstart:tend,k]
                else:
                    C_est = self.correlator.C_sp_bst[nbst,tstart:tend,k]

                # prepare covariance
                C_cov = self.correlator.C_sp_cov[tstart:tend,tstart:tend,k]

                # prepare model
                model = lambda tau, p: self.expModel( tau, p, numStates, sign)

                # prepare abscissa
                tau = np.arange(tstart,tend)

            # lsqfit expects a gvar object for the data
            corr_gvar = gv.gvar(C_est, C_cov)

            # prepare the prior
            prior = getPrior(useCoshModel, tstart, tend, numStates, k)

            # perform the fit to the central values
            fitResult = nonlinear_fit( 
                data  = (tau,corr_gvar), 
                fcn   = model, 
                prior = prior 
            )

            # compute the fit function over a dense abscissa
            fcn = model(np.linspace(tstart,tend,self.Nplot),fitResult.pmean)

            # Compute the Akaike information criterion 
            aic =  AIC(fitResult)

            return fitResult, aic, fcn

        # loop over the different irreps
        pbar = tqdm( total = self.Nx * self.NstatesMax * len(self.fitExtends()) )
        for numStates in range(1,self.NstatesMax+1):
            for extend in self.fitExtends(): 
                for k in range(self.Nx):
                    # determine start and end time slice 
                    ts,te,sign,useCoshModel = fitWindow(extend,k)

                    # ensure that we have enough points to fit
                    if (te-ts) < 2*numStates + 2: 
                        pbar.update(1)
                        continue 

                    # prepare dictionary if not yet fitted
                    if not makeDict(numStates,ts,te,k): continue
        
                    #############################################################################
                    ## Central Value Fit
                    #############################################################################
                    fitRes,aic,fcn = fit(useCoshModel, ts, te, sign, numStates, k, nbst = None)
                    if useCoshModel:
                        pbar.write(f"{k=}, (tstart={ts*self.delta-self.beta/2}, tend={te*self.delta-self.beta/2}), sign={sign}, useCoshModel={useCoshModel}, Nstates={numStates}:")
                    else:
                        pbar.write(f"{k=}, (tstart={ts*self.delta}, tend={te*self.delta}), sign={sign}, useCoshModel={useCoshModel}, Nstates={numStates}:")
                    pbar.write(f"{fitRes}")

                    # store the fit result
                    self.fit_results[ f'{numStates}/{ts}/{te}/{k}/useCoshModel' ] = useCoshModel
                    self.fit_results[ f'{numStates}/{ts}/{te}/{k}/res' ] = fitRes
                    self.fit_results[ f'{numStates}/{ts}/{te}/{k}/sign' ] = sign

                    E = fitRes.p[f"E{0}"].mean
                    self.fit_params_est[ f'{numStates}/{ts}/{te}/{k}/E{0}' ] = E
                    self.fit_params_est[ f'{numStates}/{ts}/{te}/{k}/A{0}' ] = fitRes.p[f"A{0}"].mean
                    for i in range(1,numStates):
                        E += fitRes.p[f"ΔE{i}"].mean
                        self.fit_params_est[ f'{numStates}/{ts}/{te}/{k}/E{i}' ] = E
                        self.fit_params_est[ f'{numStates}/{ts}/{te}/{k}/A{i}' ] = fitRes.p[f"A{i}"].mean
                    self.fit_params_est[ f'{numStates}/{ts}/{te}/{k}/fcn' ][:] = fcn
                    self.fit_params_est[ f'{numStates}/{ts}/{te}/{k}/AIC' ] = aic
                    
                    #############################################################################
                    ## Bootstrap Fit
                    #############################################################################
                    for nbst in range(self.Nbst):
                        fitRes,aic,fcn = fit(useCoshModel, ts, te, sign, numStates, k, nbst = nbst)

                        # store the fit result
                        E = fitRes.p[f"E{0}"].mean
                        self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/E{0}' ][nbst] = E
                        self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/A{0}' ][nbst] = fitRes.p[f"A{0}"].mean
                        for i in range(1,numStates):
                            E += fitRes.p[f"ΔE{i}"].mean
                            self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/E{i}' ][nbst] = E
                            self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/A{i}' ][nbst] = fitRes.p[f"A{i}"].mean
                        self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/fcn' ][nbst,:] = fcn
                        self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/AIC' ][nbst] = aic
                    # for nbst

                    # Compute the bootstrap error
                    for i in range(numStates):
                        self.fit_params_err[ f'{numStates}/{ts}/{te}/{k}/E{i}' ] = np.std(
                            self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/E{i}' ], axis=0
                        )
                        self.fit_params_err[ f'{numStates}/{ts}/{te}/{k}/A{i}' ] = np.std(
                            self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/A{i}' ], axis=0
                        )
                    self.fit_params_err[ f'{numStates}/{ts}/{te}/{k}/fcn' ][:] = np.std(
                        self.fit_params_bst[ f'{numStates}/{ts}/{te}/{k}/fcn' ], axis=0
                    )

                    pbar.update(1)

    def modelAverage(self, numStates = None):
        r"""
            We average the different fit results over the fit ranges and the fit models.
            As a weight we compute the normalized akaike probability as
            .. math::
                w_i = \frac{\exp(- AIC_i / 2)}{\sum_j \exp(- AIC_j/2)}
            where i,j are indexing the different fits.
        """
        if numStates is None:
            logger.info("Averaging over fits...")

        # Create dictionaries for the final averages
        self.modelAverage_bst = {}
        self.modelAverage_est = {}
        self.modelAverage_err = {}

        # Create a dictionary to store lists of the parameters and AICs 
        paramsCollection_bst = {}
        paramsCollection_est = {}
        
        # loop over all parameters A_i and E_i
        for i in range(self.NstatesMax):
            Ei_bst = []
            Ai_bst = []
            AICi_bst = []

            Ei_est = []
            Ai_est = []
            AICi_est = []

            fitID = 0
            for key in self.fit_results.keys():
                # retrieve fit specification from the key
                # key = {numstates}/{ts}/{te}/{k}/{*}
                numStates_,ts,te,k,element  = key.split("/")

                #logger.info(f"Model averaging: {numStates_} {ts} {te} {k} {element}: {numStates}")
                
                # transform from string to integers  
                numStates_,ts,te,k = int(numStates_),int(ts),int(te),int(k)

                # there are res,sign,... in the fit_results. To avoid double counting 
                # we skip all but one
                if element != 'res': continue

                # for a single N-State model average we skip all that aren't computed yet
                # this comes from the usage as priors for the next fit iff nState is set.
                if numStates is not None:
                    if numStates != numStates_: continue

                # for a N-State fit we can only estimate energy E_0,...,E_{N-1}
                if i >= numStates_: continue

                # store the data
                Ei_bst.extend( self.fit_params_bst[ f'{numStates_}/{ts}/{te}/{k}/E{i}' ] )
                Ai_bst.extend( self.fit_params_bst[ f'{numStates_}/{ts}/{te}/{k}/A{i}' ] )
                AICi_bst.extend( self.fit_params_bst[ f'{numStates_}/{ts}/{te}/{k}/AIC' ] )

                Ei_est.append( self.fit_params_est[ f'{numStates_}/{ts}/{te}/{k}/E{i}' ] )
                Ai_est.append( self.fit_params_est[ f'{numStates_}/{ts}/{te}/{k}/A{i}' ] )
                AICi_est.append( self.fit_params_est[ f'{numStates_}/{ts}/{te}/{k}/AIC' ] )

                # enhance the fitID for the next fit
                fitID+=1

            # number of fits equals the fitID modulo the number of correlators
            fitID = fitID // self.Nx
            
            # if now fit has been done for this (A)E_i we can just continue
            if numStates is not None and fitID == 0: continue

            # convert to numpy arrays
            Ei_bst = np.array(Ei_bst).reshape(fitID, self.Nx, self.Nbst).transpose(2,0,1)
            Ai_bst = np.array(Ai_bst).reshape(fitID, self.Nx, self.Nbst).transpose(2,0,1)
            AICi_bst = np.array(AICi_bst).reshape(fitID, self.Nx, self.Nbst).transpose(2,0,1)

            Ei_est = np.array(Ei_est).reshape(fitID, self.Nx)
            Ai_est = np.array(Ai_est).reshape(fitID, self.Nx)
            AICi_est = np.array(AICi_est).reshape(fitID, self.Nx)

            # model average
            self.modelAverage_bst[f"E{i}"] = np.average(Ei_bst, axis=1, weights=np.exp(-0.5*AICi_bst))
            self.modelAverage_bst[f"A{i}"] = np.average(Ai_bst, axis=1, weights=np.exp(-0.5*AICi_bst))
                                       
            self.modelAverage_est[f"E{i}"] = np.average(Ei_est, axis=0, weights=np.exp(-0.5*AICi_est))
            self.modelAverage_est[f"A{i}"] = np.average(Ai_est, axis=0, weights=np.exp(-0.5*AICi_est))

            self.modelAverage_err[f"E{i}"] = np.std( self.modelAverage_bst[f"E{i}"], axis=0)
            self.modelAverage_err[f"A{i}"] = np.std( self.modelAverage_bst[f"A{i}"], axis=0)

        self.modelAverage_bst[f"AIC"] = AICi_bst
        self.modelAverage_est[f"AIC"] = AICi_est


    def __post_init__(self):
        #isSerealized = self.read_data()

        # This is for convinience
        self.k_bst = self.correlator.k_bst
        self.Nt    = self.correlator.Nt
        self.beta  = self.correlator.beta
        self.U     = self.correlator.U
        self.mu    = self.correlator.mu
        self.Nconf = self.correlator.Nconf
        self.Nbst  = self.correlator.Nbst
        self.Nx    = self.correlator.Nx
        self.delta = self.correlator.delta
    
        # This needs to be updated... The gamma_ arrays are commented out
        self.measure_statistical_power()

        self.measure_charge()

        self.measure_meff()

        self.fit_energy()

        self.modelAverage()
            
        # Has to be redone with the correlator thing ...
        #self.serialize()
