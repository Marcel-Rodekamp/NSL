import PyNSL as NSL
import torch 
import yaml
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class ActionImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phi, S):
        ctx.save_for_backward(phi)
        ctx.S = S

        config = NSL.Configuration()
        if phi.dim() == 2:
            config["phi"] = phi
            return torch.tensor(S.eval(config))
        elif phi.dim() == 3:
            Nconf,_,_ = phi.shape
            act = torch.zeros((Nconf, ), dtype=torch.cdouble)
            for n in range(Nconf):
                config["phi"] = phi[n,:,:]
                act[n] = S.eval(config)
            return act
        else:    
            raise ValueError("phi must be a 2 or 3 dimensional tensor")


    @staticmethod
    def backward(ctx, grad_output):
        phi, = ctx.saved_tensors
        S = ctx.S

        grad_input = torch.zeros_like(phi)
        config = NSL.Configuration()


        if phi.dim() == 2:
            Nt,Nx = phi.shape
            config["phi"] = phi
            dS = S.grad(config)
            grad_input = grad_output * dS["phi"].conj()

        elif phi.dim() == 3:
            Nconf,Nt,Nx = phi.shape

            for n in range(Nconf):
                config["phi"] = phi[n,:,:]
                dS = S.grad(config)
                grad_input[n] = grad_output[n] * dS["phi"].conj()
        else:
            raise ValueError("phi must be a 2 or 3 dimensional tensor")

        return grad_input, None

class Action(torch.nn.Module):
    def __init__(self, params):
        super(Action, self).__init__()
        
        self.lattice = NSL.Lattice.Generic(params["parameter filename"])

        self.hga = NSL.Action.HubbardGaugeAction(params)
        self.hfa = NSL.Action.HubbardFermionAction(self.lattice,params)
        self.S = NSL.Action.HubbardAction_EXP_GEN(self.hga,self.hfa)

    def forward(self, phi):
        return ActionImpl.apply(phi,self.S)

class StatPower(torch.nn.Module):
    def __init__(self, params, Nbst = 100):
        super(StatPower, self).__init__()
        self.S = Action(params)

        # if we want to calculate the bootstrap error we need these stored during the forward pass
        self.last_actVals = None
        self.last_logDetJ = None
        self.last_orig_actVals = None

        self.Nbst = Nbst


    def impl_(self, actVals, logDetJ, orig_actVals):
        return torch.abs(
            torch.mean(
                torch.exp(
                      logDetJ 
                    - (actVals.real - orig_actVals.real) 
                    - 1j * actVals.imag
                ) # exp
            ) # mean
        ) # abs

    def bootstrap(self):
        with torch.no_grad():
            statPower_bst = torch.zeros( (self.Nbst,), dtype=self.last_actVals.dtype, device=self.last_actVals.device )

            for nbst in range(self.Nbst):
                idx = torch.randint( 0, self.last_actVals.shape[0], (self.last_actVals.shape[0],) )
                statPower_bst[nbst] = self.impl_(self.last_actVals[idx], self.last_logDetJ[idx], self.last_orig_actVals[idx])

            # the .real is virtual here
            return torch.mean(statPower_bst.real), torch.std(statPower_bst)


    def forward(self, phi, logDetJ, orig_actVals):

        self.last_actVals = self.S(phi)
        self.last_logDetJ = logDetJ
        self.last_orig_actVals = orig_actVals

        return torch.abs(1-self.impl_(self.last_actVals, self.last_logDetJ, self.last_orig_actVals))


    # end forward

class Shift(torch.nn.Module):
    def __init__(self,params):
        super(Shift,self).__init__()
        
        self.register_parameter("shift",
            torch.nn.Parameter(
                torch.zeros((params['Nt'], params['Nx']), dtype=torch.double)
            )
        )

        torch.nn.init.uniform_(self.shift,-0.1,0.1)

    def forward(self,phi):
        ldJ = None

        if phi.dim() == 2:
            ldJ = torch.zeros( (1,), device=phi.device, dtype=phi.dtype, requires_grad=True )
        elif phi.dim() == 3:
            ldJ = torch.zeros( (phi.shape[0],),  device=phi.device, dtype=phi.dtype, requires_grad=True )
        else:
            raise ValueError("phi must be a 2 or 3 dimensional tensor")
            
        return phi + 1j*self.shift, ldJ

if __name__ == "__main__":
    torch.set_printoptions(precision=8)

    filename = 'twoSite.yml'

    with open(filename, 'r') as f:
        yml = yaml.safe_load(f)

    params = {
        'name': str(yml['system']['name']),
        'beta': float(yml['system']['beta']),
        'Nt': int(yml['system']['Nt']),
        'Nx': int(yml['system']['nions']),
        'U': float(yml['system']['U']),
        'mu': float(yml['system']['mu']) if yml['system']['mu'] else 0.0,
        'offset': float(yml['system']['offset']),
        'save frequency': int(yml['HMC']['save frequency']),
        'Ntherm': int(yml['HMC']['Ntherm']),
        'Nconf': int(yml['HMC']['Nconf']),
        'trajectory length': float(yml['Leapfrog']['trajectory length']),
        'Nmd': int(yml['Leapfrog']['Nmd']),
        'h5file': str(yml['fileIO']['h5file']),
        'Nepoch': int(yml['Train']['Nepoch']),
        'Nbatch': int(yml['Train']['Nbatch']),
        'learning rate': float(yml['Train']['learning rate']),
        'device': 'cpu',
        'parameter filename': filename,
    }

    data_phi = torch.zeros((params['Nconf'], params['Nt'], params['Nx']), dtype=torch.cdouble)
    data_act = torch.zeros((params['Nconf'], ), dtype=torch.cdouble)

    with h5.File(params['h5file'], 'r') as h5f:
        for n in range(params['Nconf']):
            data_phi[n,:,:] = torch.tensor(
                h5f[f"{params['name']}/markovChain/{n}/phi"][()].reshape(
                    params['Nt'], params['Nx']
                )
            ).to(params['device'])
            data_act[n] = torch.tensor(
                h5f[f"{params['name']}/markovChain/{n}/actVal"][()]
            ).to(params['device'])

    # prepare the data
    Ntrain = int(params['Nconf'] * 0.8)
    Nvalid = params['Nconf'] - Ntrain

    train_phi = data_phi[:Ntrain,:,:]
    train_act = data_act[:Ntrain]

    valid_phi = data_phi[Ntrain:,:,:]
    valid_act = data_act[Ntrain:]

    # Prepare for training
    net = Shift(params)

    loss = StatPower(params)

    optimizer = torch.optim.Adam(net.parameters(), lr=params['learning rate'])

    with torch.no_grad():
        phi, ldJ = net(valid_phi)

        valid_loss = loss(phi, ldJ, valid_act)
        train_loss = valid_loss

    validFreq = 2
    train_loss_history_est = torch.zeros( (params['Nepoch'],), dtype=torch.double)
    valid_loss_history_est = torch.zeros( (params['Nepoch'],), dtype=torch.double)

    train_loss_history_err = torch.zeros( (params['Nepoch'],), dtype=torch.double)
    valid_loss_history_err = torch.zeros( (params['Nepoch'],), dtype=torch.double)

    # Train
    pbar = tqdm(range(params['Nepoch']))
    for epoch in range(params['Nepoch']):
        pbar.set_description(f"Epoch {epoch: <3} | Loss {train_loss.item(): <.4f} | Valid {valid_loss.item(): <.4f}")

        optimizer.zero_grad()
        phi, ldJ = net(train_phi)

        train_loss = loss(phi, ldJ, train_act)
        train_loss.backward()

        optimizer.step()

        train_loss_history_est[epoch],train_loss_history_err[epoch] = loss.bootstrap()

        if epoch % validFreq == 0:
            with torch.no_grad():
                phi, ldJ = net(valid_phi)
                valid_loss = loss(phi, ldJ, valid_act)

                valid_loss_history_est[epoch], valid_loss_history_err[epoch] = loss.bootstrap()

        pbar.update()

    plt.errorbar(
        np.arange(params['Nepoch']),
        train_loss_history_est.numpy(),
        yerr=train_loss_history_err.numpy(), 
        label="Train",
        capsize = 2
    )
    plt.errorbar(
        np.arange(0,params['Nepoch'],validFreq)+0.01, 
        valid_loss_history_est.numpy()[::validFreq],
        yerr=valid_loss_history_err.numpy()[::validFreq],
        label="Valid",
        capsize = 2
    )

    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel(r"Loss $ = \left\vert \left\langle \Sigma \right\rangle\right\vert$", fontsize = 16)
    plt.title(
        rf"Training History: $\lambda = {params['learning rate']}, \, N_\mathrm{{train}} = {Ntrain}, \, N_\mathrm{{valid}} = {Nvalid}$", fontsize=22
    )
    plt.legend()

    plt.show()


