from abc import abstractmethod, ABC
from scipy.optimize import minimize, basinhopping, shgo
from typing import List
from typing import List, TypedDict
from math import isclose
import math
import torch
from tqdm import tqdm
pi = torch.pi

class NelderMeadLLOptimizer:

    def __init__(self, x0: List, θ0_bounds: List[tuple]):
        self._θ0_bounds = θ0_bounds
        self._x0 = x0

    def optimize(self, log_likelihood_func: callable):
        return minimize(fun=log_likelihood_func, x0=self._x0,
                        bounds=self._θ0_bounds, method='Nelder-Mead').x
        
class DefaultSHGOOptimizer:

    def __init__(self, θ0_bounds: List[tuple]):
        self._θ0_bounds = θ0_bounds

    def optimize(self, log_likelihood_func: callable):
        result = shgo(log_likelihood_func, bounds=self._θ0_bounds)
        return result.x

class Cumulants(TypedDict):
    κ1 = None
    κ2 = None
    κ4 = None

class MertonProcessParameters(TypedDict):
    mu = None
    sigma = None
    lam = None
    m = None
    v = None

merton_process_ll_optimizer = NelderMeadLLOptimizer(
    x0 = [0.004, 0.007, 0.02, 1.004, 0.004],
    θ0_bounds=[(0.0001, 0.1), (0.001, 0.1), (0.00001, 0.9),
               (0.0001, 2), (0.0001, 2)])

merton_process_ll_optimizer_1 = DefaultSHGOOptimizer(
    θ0_bounds=[(-0.1, 0.1), (-10, 1), (-10, 1),
               (-0.1, 0.1), (-10, 0)])

class MertonModel:
    
    def __init__(self, asset_price_dataset, recovery_method, device='cpu', separate_num=None):
        asset_price_dataset = torch.where(asset_price_dataset <= 0, torch.tensor(-20.0), torch.log(asset_price_dataset))
        self.device = device
        self._x = asset_price_dataset[1:].to(self.device)
        self._size = len(self._x)
        self._t = torch.arange(1, len(asset_price_dataset), device=self.device)
        self._x_pre = asset_price_dataset[:-1].to(self.device)
        self._ll_optimizer = merton_process_ll_optimizer_1
        self._recovery_method = recovery_method
        self._parameters = self._create_empty_param_instance()
        self._separate_num = separate_num
        if self._separate_num != None:
            self._batch_size = math.ceil(self._size/self._separate_num)
        self._fit()
        
        
    def _kappa1(self, x_pre, theta, t):
        r, sigma, lam, m, v = theta
        merton_μ = r - 0.5*(sigma**2) - (lam*(torch.exp(m+0.5*(v**2))-1))
        return merton_μ + (lam * m) # + x_pre

    def _kappa2(self, x_pre, theta, t):
        r, sigma, lam, m, v = theta
        return (sigma ** 2) + (lam * ((m ** 2) + (v ** 2))) #* t

    def _kappa4(self, x_pre, theta, t):
        r, sigma, lam, m, v = theta
        return lam * ((m ** 4) + (6 * (m ** 2) * (v ** 2)) + (3 * (v ** 4))) # * t
    
    def characteristic_function(self, x_pre, omega, t, theta : MertonProcessParameters):
        r, sigma, lam, m, v = theta
        ret = torch.exp(
                1j * omega * x_pre
                + 1j * omega * (r - 0.5 * sigma**2 - (lam * (torch.exp(m + 0.5 * v**2) - 1)))
                - 0.5 * (sigma * omega)**2
                + (lam * (torch.exp(1j * omega * m - 0.5 * (omega * v)**2) - 1))
            )
        return ret
    def _sample_jumps(self, n_sample_paths):
        return torch.randn(n_sample_paths, device=self.device) * self._parameters['v'] + self._parameters['m']
    
    def _pdf(self, x_pre, x, t, theta, cumulants):
        return self._recovery_method.recover(x, t, cumulants, theta, (lambda omega, theta, t: self.characteristic_function(x_pre.unsqueeze(1).repeat(1, omega.shape[1]), omega, t.unsqueeze(1).repeat(1, omega.shape[1]), theta)))
    
    def _all_pdf(self, theta):
        if self._separate_num == None:
            return self._pdf(self._x_pre, self._x, self._t, 
                                theta,
                                cumulants={'κ1': self._kappa1(self._x_pre, theta, self._t),
                                'κ2': self._kappa2(self._x_pre, theta, self._t),
                                'κ4': self._kappa4(self._x_pre, theta, self._t)
                                }
                            )
        bs = self._batch_size
        ms = self._size
        pdf_list = [self._pdf(self._x_pre[bs*i:min(bs*(i+1), ms)], self._x[bs*i:min(bs*(i+1), ms)], self._t[bs*i:min(bs*(i+1), ms)], 
                                theta,
                                cumulants={'κ1': self._kappa1(self._x_pre[bs*i:min(bs*(i+1), ms)], theta, self._t[bs*i:min(bs*(i+1), ms)]),
                                'κ2': self._kappa2(self._x_pre[bs*i:min(bs*(i+1), ms)], theta, self._t[bs*i:min(bs*(i+1), ms)]),
                                'κ4': self._kappa4(self._x_pre[bs*i:min(bs*(i+1), ms)], theta, self._t[bs*i:min(bs*(i+1), ms)])
                                }
                            ) for i in range(self._separate_num)]
        return torch.concatenate(pdf_list)
    
    def _negative_loglikelihood(self, theta: tuple):
        _theta = torch.tensor(theta, device=self.device)
        _theta = torch.tensor([_theta[0], torch.exp(_theta[1]), torch.exp(_theta[2]), _theta[3], torch.exp(_theta[4])])
        pdf = self._all_pdf(_theta)        
        ret = -torch.sum(torch.log(pdf)/self._size).cpu()
        return ret 
    
    def _create_empty_param_instance(self):
        return MertonProcessParameters()
    
    def _tuple_to_param_order(self, theta):
        _theta = torch.tensor(theta, device=self.device)
        self._parameters['mu'], self._parameters['sigma'], self._parameters[
            'lam'], self._parameters['m'], self._parameters['v'] = torch.tensor([_theta[0], torch.exp(_theta[1]), torch.exp(_theta[2]), _theta[3], torch.exp(_theta[4])], device=self.device)
    
    def _fit(self):
        theta = self._ll_optimizer.optimize(self._negative_loglikelihood)
        self._tuple_to_param_order(theta)
        
    def sample_path(self, x_0, t, n_sample_paths=10):
        x_0 = torch.tensor([torch.log(torch.tensor(x_0, device=self.device))] * n_sample_paths)
        X = torch.zeros((n_sample_paths, t), device=self.device)
        X[:,0] = x_0
        for i in tqdm(range(1, t)):
            dt = torch.tensor(1.0, device=self.device)
            dW = torch.randn(n_sample_paths, device=self.device) * torch.sqrt(dt)
            dJ = torch.poisson(self._parameters['lam'] * dt.repeat(n_sample_paths))
            jumps = self._sample_jumps(n_sample_paths)
            X[:, i] = X[:, i-1] + (self._parameters['mu'] - 0.5 * self._parameters['sigma']**2 - self._parameters['lam'] * (torch.exp(self._parameters['m'] + 0.5 * self._parameters['v']**2) - 1 )) * dt + self._parameters['sigma'] * dW + dJ * jumps
        return torch.exp(X)

class COSMethodBasedDensityRecovery:
    def __init__(self, N_freq, device='cpu'):
        self._k = torch.arange(N_freq, device=device)
        self._N_freq = N_freq
        self.device = device
        
    def _get_integration_range(self, x, cumulants: Cumulants):
        intg_range = torch.full_like(x, 8.0, device=self.device)
        mask = (cumulants['κ2']  >= 0) & (cumulants['κ4']  >= 0)
        sqrt_val = torch.sqrt(cumulants['κ2'] [mask] + torch.sqrt(cumulants['κ4'] [mask]))
        sqrt_val = sqrt_val.to(intg_range.dtype)
        intg_range[mask] = 8 * sqrt_val
        a = x + cumulants['κ1'] - intg_range
        b = x + cumulants['κ1'] + intg_range
        return b, a

    def recover(self, x, t, cumulants: Cumulants, theta, phi_omega: callable):
        b, a = self._get_integration_range(x, cumulants)
        d = b - a
        u = torch.outer(1/d, (self._k*pi))
        f_k = 2.0 * (phi_omega(u, theta, t) * torch.exp(-1j * a.unsqueeze(1).repeat(1, u.shape[1]) * u)).real
        f_k[:,0] = f_k[:,0] * 0.5 
        ret = torch.abs(torch.sum(f_k * torch.cos( torch.outer((x-a)/d, self._k*pi) ), dim=1))/d
        return ret

        
def CRPS(predict, y, kappa):
    predict = predict.detach()
    N = predict.shape[0]
    T = predict.shape[1]
    K = kappa.shape[0]
    predict, _ = torch.sort(predict, dim = 0)
    X = torch.zeros(T, K)
    for i in range(T):
        for k in range(K):
            X[i,k] = predict[int(kappa[k]*N), i]
    def Lambda(kappa, y, q):
        return 2 * (kappa - torch.where(y<q, torch.tensor(1.0), torch.tensor(0.0))) * (y - q)
    kappa = kappa.repeat(T,1)
    y = y.unsqueeze(1).repeat(1,K)
    ret = torch.mean(Lambda(kappa, y, X))
    return ret, X

if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd

    # Apple Inc. (AAPL) の株価データを取得
    ticker = "AMZN"
    data = yf.download(ticker, start="2022-07-01", end="2022-12-30")
    test_data = yf.download(ticker, start="2022-12-29", end="2023-03-01")
    test_data = torch.tensor(test_data['Open'].values).reshape(-1)
    x = torch.tensor(data['Open'].values).reshape(-1)
    print(x.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MertonModel(x, recovery_method=COSMethodBasedDensityRecovery(N_freq=500,device=device), device=device)
    N = 100
    predict = model.sample_path(x_0=x[x.shape[0]-1], t=test_data.shape[0], n_sample_paths=N).cpu()

    import matplotlib.pyplot as plt
    M = 0
    plt.plot(torch.arange(-x.shape[0]+ 1, 1), x, label=f'Observed Path', color="#1E5EFF")
    for i in range(M):
        plt.plot(torch.arange(predict.shape[1]), predict[i,:], label=f'Path {i+1}')

    plt.plot(torch.arange(test_data.shape[0]), test_data, color="#1E5EFF") #label=f'Valid Path'

    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')  
    plt.title('Sample Paths from Merton Model')


    kappa = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    test_data = test_data[1:]
    predict = predict[:, 1:]
    score,  quantile_x = CRPS(predict, test_data, kappa)
    plt.plot(torch.arange(1, predict.shape[1]+1), quantile_x[:,1], label=f'Path {0.2} quantile', color='#FF0000')
    plt.plot(torch.arange(1, predict.shape[1]+1), quantile_x[:,4], label=f'Median Path', color='#FF0000')
    plt.plot(torch.arange(1, predict.shape[1]+1), quantile_x[:,7], label=f'Path {0.8} quantile', color='#FF0000')
    plt.fill_between(torch.arange(1, predict.shape[1]+1), quantile_x[:,1], quantile_x[:,7], color="#FF4848", alpha=0.2)
    plt.legend()
    plt.show()

    """
    def _gaussian_cf(ω, θ, t):
        mu, σ = θ
        return torch.exp(1j*mu*ω-0.5*(σ ** 2)*(ω ** 2))


    def _κ1(θ, t):
        mu, _ = θ
        return torch.full_like(t, mu)


    def _κ2(θ, t):
        _, σ = θ
        return torch.full_like(t, σ ** 2)


    def _κ4(θ, t): return torch.full_like(t, 0)


    def test_gaussian_density_recovery_cos_method():
        θ = (3.05, 1.2)
        b = 3.0
        x = torch.linspace(start=b - 1.5, end=b + 1.5, steps=int(1000))
        t = torch.zeros_like(x)
        pdf = COSMethodBasedDensityRecovery(N_freq=2000).recover(x, t, {'κ1': _κ1(θ, t),
                                'κ2': _κ2(θ, t),
                                'κ4': _κ4(θ, t)}, θ, _gaussian_cf)
        plt.clf()
        plt.plot(x, pdf, label='True Density')
        plt.show()

    test_gaussian_density_recovery_cos_method()

    """