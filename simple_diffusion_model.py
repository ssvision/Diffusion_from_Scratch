import torch
import numpy as np



class DDPM():

    def __init__(self, n_steps=1000, beta_min=0.0001, beta_max=0.02, device='cpu'):

        self.n_steps = n_steps

        self.beta_min = beta_min
        self.beta_max = beta_max

        self.device = device    

        # Create the noise scheduler
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alphas = 1 - self.betas

        self.alpha_bars = torch.empty_like(self.alphas)

        product = 1

        for i, alpha in enumerate(self.alphas):

            product = product * alpha
            self.alpha_bars[i] = product

    @staticmethod
    def sqrt(x):
        if isinstance(x, torch.Tensor):
            return torch.sqrt(x)
        
        return x**0.5
    
        
    def forward_process(self, x0, t, noise=None):

        alpha_t = self.alpha_bars[t].reshape(-1,1,1,1)
        eps = torch.randn_like(x0) if noise is None else noise
        
        # x_t = sqrt(alpha_bar) * x_0  + sqrt(1 - alpha_bar) * eps
        res = self.sqrt(alpha_t) * x0 + self.sqrt(1 - alpha_t) * eps  
        return res
        

    # Sampling from a DDPM model, check algorithm explanation on this
    def sample_backward_t(self, net, x_t, t):
        eps_t = net(x_t, torch.tensor([t] * x_t.shape[0], dtype=torch.long).to(x_t.device).unsqueeze(1))
        mu_t = (x_t - (1 - self.alphas[t]) / self.sqrt(1 - self.alpha_bars[t]) * eps_t) / self.sqrt(self.alphas[t])  # posterior mean
        if t == 0:
            noise_t = 0
        else:
            beta_t = self.betas[t]
            beta_tilde_t =  (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * beta_t  # posterior variance
            noise_t = self.sqrt(beta_tilde_t) * torch.randn_like(x_t)
        x_t_minus_1 = mu_t + noise_t  # x_{t-1} = N(x_t-1; mu(x_t, x_0), beta(_tilde)_t I)
        return x_t_minus_1
    

    def sample_backward(self, net, in_shape):
        x = torch.randn(in_shape).to(self.device)
        net = net.to(self.device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_t(net, x, t)  # x_{t-1} = sample_backward_t(x_t)
        return x  # x_0
    