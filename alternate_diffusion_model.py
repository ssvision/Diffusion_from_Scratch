import torch

class DeepDenoisingProbModel():

    def __init__(self, num_steps, beta_min, beta_max, device='cpu'):

        ''' num_steps = total no. of steps in the diffusion process
            beta_min & beta_max = beta values for noise scheduling
        '''

        self.num_steps = num_steps

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta = torch.linspace(beta_min, beta_max, num_steps).to(device)

        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1-self.alpha_bar)


    def fwd_process(self, original_img, timestep, noise=None):


        ''' Takes in an image and adds noise to it
            image and noise are of BxCxHxW
            timestep is a 1-D tensor of size B = batch size
        '''
        original_img_shape = original_img.shape
        batch_size = original_img_shape[0]

        alpha_bar_sqrt = self.alpha_bar_sqrt[timestep].reshape(batch_size)
        one_minus_alpha_bar_sqrt = self.one_minus_alpha_bar_sqrt[timestep].reshape(batch_size)

        alpha_bar_sqrt = alpha_bar_sqrt.reshape(-1,1,1,1)
        one_minus_alpha_bar_sqrt = one_minus_alpha_bar_sqrt.reshape(-1,1,1,1)

        '''The above reshaping operations convert alpha_bar_sqrt and one_minus_alpha_bar_sqrt to
           a 4-D tensor of shape (B,1,1,1) where the only element is the scalar form of alpha_bar_sqrt and one_minus_alpha_bar_sqrt
        '''

        eps = torch.randn_like(original_img) if noise is None else noise
        ''' creates B samples of random CxHxW images drawn from a normal distribution'''

        noisy_img = alpha_bar_sqrt*original_img + one_minus_alpha_bar_sqrt*eps
        ''' implements the forward update rule '''

        return noisy_img


    def sample_previous_step(self, X_t, noise_pred, timestep):

        ''' Returns image of previous timestep acc to algorithm
            based on the mean and variance & reparameterization trick
        '''

        variance = ((1-self.alpha_bar[timestep-1])/(1-self.alpha_bar[timestep])) * self.beta[timestep]
        std_dev = variance ** 0.5
        ''' variance given by the equation '''

        mean = (1/self.alpha_bar_sqrt[timestep]) * (X_t - (((self.beta[timestep])/(self.one_minus_alpha_bar_sqrt[timestep]))*noise_pred))

        if timestep == 0:
            noise = 0
        else:
            _eps = torch.rand_like(X_t)
            noise = std_dev * _eps

        X_t_minus_one = mean + noise

        return X_t_minus_one


if __name__ == '__main__':

    # experimenting here rough work ignore finally

    model = DeepDenoisingProbModel(10, 0.1, 1)

    og_img = torch.randn(32,3,28,28)
    timestep = torch.full((10,),3)

    fwd_eqn = model.fwd_process(og_img, timestep)
