import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm
import math

def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)



def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    ts = torch.linspace(0, timesteps, timesteps)
    s_limit = 6 # define a suitable limit for saturation
    sigmoid_input = -s_limit + (2 * ts / timesteps) * s_limit
    sigmoid = torch.sigmoid(sigmoid_input)
    
    return beta_start + sigmoid * (beta_end - beta_start)


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        # define alphas (alpha_t)
        # TODO
        self.alphas = 1. - self.betas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # TODO
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # sqrt(1 - alpha_bar_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # TODO
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # beta_t / sqrt(1 - alpha_bar_t) coefficient for the noise
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        )

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y=None, cfg_scale=0.0):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)

        # Predict noise (CFG logic)
        if cfg_scale > 0 and y is not None:
            noise_cond = model(x, t, class_label=y)
            null_token = model.num_classes
            y_uncond = torch.full_like(y, null_token)
            noise_uncond = model(x, t, class_label=y_uncond)
            predicted_noise = (1 + cfg_scale) * noise_cond - cfg_scale * noise_uncond
        
        # Conditional sampling without CFG
        elif y is not None:
            predicted_noise = model(x, t, class_label=y)
        
        # Unconditional sampling 
        else:
            predicted_noise = model(x, t, class_label=None)
        
        # Clamp x0 for stability
        # leave x0 alone in equation 4
        x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        x0_pred = x0_pred.clamp(-1, 1)
        
        # Recompute noise from clamped x0
        # Clamp x0_pred to keep it in valid data range [-1,1].
        # BUT clamping breaks consistency between x0_pred and the noise ε used in DDPM Eq.8.
        # Since the reverse process uses ε, we must recompute ε from the clamped x0_pred.
        # Otherwise: x0_pred and ε mismatch → incorrect model_mean → unstable sampling (white/black blowouts).

        # leave noise alone in equation 4
        predicted_noise = (x - sqrt_alphas_cumprod_t * x0_pred) / sqrt_one_minus_alphas_cumprod_t
        
        # PDF Equation 8: x_{t-1} = 1/√(α_t) · (x_t - β_t/√(1-ᾱ_t) · ε) + √(β̃_t)·z
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        # First compute the posterior mean μ_t (Eq.8). This is the most likely x_{t-1}.
        # If t > 0, sample from the Gaussian by adding noise scaled by posterior variance.
        # If t == 0, return μ_t directly (final clean image, no noise added).

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, y=None, cfg_scale=0.0, return_all_timesteps=False):
        # Exercise 2.2b: Full sampling loop
        # Exercise 2.4: Added y and cfg_scale args
        
        device = next(model.parameters()).device
        
        # Start from pure noise x_T
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        # If conditional and y is not provided, we can't do CFG properly unless we want random classes
        # For this exercise, we assume y is passed if cfg_scale > 0.
        intermediates = [img.cpu()] if return_all_timesteps else []
 
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i, y=y, cfg_scale=cfg_scale)

            if return_all_timesteps:
                intermediates.append(img.cpu())

        if return_all_timesteps:
            return img, intermediates

        # TODO (2.2): Return the generated images
        return img

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; 
        # if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        # Apply equation 4: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise 
        return (
            extract(self.sqrt_alphas_cumprod, t, x_zero.shape) * x_zero +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape) * noise
        )

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1",classes=None):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        # Get noisy image x_t
        x_noisy = self.q_sample(x_zero=x_zero, t=t, noise=noise)

        # Predict the noise
        predicted_noise = denoise_model(x_noisy, t, class_label=classes)

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
