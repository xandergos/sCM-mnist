import math
import os
import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from toy.toy_utils import calc_karras_sigmas
from unet import Unet
import matplotlib.pyplot as plt

@click.command()
@click.option("--device", default="cuda", help="Device to train on")
@click.option("--epochs", default=200, help="Number of epochs to train")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--lr", default=3e-5, help="Learning rate")
@click.option("--extra-plots", is_flag=True, help="Generate additional training plots", default=True)
def train(device, epochs, batch_size, lr, extra_plots):
    # Setup model and optimizer
    model = Unet(256, 1, 1, base_dim=64, dim_mults=[2, 4]).to(device)
    model_pretrained = None
    
    if os.path.exists('model.pt'):
        model.load_state_dict(torch.load('model.pt', weights_only=True), strict=False)
        
        # Reset logvar weights for CM
        model.logvar_linear = nn.Linear(model.logvar_linear.in_features, model.logvar_linear.out_features).to(device)
        
        model_pretrained = Unet(256, 1, 1, base_dim=64, dim_mults=[2, 4]).to(device)
        model_pretrained.load_state_dict(torch.load('model.pt', weights_only=True), strict=False)
        print("Loaded model from model.pt, performing consistency distillation.")
        consistency_training = False
    else:
        print("No model found, performing consistency training.")
        consistency_training = True
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.99))
    sigma_data = 0.5
    
    P_mean = -2
    P_std = 1.6
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (1.0))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
            
    step = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch, class_label in progress_bar:
            x0 = batch.to(device)
    
            # Will use this for jvp calcs
            def model_wrapper(scaled_x_t, t):
                pred, logvar = model(scaled_x_t, t.flatten(), return_logvar=True)
                # If you want the model to be conditioned on class label (or anything else), just add it as an additional argument:
                # You do not need to change anything else in the algorithm.
                # pred, logvar = model(scaled_x_t, t.flatten(), class_label, return_logvar=True)
                return pred, logvar
            
            # Sample noise from log-normal distribution
            sigma = torch.randn(x0.shape[0], device=x0.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()  # Sample from proposal distribution
            t = torch.arctan(sigma / sigma_data)  # Convert to t using arctan
            
            # Generate z and x_t
            z = torch.randn_like(x0) * sigma_data
            x_t = torch.cos(t) * x0 + torch.sin(t) * z
            
            if consistency_training:
                # Estimate dx_t/dt (For consistency TRAINING)
                dxt_dt = torch.cos(t) * z - torch.sin(t) * x0
            else:
                # For consistency DISTILLATION
                # (model_pretrained is assumed to output v-predictions)
                with torch.no_grad():
                    pretrain_pred = model_pretrained(x_t / sigma_data, t.flatten())
                    dxt_dt = sigma_data * pretrain_pred
            
            # Next we have to calculate g and F_theta. We can do this simultaneously with torch.func.jvp
            # This doesn't match the paper because I think the paper had a typo
            v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
            v_t = torch.cos(t) * torch.sin(t)
            F_theta, F_theta_grad, logvar = torch.func.jvp(
                model_wrapper, 
                (x_t / sigma_data, t),
                (v_x, v_t),
                has_aux=True
            )
            logvar = logvar.view(-1, 1, 1, 1)
            F_theta_grad = F_theta_grad.detach()
            F_theta_minus = F_theta.detach()
            
            # Warmup steps. 10000 was used in the paper. I'm using 1000 for MNIST since it's an easier dataset.
            r = min(1.0, step / 1000)
            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            # Note that F_theta_grad is already multiplied by sin(t) cos(t) from the tangents. Doing it early helps with stability.
            second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
            g = g + second_term
            
            # Tangent normalization
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
            g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  # Multiplying by sqrt(numel(g_norm) / numel(g)) ensures that the norm is invariant to the spatial dimensions.
            g = g / (g_norm + 0.1)  # 0.1 is the constant c, can be modified but 0.1 was used in the paper
            
            # Tangent clipping (Only use this OR normalization)
            # g = torch.clamp(g, min=-1, max=1)
            
            # Calculate loss with adaptive weighting
            # Paper uses weight = 1 / sigma, but I've found this to cause instability.
            weight = 1
            loss = (weight / torch.exp(logvar)) * torch.square(F_theta - F_theta_minus - g) + logvar
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            step += 1
            
        torch.save(model.state_dict(), f'model_consistency.pt')
        
        #
        #
        #
        #
        # Everything below is just for visualization and can be removed safely
        #
        #
        #
        #
        #
        
        if not extra_plots:
            continue
        
        z = torch.randn(16, 1, 28, 28, generator=torch.Generator().manual_seed(42)).to(device)
        t0 = 1.56454 * torch.ones(16, device=device)
        t1 = 1.1 * torch.ones(16, device=device)
        with torch.no_grad():
            pred_x0 = torch.clamp(-sigma_data * model(z, t0), min=-0.5, max=0.5)
            
            plt.figure(figsize=(12, 12))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(pred_x0[i, 0].cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            os.makedirs('outputs_consistency/samples_1step', exist_ok=True)
            plt.savefig(f'outputs_consistency/samples_1step/epoch_{epoch:04d}.png')
            plt.close()
            
            z = torch.randn(16, 1, 28, 28, generator=torch.Generator().manual_seed(43)).to(device)
            t1_exp = t1.view(-1, 1, 1, 1)
            x_t = torch.sin(t1_exp) * z * sigma_data + torch.cos(t1_exp) * pred_x0
            pred_x0 = torch.clamp(torch.cos(t1_exp) * x_t - torch.sin(t1_exp) * sigma_data * model(x_t / sigma_data, t1), min=-0.5, max=0.5)
        
            plt.figure(figsize=(12, 12))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(pred_x0[i, 0].cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            os.makedirs('outputs_consistency/samples_2step', exist_ok=True)
            plt.savefig(f'outputs_consistency/samples_2step/epoch_{epoch:04d}.png')
            plt.close()
        
        # Plot one noise sample with different timesteps
        # Sample one image from the dataset for visualization
        sample_img, _ = dataset[0]
        sample_img = sample_img.unsqueeze(0).to(device)
        
        z = torch.randn(1, 1, 28, 28, generator=torch.Generator().manual_seed(42)).to(device) * sigma_data
        z = z.repeat(16, 1, 1, 1)  # Repeat the same noise 16 times
        t = torch.linspace(0, 1.56454, 16, device=device).view(-1, 1, 1, 1)  # Linearly spaced timesteps
        x_t = torch.cos(t) * sample_img + torch.sin(t) * z
        with torch.no_grad():
            pred_x0 = torch.clamp(torch.cos(t) * x_t - torch.sin(t) * sigma_data * model(x_t / 0.5, t.flatten()), min=-0.5, max=0.5)
            plt.figure(figsize=(12, 12))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(pred_x0[i, 0].cpu().numpy(), cmap='gray')
                plt.title(f't={t[i].item():.2f}')
                plt.axis('off')
            plt.tight_layout()
            os.makedirs('outputs_consistency/timesteps', exist_ok=True)
            plt.savefig(f'outputs_consistency/timesteps/epoch_{epoch:04d}.png')
            plt.close()
            
        # Testing gradients numerically
        if model_pretrained is not None and epoch % 5 == 0:
            sampling_timesteps = torch.arctan(calc_karras_sigmas(sigma_min=0.002, sigma_max=80, steps=64, rho=7) / sigma_data).to(device)
            trajectory_x0 = []
            trajectory_xt = []
            trajectory_t = []
            trajectory_grad = []
            
            endpoints = torch.randn(2, 1, 28, 28).to(device) * sigma_data
            alphas = torch.linspace(0, 1, 64).view(-1, 1, 1, 1).to(device)
            x_t = (endpoints[0] * alphas + endpoints[1] * (1 - alphas)) / torch.sqrt(alphas**2 + (1 - alphas)**2)
            with torch.no_grad():
                trajectory_guess = -sigma_data * model_wrapper(x_t / sigma_data, sampling_timesteps[0].repeat(x_t.shape[0]))[0]
            
            for t in tqdm(sampling_timesteps):
                trajectory_t.append(t.cpu().numpy())
                trajectory_xt.append(x_t.detach().cpu().numpy())
                
                t_repeated = t.repeat(x_t.shape[0]).view(-1, 1, 1, 1)
            
                with torch.no_grad():
                    dxt_dt = sigma_data * model_pretrained(x_t / sigma_data, t_repeated.flatten())
                
                v_x = torch.cos(t_repeated) * torch.sin(t_repeated) * dxt_dt / sigma_data
                v_t = torch.cos(t_repeated) * torch.sin(t_repeated)
                
                with torch.no_grad():
                    # Calculate JVP using torch.func.jvp
                    F_theta, F_theta_grad, logvar = torch.func.jvp(
                        model_wrapper, 
                        (x_t / sigma_data, t_repeated),
                        (v_x, v_t),
                        has_aux=True
                    )
                    F_theta_grad = F_theta_grad.detach()
                    F_theta_minus = F_theta.detach()
                    
                # Calculate predicted x0
                pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * F_theta
                trajectory_x0.append(pred_x0.detach().cpu().numpy())
                
                r = 1
                # Calculate gradient g using JVP rearrangement
                g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
                # Note that F_theta_grad is already multiplied by sin(t) cos(t) from the tangents. Doing it early helps with stability.
                second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
                g = g + second_term
                trajectory_grad.append(torch.sqrt(torch.square(g).sum(dim=(1, 2, 3))).cpu().numpy())
                
                next_t = sampling_timesteps[sampling_timesteps < t]
                if len(next_t) == 0:
                    next_t = 0
                else:
                    next_t = next_t[0]
                
                # Calculate x0 prediction and update x
                x_t = torch.cos(t - next_t) * x_t - torch.sin(t - next_t) * dxt_dt
                x_t = x_t.detach()
                
            trajectory_x0 = np.array(trajectory_x0)
            trajectory_xt = np.array(trajectory_xt)
            trajectory_t = np.array(trajectory_t)
            trajectory_grad = np.array(trajectory_grad)
                
            diff = torch.abs(x_t - trajectory_guess).mean().item()
            print(f"Consistency error: {diff:.4f}")
            
            discrete_grad = np.zeros_like(trajectory_x0)
            discrete_grad[:-1] = -np.cos(trajectory_t[:-1])[:, None, None, None, None] * (trajectory_x0[1:] - trajectory_x0[:-1]) / (trajectory_t[1:] - trajectory_t[:-1])[:, None, None, None, None]
            discrete_grad = np.sqrt(np.sum(np.square(discrete_grad), axis=(2, 3, 4)))

            # Create figure with two subplots side by side
            plt.figure(figsize=(12, 4))

            # Get shared vmin/vmax for consistent scale
            vmin = min(trajectory_grad.min(), discrete_grad.min())
            vmax = max(trajectory_grad.max(), discrete_grad.max())

            # Plot trajectory gradient 
            plt.subplot(121)
            plt.imshow(trajectory_grad, aspect='auto', cmap='RdBu', 
                      extent=[0, trajectory_grad.shape[1], 0, math.pi/2],
                      vmin=vmin, vmax=vmax)
            plt.colorbar(label='Gradient magnitude')
            plt.title('Trajectory Gradient')
            plt.xlabel('Initial X')
            plt.ylabel('Time step')
            
            # Plot discrete gradient
            plt.subplot(122)
            plt.imshow(discrete_grad, aspect='auto', cmap='RdBu',
                      extent=[0, discrete_grad.shape[1], 0, math.pi/2],
                      vmin=vmin, vmax=vmax)
            plt.colorbar(label='Gradient magnitude')
            plt.title('Discrete Gradient')
            plt.xlabel('Initial X')
            plt.ylabel('Time step')

            plt.tight_layout()
            os.makedirs('outputs_consistency/gradients', exist_ok=True)
            plt.savefig(f'outputs_consistency/gradients/epoch_{epoch:04d}.png')
            plt.close()
        

if __name__ == '__main__':
    train()