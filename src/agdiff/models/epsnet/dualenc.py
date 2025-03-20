import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm

from agdiff.utils.chem import BOND_TYPES
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, generate_symmetric_edge_noise, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform
import pdb
import torch.nn.functional as F
import time


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DualEncoderEpsNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)

        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder_global = SchNetEncoder(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.cutoff,
            smooth=config.smooth_conv,
        )

        self.encoder_local = GINEncoder(
            hidden_dim=config.hidden_dim,
            num_convs=config.num_convs_local,
        )

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )
            
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )

        '''
        Incorporate parameters together
        '''
        self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp])
        self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp])

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'

        if self.model_type == 'diffusion':
            # denoising diffusion
            ## betas
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            betas = torch.from_numpy(betas).float()
            self.betas = nn.Parameter(betas, requires_grad=False)
            ## variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            self.num_timesteps = self.betas.size(0)
        elif self.model_type == 'dsm':
            # denoising score matching
            sigmas = torch.tensor(
                np.exp(np.linspace(np.log(config.sigma_begin), np.log(config.sigma_end),
                                config.num_noise_level)), dtype=torch.float32)
            self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)
            self.num_timesteps = self.sigmas.size(0)  # betas.shape[0]


    def forward(self, atom_type, pos, bond_index, bond_type, batch, time_step, 
                edge_index=None, edge_type=None, edge_length=None, return_edges=False, 
                extend_order=True, extend_radius=True):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """

        N = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        local_edge_mask = is_local_edge(edge_type)  # (E, )

      
        if self.model_type == 'diffusion':
            # with the parameterization of NCSNv2
            # DDPM loss implicit handle the noise variance scale conditioning
            sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device)  # (E, 1)

        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )   # Embed edges
        # edge_attr += temb_edge

        # Global
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )

        ## Assemble pairwise features
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )    # (E_global, 2H)
        ## Invariant features of edges (radius graph, global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)    # (E_global, 1)
        
        # Encoding local
        edge_attr_local = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )   # Embed edges
        # edge_attr += temb_edge

        # Local
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        ## Assemble pairwise features
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )    # (E_local, 2H)
        ## Invariant features of edges (bond graph, local)
        if isinstance(sigma_edge, torch.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask]) # (E_local, 1)
        else:
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge) # (E_local, 1)

        if return_edges:
            return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask
        else:
            return edge_inv_global, edge_inv_local
    

    def get_loss(self, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True):
        if self.model_type == 'diffusion':
            return self.get_loss_diffusion(atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                anneal_power, return_unreduced_loss, return_unreduced_edge_loss, extend_order, extend_radius)
        


    def get_loss_diffusion(self, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True):
        N = atom_type.size(0)
        node2graph = batch

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2+1, ), device=pos.device)
        time_step = torch.cat(
            [time_step, self.num_timesteps-time_step-1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)  # (G, )
        # Perterb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()


        # Update invariant edge features, as shown in equation 5-7
        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
            atom_type = atom_type,
            pos = pos_perturbed,
            bond_index = bond_index,
            bond_type = bond_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius
        )   # (E_global, 1), (E_local, 1)

        edge2graph = node2graph.index_select(0, edge_index[0])
        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        d_perturbed = edge_length
        # Filtering for protein
        train_edge_mask = is_train_edge(edge_index)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()  # (E_global, 1), denoising direction

        global_mask = torch.logical_and(
                            torch.logical_or(d_perturbed <= self.config.cutoff, local_edge_mask.unsqueeze(-1)),
                            ~local_edge_mask.unsqueeze(-1)
                        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global)**2
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)

        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = (node_eq_local - target_pos_local)**2
        loss_local = 5 * torch.sum(loss_local, dim=-1, keepdim=True)

        # loss for atomic eps regression
        loss = loss_global + loss_local
        # loss_pos = scatter_add(loss_pos.squeeze(), node2graph)  # (G, 1)

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss


    def langevin_dynamics_sample(self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius=True, 
                                 n_steps=5000, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        if self.model_type == 'diffusion':
            return self.langevin_dynamics_sample_diffusion(atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius, 
                        n_steps, step_lr, clip, clip_local, clip_pos, min_sigma,global_start_sigma, w_global, w_reg, 
                        sampling_type=kwargs.get("sampling_type", 'ddpm_noisy'), eta=kwargs.get("eta", 1.))
        


    def langevin_dynamics_sample_diffusion(self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius=True, 
                                 n_steps=5000, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a
        
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
    
        self.eval()
        with torch.no_grad():

            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])
            pos = pos_init * sigmas[-1]

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius
                )   # (E_global, 1), (E_local, 1)

                # Local
                node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                # Global
                if sigmas[i] < global_start_sigma:
                    edge_inv_global = edge_inv_global * (1-local_edge_mask.view(-1, 1).float())
                    node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                # Sum
                eps_pos = node_eq_local + node_eq_global * w_global # + eps_pos_reg * w_reg
                noise = torch.randn_like(pos)  #  center_pos(torch.randn_like(pos), batch)
                step_size = step_lr * (sigmas[i] / 0.01) ** 2
                pos_next = pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size*2)
                pos = pos_next
                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
            
        return pos, pos_traj
    

def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index):
    
    return torch.ones(edge_index.size(1), device=edge_index.device).bool()
   

def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
