import torch
import torch.nn.functional as F
import torch.nn as nn

from ...modules.transformer.transformer_modules import MultiHeadAttentionBlock



class MultiGridAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in: int,
                 model_dim_out: int,
                 n_grids: list,
                 att_dim=None,
                 n_head_channels:int=16
                ) -> None: 
      
        super().__init__()
        
        grid_embedding = torch.randn(n_grids, att_dim*2)
        self.grid_embedding = nn.Parameter(grid_embedding, requires_grad=True)

        self.layer_norm = nn.LayerNorm(att_dim, elementwise_affine=True)

        n_heads = att_dim//n_head_channels if att_dim > n_head_channels else 1

        self.MHA = MultiHeadAttentionBlock(
            att_dim, att_dim, 1, input_dim=att_dim, qkv_proj=False, v_proj=False
            )   
       


    def forward(self, x_grids:list, mask_grids: list):
        if mask_grids[0] is not None:
            mask_grids = torch.stack(mask_grids, dim=-1)
        else:
            mask_grids = None
        
        x_grids = torch.stack(x_grids, dim=-2)

        shift, scale = self.grid_embedding.unsqueeze(dim=1).chunk(2, dim=-1)
        x_att = self.layer_norm(x_grids.unsqueeze(dim=-1) * (scale + 1) + shift) 

        b,n,nv,ng,nc,ngc = x_att.shape

        q = x_att[:,:,:,0]
        k = x_att

        q = q.view(b*n*nv,1,-1)
        k = k.view(b*n *nv,ng, -1)
        x_grids = x_grids.view(b*n*nv,ng, -1)
        
        if mask_grids is not None:
            mask_grids = mask_grids.view(b*n*nv,ng)

        x = self.MHA(q=q, k=k, v=x_grids, mask=mask_grids)

        x = x.view(b,n,nv,-1)

        return x
