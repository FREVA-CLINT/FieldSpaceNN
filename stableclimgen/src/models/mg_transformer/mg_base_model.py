import torch
import torch.nn as nn

from ...modules.grids.grid_layer import GridLayer

class MG_base_model(nn.Module):
    def __init__(self,
                 mgrids
                 ) -> None:
        
                
        super().__init__()

        # Create grid layers for each unique global level
        zooms = []
        self.grid_layers = nn.ModuleDict()
        for zoom, mgrid in enumerate(mgrids):
            self.grid_layers[str(int(zoom))] = GridLayer(zoom, mgrid['adjc'], mgrid['adjc_mask'], mgrid['coords'], coord_system='polar')
            zooms.append(zoom)

        self.register_buffer('zooms', torch.tensor(zooms), persistent=False)
        self.zoom_max = int(self.zooms[-1])

        self.grid_layer_max = self.grid_layers[str(int(self.zooms[-1]))]