import torch
import torch.nn as nn

from ...modules.grids.grid_layer import GridLayer, MultiRelativeCoordinateManager

class MGNO_base_model(nn.Module):
    def __init__(self,
                 mgrids,
                 rotate_coord_system=True,
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
        # Construct blocks based on configurations

        self.rcm = MultiRelativeCoordinateManager(
            self.grid_layers,
            rotate_coord_system=rotate_coord_system
            )