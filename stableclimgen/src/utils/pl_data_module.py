from lightning.pytorch import Trainer, LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler

class DataModule(LightningDataModule):
    def __init__(self, dataset_train=None, dataset_val=None, dataset_test=None, batch_size=16, num_workers=16, use_costum_ddp_sampler=False):
        super().__init__()

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.batch_size=batch_size
        self.num_workers= num_workers
        self.use_costum_ddp_sampler = use_costum_ddp_sampler
    

    def train_dataloader(self):
        
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_train, shuffle=False)
        else:
            sampler = None
        dataloader = DataLoader(self.dataset_train, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers) 

        return dataloader
    
    def val_dataloader(self):
        
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_val, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_val, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers) 

        return dataloader

    def test_dataloader(self):

        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_test, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_test, sampler=sampler, batch_size=self.batch_size,
                                num_workers=self.num_workers)

        return dataloader