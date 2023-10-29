import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data_utils import custom_collator


class Tester:
    def __init__(
        self,
        args,
        train_ds,
        test_ds,
        model,
        optimizer,
        device
    ):  
        self.args = args
        
        self.test_ds = test_ds
        self.test_batch_size = args.test_batch_size
        
        self.device = device
        self.model = model
        self.loss_func = nn.MSELoss()
        self.optimizer = optimizer
        
        self.loss = 0
    
    def test(self):
        test_loader = DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=custom_collator
        )
        
        with torch.no_grad():
            for step, batch in enumerate(test_loader, 1):
                batch = [b.to(self.device) for b in batch]
                image, _ = batch
                y_pred, mu, logvar = self.model(image)
                
                kl_divergence = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
                loss = F.binary_cross_entropy(y_pred, image) + kl_divergence
                self.loss += loss.detach().cpu().item()
                
        self.loss /= len(self.test_ds)
        
        print(f"Test Average Loss: {self.loss:.4f}")
        
    def reconstruct_test_image(self, test_size: int = 64):
        test_loader = DataLoader(
            dataset=self.test_ds,
            batch_size=test_size,
            shuffle=False,
            collate_fn=custom_collator
        )
        
        for batch in test_loader:
            batch = [b.to(self.device) for b in batch]
            image, _ = batch
            outputs, mu, logvar = self.model(image)
            outputs = outputs.view(outputs.size(0), 3, 32, 32).detach().cpu().data
            save_image(
                tensor=image,
                fp=os.path.join(self.args.save_dir, 'cifar10_test_images.png')
            )
            save_image(
                tensor=outputs,
                fp=os.path.join(self.args.save_dir, 'cifar10_pred_images.png'))
            break