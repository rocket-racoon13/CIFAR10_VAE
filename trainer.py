import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import custom_collator


class Trainer:
    def __init__(
        self,
        args,
        train_ds,
        test_ds,
        model,
        optimizer,
        scheduler,
        device
    ):
        self.args = args
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        
        self.device = device
        self.model = model
        self.loss_func = nn.MSELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.steps = 0
        self.train_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'log/train'))
        self.valid_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'log/valid'))
        self.best_loss = 100000000000
    
    def update_tensorboard(self, loss, mode="train"):
        if mode == "train":
            self.train_writer.add_scalar("Loss/train", loss, self.steps)
        elif mode == "valid":
            self.valid_writer.add_scalar("Loss/valid", loss, self.steps)
    
    def valid(self):
        total_loss = 0
        test_loader = DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=custom_collator
        )
        
        self.model.eval() # eval mode
        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(self.device) for b in batch]
                image, _ = batch
                y_pred = self.model(image)
                loss = self.loss_func(y_pred, image)
                total_loss += loss.detach().cpu().item()
                
        average_loss = total_loss / len(self.test_ds)
        
        self.update_tensorboard(
            loss=average_loss,
            mode="valid"
        )
        
        # Save Best Loss
        if average_loss < self.best_loss:
            self.best_loss = average_loss
            
            # Save Best Model Checkpoint
            torch.save({
                "steps": self.steps,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": average_loss
            }, os.path.join(self.args.save_dir, 'best-model.ckpt'))
    
    def train(self):
        self.model.train()
        
        for epoch in tqdm(range(1, self.args.num_epochs+1)):
            train_loader = DataLoader(
                dataset=self.train_ds,
                batch_size=self.train_batch_size,
                shuffle=True,
                collate_fn=custom_collator
            )
            
            for step, batch in enumerate(train_loader, 1):
                batch = [b.to(self.device) for b in batch]
                image, _ = batch
                y_pred = self.model(image)
                loss = self.loss_func(y_pred, image)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.steps += 1
                
                if step % self.args.logging_steps == 0:
                    print(f"Epoch:{epoch:2d} Batch:{step:2d} Loss:{loss:4.4f}")
                    
                    self.update_tensorboard(
                        loss=loss.detach().cpu().item(),
                        mode="train"
                    )

                # Save Latest Model Checkpoint
                if step % self.args.save_steps == 0:
                    torch.save({
                        "epochs": epoch,
                        "steps": step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, os.path.join(self.args.save_dir, "lastest-model.ckpt"))
                
            self.scheduler.step()
            
            # validation
            self.valid()
            
        self.train_writer.flush()
        self.valid_writer.flush()
        self.train_writer.close()
        self.valid_writer.close()