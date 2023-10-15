import torch
import torch.optim as optim

from model import *
    

def get_optimizer(args, model):
    optim_name = args.optimizer.lower()
    
    if optim_name == "adagrad":
        optimizer = optim.Adagrad(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.eps
        )
    elif optim_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.eps
        )
    elif optim_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError
    
    return optimizer

        
def get_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()
    
    if scheduler_name == "lambdalr":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 0.95**epoch
        )
    elif scheduler_name == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.5
        )
    else:
        raise NotImplementedError
    
    return scheduler