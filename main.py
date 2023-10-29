import argparse
from datetime import datetime

from torch.utils.data import DataLoader

from dataset import *
from data_utils import *
from model import VAE
from model_utils import *
from tester import *
from trainer import *
from utils import *


def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=77)
    
    parser.add_argument("--data_dir", type=str, default="dataset/cifar10")
    parser.add_argument("--save_dir", type=str, default=f"outputs/{datetime.now().strftime('%Y%m%d_%H-%M-%S')}/ckpt")
    parser.add_argument("--model_name", type=str)
    
    parser.add_argument('--image_width', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=32)
    parser.add_argument('--image_channel', type=int, default=3)
    
    parser.add_argument("--norm_mean", type=tuple, default=(0, 0, 0))
    parser.add_argument("--norm_stdev", type=tuple, default=(1, 1, 1))
    
    parser.add_argument('--conv_channels', type=list, default=[32, 64])
    parser.add_argument('--kernel_size', type=list, default=[3, 3])
    parser.add_argument('--enc_stride', type=int, default=[1, 1])
    parser.add_argument('--dec_stride', type=int, default=[1, 1])
    parser.add_argument('--enc_padding', type=list, default=[1, 1])
    parser.add_argument('--dec_padding', type=list, default=[1, 1])
    parser.add_argument('--latent_dim', type=int, default=256)
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--scheduler', type=str, default="lambdaLR")
    
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=100)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--reconstruct_image', action="store_true")
    parser.add_argument('--no_cuda', action="store_true")
    
    args = parser.parse_args()
    
    return args


def main(args):
    device = get_device(args)
    print(f"=== Device Type: {device} ===")

    if args.train:
        # create dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        log_dir = os.path.join(args.save_dir, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    train_ds = CIFAR10Dataset(args, train=True, transform=Normalize(args.norm_mean, args.norm_stdev))
    test_ds = CIFAR10Dataset(args, train=False, transform=Normalize(args.norm_mean, args.norm_stdev))
    
    # create model, optimizer, scheduler
    model = VAE(args).to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    # load model
    if args.model_name is not None:
        ckpt = torch.load(
            os.path.join(args.save_dir, args.model_name),
            map_location=device
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        steps = ckpt["steps"]
        print(f"=== {args.model_name} -> LOAD COMPLETE ===")
    
    # train
    if args.train:
        trainer = Trainer(
            args,
            train_ds=train_ds,
            test_ds=test_ds,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        trainer.train()
    
    # test
    if args.test:
        tester = Tester(
            args,
            test_ds=test_ds,
            model=model,
            optimizer=optimizer,
            device=device
        )
        tester.test()
        
    # reconstruct image
    if args.reconstruct_image:
        tester = Tester(
            args,
            train_ds=train_ds,
            test_ds=test_ds,
            model=model,
            optimizer=optimizer,
            device=device
        )
        tester.reconstruct_test_image()
    

if __name__ == "__main__":
    args = config()
    main(args)