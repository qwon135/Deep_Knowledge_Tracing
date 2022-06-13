import os

import torch
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess,add_features


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args = add_features(args)    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    
    test_data = preprocess.get_test_data()

    trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="test")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
