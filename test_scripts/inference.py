sys.path.append('../models/')
sys.path.append('../data_generator/')
from ga_transformer import AttackTransformer
from data_generator import DataLoader
import argparse
import yaml
import time


config_path = '/scratch/gpfs/bdedhia/attack_txf/exp_configs/config.yaml'
device = torch.device('cuda:0' if torch.cuda.is_available and ngpu==1 else "cpu")	
config = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
datapath = '/scratch/gpfs/bdedhia/attack_txf/dataset/'

dataloader = DataLoader(1,config['max_len']
							args.datapath,
							device,
							0)

model = AttackTransformer(block_dims=dataloader.dim,
							attack_classes=config['attack_classes'],
							hidden_size=config['hidden_size'],
							max_length=config['max_length'],
							max_seq_len=config['max_seq_len'],
							random_seed=args.random_seed,
							n_layer=config['n_layer'],
							n_head=config['n_head'],
							n_inner=4*config['n_layer'],
							resid_pdrop=config['dropout'],
							attn_pdrop=config['dropout'],
									)

model = model.to(device)
blocks, attacks, attention_masks, timestamps = self.dataloader.get_batch()

start_time = time.time()
pred_attacks = self.model(blocks, attacks, timestamps, attention_masks)
end_time = time.time()

print("Takes inference")