import torch
import torch.nn as nn
import os
import sys
import numpy as np
sys.path.append('../models/')
sys.path.append('../trainer/')
sys.path.append('../data_generator/')
from ga_transformer import AttackTransformer
from trainer import Trainer
from data_generator import DataLoader
import argparse
import yaml



def run_experiment(args, ngpu=1, num_iters =5e5):

	device = torch.device('cuda:0' if torch.cuda.is_available and ngpu==1 else "cpu")	
	config = yaml.load(open(args.config,'r'),Loader=yaml.FullLoader)

	if not(os.path.exists(args.op_path)):
		os.mkdir(args.op_path)

	dataloader = DataLoader(config['batch_size'],
							config['max_len']
							args.datapath,
							device,
							args.random_seed,
							)

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
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	model = model.to(device)

	

	optimizer = torch.optim.AdamW(model.parameters(),
								lr=eval(config['lr']),
								weight_decay=eval(config['weight_decay']),
								)
	warmup_steps = config['warmup_steps']
	scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer,
				lambda steps: min((steps+1)/warmup_steps,1))
	batch_size = config['batch_size']
	trainer = Trainer(model,
					optimizer,
					scheduler,
					dataloader,
					batch_size,
					args.op_path,
					args.random_seed)

	trainer.train(int(num_iters),args.checkpoint)


def main():

	'''Trains Evolutionary Transformer'''

	parser = argparse.ArgumentParser(description='Experiment parameters',
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--datapath',type=str,default='')
	parser.add_argument('--config',type=str, default='')
	parser.add_argument('--op_path',type=str,default='')
	parser.add_argument('--random_seed',type=int,default=0)
	parser.add_argument('--checkpoint',type=str,default=None)
	args = parser.parse_args()
	run_experiment(args)

if __name__ == '__main__'	:
	main()



