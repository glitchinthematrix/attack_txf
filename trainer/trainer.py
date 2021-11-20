import torch
import numpy as np
import torch.nn as nn

class Trainer:

	def __init__(self, model, optimizer, scheduler, dataloader, batch_size, output_path, random_seed= 0, log_freq = 100):

		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.dataloader = dataloader
		self.log_freq = log_freq
		self.loss_fn = nn.BCEWithLogitsLoss()
		self.op_path = output_path

	def train(self, numsteps, load_from_checkpoint = None):

		self.model.train()
		log = dict()
		start_step = 0

		if load_from_checkpoint is not None:

			#load model, optimizer, scheduler, epoch_step
			checkpoint = torch.load(self.op_path+'model.pth')
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optim_state_dict'])
			log = checkpoint['logs']
			start_step = checkpoint['iter']+1
			print('Checkpoint loaded')

		loss_accumulate = []

		for i in range(start_step, start_step+numsteps):

			blocks, attacks, attention_masks, timestamps = self.dataloader.get_batch()
			target_attacks = torch.clone(attacks).to(attacks.get_device())
			pred_attacks = self.model(blocks, attacks, timestamps, attention_masks)
			loss = self.loss_fn_cont(pred_attacks[attention_masks],target_attacks[attention_masks])
			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(self.model.parameters(),0.25)
			self.optimizer.step()
			loss_accumulate.append(loss.item())
			if self.scheduler is not None:
				self.scheduler.step()
			if i%self.log_freq == 0:
				loss_accumulate = np.array(loss_accumulate)
				log[str(i)] = {'train_loss_mean':np.mean(loss_accumulate),'train_loss_std':np.std(loss_accumulate)}
				print(f'Iteration:{i}\tLoss_mean:{np.mean(loss_accumulate)}\tLoss_std:{np.std(loss_accumulate)}')
				loss_accumulate = []
				checkpoint = {'model_state_dict': self.model.state_dict(),
							'optim_state_dict':self.optimizer.state_dict(),
							'logs':log,
							'iter':i}
				if self.scheduler is not None:
					checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
				torch.save(checkpoint,self.op_path+'model.pth')


