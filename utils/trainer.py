import os
import csv
import torch
import numpy as np


class trainer:

    def __init__(self, model, optimizer, criterion, scheduler,
                 train_loader, val_loader, device,
                 output_dir='outputs', resume_ckpt=None):
        
        self.model = model
        self.opt = optimizer
        self.criterion = criterion
        self.sched = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)

        self.start_epoch = 0
        self.best_val_loss = np.inf

        if resume_ckpt:
            print(f"Loading checkpoint from {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location=device)
            self.model.load_state_dict(ckpt['model_state'])
            self.opt.load_state_dict(ckpt['opt_state'])
            self.sched.load_state_dict(ckpt['sched_state'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_val_loss = ckpt.get('best_val_loss', self.best_val_loss)

        # CSV Logger
        self.log_path = os.path.join(self.output_dir, 'logs', 'metrics.csv')
        if self.start_epoch == 0:
            with open(self.log_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr'])


    def train(self, epochs):

        for epoch in range(self.start_epoch, epochs):
            train_loss = self._train_one_epoch()
            val_loss, val_acc = self._validate()
            lr = self.sched.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs}: Train {train_loss:.4f}, Val {val_loss:.4f}, Acc {val_acc:.2f}, LR {lr:.6f}")

            # Logging
            with open(self.log_path, 'a') as f:
                csv.writer(f).writerow([epoch+1, train_loss, val_loss, val_acc, lr])

            # Checkpoint
            ckpt = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'opt_state': self.opt.state_dict(),
                'sched_state': self.sched.state_dict(),
                'best_val_loss': self.best_val_loss
            }
            torch.save(ckpt, os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch+1:03d}.pth'))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(ckpt, os.path.join(self.output_dir, 'checkpoints', 'best.pth'))

            self.sched.step()

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.opt.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(self.train_loader.dataset)

    def _validate(self):
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
        val_loss = total_loss / len(self.val_loader.dataset)
        val_acc = correct / len(self.val_loader.dataset)
        return val_loss, val_acc

