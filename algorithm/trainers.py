import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Generator
import algorithm.utils as utils
from algorithm.datasets import MaskedDataset
from algorithm.estimators import DualT

class Trainer(object):
    def __init__(self) -> None:
        """
        Initialize the trainer.
        """
        # Inialize the lists to store the loss and accuracy
        self.loss_list = []
        self.acc_list = []
        self.valid_loss_list = [np.inf] # Initialize the validation loss list with infinity
        self.valid_acc_list = [0]
        # Set the early stopping setting to None
        self.early_stop_counter = None

    def early_stopp_setting(self, early_stop_threshold=3, frequency=3) -> None:
        """
        Store the early stopping setting.
        """
        # Initialize the early stopping counter
        self.early_stop_counter = 0
        # Store the early stopping setting
        self.early_stop_dict = {'early_stop_threshold': early_stop_threshold,
                                'frequency': frequency}

    def early_stop_checker(self, valid_loss_avg: float, valid_acc_avg: float, monitor='accuracy') -> bool:
        """
        Check if the early stopping condition is met.

        Parameters:
        valid_loss_avg (float): The average validation loss.
        valid_acc_avg (float): The average validation accuracy.

        Returns:
        bool: Whether the early stopping condition is met.
        """
        if monitor == 'accuracy':
            valid_avg = valid_acc_avg
            valid_list = self.valid_acc_list
        elif monitor == 'loss':
            valid_avg = valid_loss_avg
            valid_list = self.valid_loss_list
        if valid_avg < valid_list[-1]:
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        self.valid_loss_list.append(valid_loss_avg)
        self.valid_acc_list.append(valid_acc_avg)
        return self.early_stop_counter >= self.early_stop_dict['early_stop_threshold']
    
    def print_early_stop_message(self, print_early_stopping):
        if print_early_stopping:
            print("\nModel training finished due to early stopping.")
            
    def plot_curve(self) -> None:
        """
        Plot the loss and accuracy curves.
        """
        # Visualize the metrics after training
        _, axs = plt.subplots(1, 2, figsize=(12, 4))
        # Visualize the cross entropy loss curve
        axs[0].plot(self.loss_list, color='red', label='Training Loss')
        axs[0].plot([i*self.early_stop_dict['frequency'] for i in range(len(self.valid_loss_list))], self.valid_loss_list, color='blue', label='Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss during Training')
        axs[0].legend(loc='best')
        # Visualize the accuracy curve
        axs[1].plot(self.acc_list, color=(1.0, 0.5, 0.0), label='Training Accuracy')
        axs[1].plot([i*self.early_stop_dict['frequency'] for i in range(len(self.valid_acc_list))], self.valid_acc_list, color='blue', label='Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy during Training')
        axs[1].legend(loc='best')
        plt.show()

    def conditional_tqdm(self, iterable: range, use_progress_bar: bool=False) -> Generator[int, None, None]:
        """
        Determine whether to use tqdm or not based on the use_progress_bar flag.

        Parameters:
        - iterable (range): Range of values to iterate over.
        - use_progress_bar (bool, optional): Whether to print progress bar. Default is False.

        Returns:
        - item (int): Current iteration.
        """

        if use_progress_bar:
            for item in tqdm(iterable):
                yield item
        else:
            for item in iterable:
                yield item

class ForwardLossCorrection(Trainer):
    def train(self, model, train_dataset, valid_dataset, epochs=20, 
              T=None, batch_sizes=[128, 128], adjust_threshold=5,
              show_training_curve=True, show_progress_bar=True, print_early_stopping=True,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if self.early_stop_counter is None:
            self.early_stopp_setting()
        # Define the criterion to save the model
        best_valid_acc = float('-inf')
        best_model_state = None
        # Define the loss function
        criterion = utils.SymmetricCrossEntropyLoss()
        # Set the model to train mode
        model.train()
        # Put the model to the device
        model.to(device)
        # Define the initial learning rate and the number of warmup epochs
        initial_lr = 1e-7
        warmup_epochs = 5
        # Define the optimizer for the model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # Define the learning rate scheduler for the optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # Define the data loaders
        train_loader = data.DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=2)
        valid_loader = data.DataLoader(valid_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=2)
        adjust = 0
        if T is None:
            T_tensor = None
            known = False
        else:
            self.T = T
            # Convert the inverse of transition matrix to a torch tensor
            T_tensor = torch.from_numpy(T).float().to(device)
            known = True
        
        for ep in self.conditional_tqdm(range(epochs), show_progress_bar):
            ep_loss = 0.0
            ep_acc = 0.0
            for step, (_, x, y) in enumerate(train_loader):
                # Learning rate warmup
                if ep < warmup_epochs:
                    # Calculate the learning rate for the current epoch
                    warmup_lr = initial_lr + (1e-5 - initial_lr) * (ep / warmup_epochs)
                    # Update optimizer learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                x = x.float().to(device)
                y = y.to(device)
                # Clear the gradients
                optimizer.zero_grad()
                # Compute the output
                logits = model(x)
                # Convert logits to log probabilities
                log_probs = F.log_softmax(logits, dim=1)
                # Forward correction on the logits using T
                if T_tensor is not None:
                    p = torch.matmul(log_probs, T_tensor)
                else:
                    p = log_probs
                # Use negative log likelihood loss
                loss = criterion(p, y)
                # Compute the accuracy
                acc = utils.accuracy(p, y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                ep_acc += acc.item()
            if ep >= warmup_epochs:
                scheduler.step(ep_loss)
            # Append the loss and accuracy to the lists
            self.loss_list.append(ep_loss/(step+1))
            self.acc_list.append(ep_acc/(step+1))

            if ep % self.early_stop_dict['frequency'] == 0:
                # Validation
                model.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                with torch.no_grad():
                    for _, x, y in valid_loader:
                        x = x.float().to(device)
                        y = y.to(device)
                        logits = model(x)
                        # Convert logits to probabilities
                        p = F.log_softmax(logits, dim=1)
                        loss = criterion(p, y)
                        acc = utils.accuracy(p, y)
                        valid_loss += loss.item()
                        valid_acc += acc.item()
                # Compute the average loss and accuracy
                valid_loss_avg = valid_loss / len(valid_loader)
                valid_acc_avg = valid_acc / len(valid_loader)
                flag = self.early_stop_checker(valid_loss_avg, valid_acc_avg, 'accuracy')
                # Update best validation accuracy and model state
                if valid_acc_avg > best_valid_acc:
                    best_valid_acc = valid_acc_avg
                    best_model_state = model.state_dict()
                model.train()
            # Print it if the early stopping works
            if flag:
                if T is None:
                    T_tensor = self.update_T(model, train_loader, device)
                    adjust += 1
                else:
                    if not known:
                        T_old = self.T
                        T_tensor = self.update_T(model, train_loader, device)
                        adjust += 1
                        diff = np.linalg.norm(self.T - T_old)
                    if known or diff < 1e-3 or adjust >= adjust_threshold:
                        self.print_early_stop_message(print_early_stopping)
                        break
        
        if T is None:
            self.T = self.update_T(model, train_loader, device)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        if show_training_curve:
            self.plot_curve()
    
    def update_T(self, model, train_loader, device):
        T = DualT().estimate(model, train_loader).T
        self.T = T
        return torch.from_numpy(T).float().to(device)

class CoTeaching(Trainer):
    def train(self, model1, model2, train_dataset, valid_dataset, epochs=20, T=None, batch_sizes=[128, 128],
              show_training_curve=True, show_progress_bar=True, print_early_stopping=True,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if self.early_stop_counter is None:
            self.early_stopp_setting()
        best_valid_acc = float('-inf')
        best_model1_state = None
        best_model2_state = None
        # Set the model to train mode
        model1.train()
        model2.train()
        # Put the model to the device
        model1.to(device)
        model2.to(device)
        # Define the optimizer for the model
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-5)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-5)
        # Define the learning rate scheduler for the optimizer
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2)
        # Compute the select ratio
        select_ratio = T.max() if T is not None else 0.33
        # Define the data loaders
        train_loader = data.DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=2)
        valid_loader = data.DataLoader(valid_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=2)

        for ep in self.conditional_tqdm(range(epochs), show_progress_bar):
            ep_loss = 0.0
            ep_acc = 0.0
            for step, (_, x, y) in enumerate(train_loader):
                x = x.float().to(device)
                y = y.to(device)
                # Compute the key components
                loss1_selected, loss2_selected, acc1, acc2 = self.helper_co_teaching(model1, model2, x, y, select_ratio)
                # Backpropagation and optimization for model1
                optimizer1.zero_grad()
                loss1_selected.mean().backward()
                optimizer1.step()
                # Backpropagation and optimization for model2
                optimizer2.zero_grad()
                loss2_selected.mean().backward()
                optimizer2.step()
                ep_loss += (loss1_selected.mean().item() + loss2_selected.mean().item()) / 2
                ep_acc += (acc1.item() + acc2.item()) / 2
            # Update learning rates
            scheduler1.step(ep_loss)
            scheduler2.step(ep_loss)
            # Append the loss and accuracy to the lists
            self.loss_list.append(ep_loss/(step+1))
            self.acc_list.append(ep_acc/(step+1))

            if ep % self.early_stop_dict['frequency'] == 0:
                # Validation
                model1.eval()
                model2.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                with torch.no_grad(): # No need to compute the gradients
                    for _, x, y in valid_loader:
                        x = x.float().to(device)
                        y = y.to(device)
                        loss1, loss2, acc1, acc2 = self.helper_co_teaching(model1, model2, x, y, select_ratio)
                        valid_loss += (loss1.mean().item() + loss2.mean().item()) / 2
                        valid_acc += (acc1.item() + acc2.item()) / 2
                # Compute the average loss and accuracy
                valid_loss_avg = valid_loss / len(valid_loader)
                valid_acc_avg = valid_acc / len(valid_loader)
                if valid_acc_avg > best_valid_acc:
                    best_valid_acc = valid_acc_avg
                    best_model1_state = model1.state_dict()
                    best_model2_state = model2.state_dict()
                flag = self.early_stop_checker(valid_loss_avg, valid_acc_avg)
                model1.train()
                model2.train()
            # Print it if the early stopping works
            if flag:
                self.print_early_stop_message(print_early_stopping)
                break
        if best_model1_state is not None:
            model1.load_state_dict(best_model1_state)
            model2.load_state_dict(best_model2_state)
        if show_training_curve:
            self.plot_curve()

    def helper_co_teaching(self, model1, model2, x, y, select_ratio):
        """
        Compute the key components for the co-teaching algorithm.

        Parameters:
        model1 (torch.nn.Module): The first model.
        model2 (torch.nn.Module): The second model.
        x (torch.Tensor): The input.
        y (torch.Tensor): The target.
        select_ratio (float): The select ratio.

        Returns:
        loss1_selected (torch.Tensor): The selected loss for the first model.
        loss2_selected (torch.Tensor): The selected loss for the second model.
        loss (float): The loss.
        acc (float): The accuracy.
        """
        # Define the loss function
        criterion = nn.CrossEntropyLoss(reduction='none')
        # Compute outputs for both models
        logits1 = model1(x)
        logits2 = model2(x)
        # Compute losses for both models
        per_instance_loss2 = criterion(logits2, y)
        per_instance_loss1 = criterion(logits1, y)
        # Select a subset of data for each model based on the loss values
        k = int(select_ratio * len(y))
        idx1 = per_instance_loss2.topk(k, largest=False)[1]
        idx2 = per_instance_loss1.topk(k, largest=False)[1]
        loss1_selected = criterion(logits1[idx1], y[idx1])
        loss2_selected = criterion(logits2[idx2], y[idx2])
        # Compute accuracy for both models
        acc1 = utils.accuracy(logits1[idx1], y[idx1])
        acc2 = utils.accuracy(logits2[idx2], y[idx2])
        return loss1_selected, loss2_selected, acc1, acc2

class O2UNet(Trainer):
    def train(self, model1, model2, train_dataset, valid_dataset, epochs=[25, 100, 100], T=None, batch_size=[128, 16, 128],
              show_training_curve=True, show_progress_bar=True, print_training_start=True, print_early_stopping=True,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.pre_train(model1, train_dataset, valid_dataset, epochs=epochs[0], T=T, batch_sizes=[batch_size[0]] * 2, 
                       show_progress_bar=show_progress_bar, print_training_start=print_training_start, 
                       show_training_curve=show_training_curve, print_early_stopping=print_early_stopping, device=device)
        mask = self.cyclical_train(model2, train_dataset, epochs=epochs[1], T=T, batch_size=batch_size[1],
                                   show_progress_bar=show_progress_bar, print_training_start=print_training_start, 
                                   device=device)
        masked_train_dataset = MaskedDataset(train_dataset, mask)
        self.clean_data_train(model2, masked_train_dataset, valid_dataset, epochs=epochs[2], batch_sizes=[batch_size[2]] * 2, 
                              show_training_curve=show_training_curve, show_progress_bar=show_progress_bar, 
                              print_training_start=print_training_start, print_early_stopping=print_early_stopping, 
                              device=device)
        
    def pre_train(self, model, train_dataset, valid_dataset, epochs=20, T=None, batch_sizes=[128, 128], 
              show_training_curve=True, show_progress_bar=True, print_training_start=True, print_early_stopping=True, 
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if self.early_stop_counter is None:
            self.early_stopp_setting()
        # Define the criterion to save the model
        best_valid_acc = float('-inf')
        best_model_state = None
        # Define the loss function
        criterion = utils.SymmetricCrossEntropyLoss()
        # Set the model to train mode
        model.train()
        # Put the model to the device
        model.to(device)
        # Define the initial learning rate and the number of warmup epochs
        initial_lr = 1e-7
        warmup_epochs = 5
        # Define the optimizer for the model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # Define the learning rate scheduler for the optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # Define the data loaders
        train_loader = data.DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=2)
        valid_loader = data.DataLoader(valid_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=2)
        adjust = 0
        if T is None:
            T_tensor = None
            known = False
        else:
            self.T = T
            # Convert the inverse of transition matrix to a torch tensor
            T_tensor = torch.from_numpy(T).float().to(device)
            known = True
        
        if print_training_start:
            print("Pre-Training Step Starts.")
            
        for ep in self.conditional_tqdm(range(epochs), show_progress_bar):
            ep_loss = 0.0
            ep_acc = 0.0
            for step, (_, x, y) in enumerate(train_loader):
                # Learning rate warmup
                if ep < warmup_epochs:
                    # Calculate the learning rate for the current epoch
                    warmup_lr = initial_lr + (1e-5 - initial_lr) * (ep / warmup_epochs)
                    # Update optimizer learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                x = x.float().to(device)
                y = y.to(device)
                # Clear the gradients
                optimizer.zero_grad()
                # Compute the output
                logits = model(x)
                # Convert logits to log probabilities
                log_probs = F.log_softmax(logits, dim=1)
                # Forward correction on the logits using T
                if T_tensor is not None:
                    p = torch.matmul(log_probs, T_tensor)
                else:
                    p = log_probs
                # Use negative log likelihood loss
                loss = criterion(p, y)
                # Compute the accuracy
                acc = utils.accuracy(p, y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                ep_acc += acc.item()
            if ep >= warmup_epochs:
                scheduler.step(ep_loss)
            # Append the loss and accuracy to the lists
            self.loss_list.append(ep_loss/(step+1))
            self.acc_list.append(ep_acc/(step+1))

            if ep % self.early_stop_dict['frequency'] == 0:
                # Validation
                model.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                with torch.no_grad():
                    for _, x, y in valid_loader:
                        x = x.float().to(device)
                        y = y.to(device)
                        logits = model(x)
                        # Convert logits to probabilities
                        p = F.log_softmax(logits, dim=1)
                        loss = criterion(p, y)
                        acc = utils.accuracy(p, y)
                        valid_loss += loss.item()
                        valid_acc += acc.item()
                # Compute the average loss and accuracy
                valid_loss_avg = valid_loss / len(valid_loader)
                valid_acc_avg = valid_acc / len(valid_loader)
                flag = self.early_stop_checker(valid_loss_avg, valid_acc_avg, 'accuracy')
                # Update best validation accuracy and model state
                if valid_acc_avg > best_valid_acc:
                    best_valid_acc = valid_acc_avg
                    best_model_state = model.state_dict()
                model.train()
            # Print it if the early stopping works
            if flag:
                if T is None:
                    T_tensor = self.update_T(model, train_loader, device)
                    adjust += 1
                else:
                    if not known:
                        T_old = self.T
                        T_tensor = self.update_T(model, train_loader, device)
                        adjust += 1
                        diff = np.linalg.norm(self.T - T_old)
                    if known or diff < 1e-3 or adjust >= 5:
                        self.print_early_stop_message(print_early_stopping)
                        break
        
        if T is None:
            self.T = DualT().estimate(model, train_loader)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        if show_training_curve:
            self.plot_curve()
    
    def update_T(self, model, train_loader, device):
        T = DualT().estimate(model, train_loader)
        self.T = T
        return torch.from_numpy(T).float().to(device)
        
    def cyclical_train(self, model, train_dataset, epochs=100, T=None, batch_size=16,
                       show_progress_bar=True, print_training_start=True, 
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Perform the cyclical training step of the O2U-Net.
        """
        model.train()
        model.to(device)
        # Define the loss function
        criterion = nn.CrossEntropyLoss() 
        # Define the optimizer for the model
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        # Define the learning rate scheduler for the optimizer
        scheduler = utils.CyclicalLR(optimizer)
        # Define the data loaders
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        record_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
        # Compute the k percentage
        k_percentage = (1 - T.max()) if T is not None else 0.1

        if print_training_start:
            print("Cyclical Training Step Starts.")
        
        sample_losses = np.zeros((epochs, len(train_dataset)))
        for ep in self.conditional_tqdm(range(epochs), show_progress_bar):
            scheduler.step()
            for (_, x, y) in train_loader:
                x = x.float().to(device)
                y = y.to(device)
                optimizer.zero_grad()
                p = model(x)
                # Calculate loss
                loss = criterion(p, y)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                for (idx, x, y) in record_loader:
                    x = x.float().to(device)
                    y = y.to(device)
                    p = model(x)
                    # Calculate the individual loss
                    individual_loss = criterion(p, y)
                    # Convert loss to a NumPy array after detaching from the computational graph
                    individual_loss = individual_loss.detach().cpu().numpy()
                    # Store the individual loss in the corresponding location
                    sample_losses[ep, idx] = individual_loss
            # Normalize the sample losses
            max_loss = np.max(sample_losses[ep])
            sample_losses[ep] = sample_losses[ep] / max_loss
        
        # Compute the normalized average loss ln of every sample in all the epochs
        normalized_average_losses = np.mean(sample_losses, axis=0)
        # Rank all the samples in descending order based on their normalized average losses
        sorted_indices = np.argsort(normalized_average_losses)[::-1]
        num_to_remove = int(len(sorted_indices) * k_percentage)
        # Identify the indices of the noisy samples
        noisy_samples_indices = sorted_indices[:num_to_remove]
        # Return a mask where the noisy samples are marked as False
        mask = np.ones(len(train_dataset), dtype=bool)
        mask[noisy_samples_indices] = False
        return mask

    def clean_data_train(self, model, train_dataset, valid_dataset, epochs=300, batch_sizes=[128, 128],
                         show_training_curve=True, show_progress_bar=True, print_training_start=True, print_early_stopping=True, 
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        """
        super().__init__()
        if self.early_stop_counter is None:
            self.early_stopp_setting()
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        # Set the model to train mode
        model.train()
        # Put the model to the device
        model.to(device)
        # Define the initial learning rate and the number of warmup epochs
        initial_lr = 1e-7
        warmup_epochs = 5
        # Define the optimizer for the model
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        # Define the learning rate scheduler for the optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # Define the data loaders
        train_loader = data.DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=2)
        valid_loader = data.DataLoader(valid_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=2)
        
        if print_training_start:
            print('Training on clean data step starts.')
        for ep in self.conditional_tqdm(range(epochs), show_progress_bar):
            ep_loss = 0.0
            ep_acc = 0.0
            for step, (_, x, y) in enumerate(train_loader):
                # Learning rate warmup
                if ep < warmup_epochs:
                    # Calculate the learning rate for the current epoch
                    warmup_lr = initial_lr + (1e-5 - initial_lr) * (ep / warmup_epochs)
                    # Update optimizer learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                x = x.float().to(device)
                y = y.to(device)
                # Clear the gradients
                optimizer.zero_grad()
                # Compute the output
                p = model(x)
                # Compute the loss
                loss = criterion(p, y)
                # Compute the accuracy
                acc = utils.accuracy(p, y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                ep_acc += acc.item()
            if ep >= warmup_epochs:
                scheduler.step(ep_loss)
            # Append the loss and accuracy to the lists
            self.loss_list.append(ep_loss/(step+1))
            self.acc_list.append(ep_acc/(step+1))

            if ep % self.early_stop_dict['frequency'] == 0:
                # Validation
                model.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                with torch.no_grad():
                    for _, x, y in valid_loader:
                        x = x.float().to(device)
                        y = y.to(device)
                        p = model(x)
                        loss = criterion(p, y)
                        acc = utils.accuracy(p, y)
                        valid_loss += loss.item()
                        valid_acc += acc.item()
                # Compute the average loss and accuracy
                valid_loss_avg = valid_loss / len(valid_loader)
                valid_acc_avg = valid_acc / len(valid_loader)
                flag = self.early_stop_checker(valid_loss_avg, valid_acc_avg)
                model.train()
            # Print it if the early stopping works
            if flag:
                self.print_early_stop_message(print_early_stopping)
                break

        if show_training_curve:
            self.plot_curve()

class JoCoR(Trainer):
    def train(self, model1, model2, train_dataset, valid_dataset, epochs=200, T=None, batch_sizes=[128, 128], lambda_=0.9, T_k=10,
              show_training_curve=True, show_progress_bar=True, print_early_stopping=True,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        if self.early_stop_counter is None:
            self.early_stopp_setting()
        # Set the model to train mode
        model1.train()
        model2.train()
        # Put the model to the device
        model1.to(device)
        model2.to(device)
        # Define the optimizer for the model
        optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3)
        # Define the learning rate scheduler rule for the optimizer
        decay_start_epoch = 80
        decay_end_epoch = 200
        total_decay_epochs = decay_end_epoch - decay_start_epoch
        lambda_lr = lambda epoch: 1 - max(0, epoch - decay_start_epoch) / total_decay_epochs if epoch < decay_end_epoch else 0
        # Define the learning rate scheduler for the optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        # Define the loss function
        criterion = nn.CrossEntropyLoss(reduction='none')
        # Define the noise ratio
        noise_ratio = (1 - T.max()) if T is not None else 0.66
        # Define the data loaders
        train_loader = data.DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=2)
        valid_loader = data.DataLoader(valid_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=2)

        for ep in self.conditional_tqdm(range(epochs), show_progress_bar):
            ep_loss = 0.0
            ep_acc = 0.0
            for step, (_, x, y) in enumerate(train_loader):
                x = x.float().to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits1 = model1(x)
                logits2 = model2(x)
                loss = self.compute_joint_loss(logits1, logits2, y, lambda_)
                if ep < 1:
                  L = loss
                else:
                  # Select a subset of data for each model based on the loss values
                  num_selected = int(R_t * len(x))
                  idx = loss.topk(num_selected, largest=False)[1]
                  L = loss[idx]
                L.mean().backward()
                optimizer.step()
                acc1 = utils.accuracy(logits1, y)
                acc2 = utils.accuracy(logits2, y)
                ep_loss += L.mean().item() / 2
                ep_acc += (acc1.item() + acc2.item()) / 2
            # Update the select ratio
            temp = [1, (ep + 1) / T_k]
            R_t = 1 - min(temp) * noise_ratio
            # Update learning rates
            scheduler.step()
            # Append the loss and accuracy to the lists
            self.loss_list.append(ep_loss/(step+1))
            self.acc_list.append(ep_acc/(step+1))

            if ep % self.early_stop_dict['frequency'] == 0:
                # Validation
                model1.eval()
                model2.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                with torch.no_grad(): # No need to compute the gradients
                    for _, x, y in valid_loader:
                        x = x.float().to(device)
                        y = y.to(device)
                        logits1 = model1(x)
                        logits2 = model2(x)
                        loss = self.compute_joint_loss(logits1, logits2, y, lambda_)
                        acc1 = utils.accuracy(logits1, y)
                        acc2 = utils.accuracy(logits2, y)
                        valid_loss += loss.mean().item()
                        valid_acc += (acc1.item() + acc2.item()) / 2
                # Compute the average loss and accuracy
                valid_loss_avg = valid_loss / len(valid_loader)
                valid_acc_avg = valid_acc / len(valid_loader)
                flag = self.early_stop_checker(valid_loss_avg, valid_acc_avg)
                model1.train()
                model2.train()
            # Print it if the early stopping works
            if flag:
                if print_early_stopping:
                    print("\nModel training finished due to early stopping.")
                break
        if show_training_curve:
            self.plot_curve()

    def compute_joint_loss(self, logits1, logits2, y, lambda_):
      # Use log-probabilities for numerical stability
      p1_log_prob = F.log_softmax(logits1, dim=1)
      p2_log_prob = F.log_softmax(logits2, dim=1)
      
      # Compute supervised loss using log-probabilities
      l_c1 = F.nll_loss(p1_log_prob, y, reduction='none')
      l_c2 = F.nll_loss(p2_log_prob, y, reduction='none')
      l_sup = l_c1 + l_c2
      
      # Compute KL divergence using log-probabilities directly
      D_KL_1 = torch.sum(torch.exp(p1_log_prob) * (p1_log_prob - p2_log_prob), dim=1)
      D_KL_2 = torch.sum(torch.exp(p2_log_prob) * (p2_log_prob - p1_log_prob), dim=1)
      l_con = D_KL_1 + D_KL_2

      return (1 - lambda_) * l_sup + lambda_ * l_con