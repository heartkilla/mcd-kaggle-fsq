import gc
import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)

        latitude = data['latitude'].to(device, dtype=torch.float)
        longitude = data['longitude'].to(device, dtype=torch.float)
        coord_x = data['coord_x'].to(device, dtype=torch.float)
        coord_y = data['coord_y'].to(device, dtype=torch.float)
        coord_z = data['coord_z'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)
        
        outputs = model(ids, mask, latitude, longitude, coord_x, coord_y, coord_z, labels)
        loss = criterion(outputs, labels)
        loss = loss / config.n_accumulate
        loss.backward()
        
        grad = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    
        if (step + 1) % config.n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                       LR=optimizer.param_groups[0]['lr'], 
                       Grad=grad.item())
    gc.collect()
    
    return epoch_loss


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)

        latitude = data['latitude'].to(device, dtype=torch.float)
        longitude = data['longitude'].to(device, dtype=torch.float)
        coord_x = data['coord_x'].to(device, dtype=torch.float)
        coord_y = data['coord_y'].to(device, dtype=torch.float)
        coord_z = data['coord_z'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask, latitude, longitude, coord_x, coord_y, coord_z, labels)
        loss = criterion(outputs, labels)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss


def run_training(model, optimizer, scheduler, train_loader, valid_loader, device, num_epochs):
    # To automatically log gradients
    # wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    
    margins = np.linspace(config.m_start, config.m_end, num_epochs)
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        # margin を epoch ごとに更新する
        model.update_margin(margins[epoch - 1])
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=config.device, epoch=epoch)
        
        val_epoch_loss = valid_one_epoch(model, valid_loader, device=config.device, epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        # Log the metrics
        # wandb.log({"Train Loss": train_epoch_loss})
        # wandb.log({"Valid Loss": val_epoch_loss})
        
        # deep copy the model
        print(f"Validation Loss ({best_epoch_loss} ---> {val_epoch_loss})")
        best_epoch_loss = val_epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"{OUT_DIR}{config.model_name}_epoch{epoch}.bin"
        torch.save(model.state_dict(), PATH)
        # Save a model file from the current directory
        print(f"Model Saved")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history
