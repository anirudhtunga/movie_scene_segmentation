import torch.optim as optim
import tqdm
import os
import json
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers.activations import ACT2FN, gelu
from sklearn.metrics import average_precision_score

from dataset import SegDataset
from model import MultimodalLongFormer


def train(max_epochs,model,sdata,test_data):
    patience = 10
    lr_patience = 2
    
    checkpoint_path = "./models"
    
    
    train_loader = DataLoader(sdata, batch_size=1,
                        shuffle=True, num_workers=0)


    
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
          optimizer, "max", patience=lr_patience, verbose=True, factor=0.5
      )
    
    n_no_improve, best_metric = 0, -np.inf

    model.cuda()

    for i_epoch in range(0, max_epochs):
        
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for i,batch in enumerate(train_loader):
            
            
            input_feat, coarse, mask, global_mask, labels, ids= batch
            


            input_feat= input_feat.cuda()
            coarse=coarse.cuda()
            mask=mask.cuda()
            global_mask=global_mask.cuda()
            labels=labels.cuda()

            logits = model(input_feat, coarse, mask, global_mask)

            loss = criterion(logits.view(-1),labels.view(-1))



            train_losses.append(loss.item())
            loss.backward()


            optimizer.step()
            optimizer.zero_grad()
        

        print("Epoch:{} and Loss:{}".format(i_epoch,sum(train_losses)))
        model.eval()
        print("Train")
        metrics = model_eval(model,sdata)
        print("Test")
        test_metric,_,_ = model_eval(model,test_data)
        
        scheduler.step(test_metric)
        
        is_improvement = test_metric > best_metric
        
        if is_improvement:
            best_metric = test_metric
            n_no_improve = 0
            filename ="model_{:.2f}.pth".format(best_metric*100)
            filename = os.path.join(checkpoint_path, filename)
            torch.save(model, filename)
        else:
            n_no_improve += 1
        
        
        if n_no_improve >= patience:
            print("No improvement. Breaking out of loop.")
            break
    print("Best_metric:",best_metric)
            
    return test_metric


# In[ ]:


def model_eval(model,sdata):
    
    loader = DataLoader(sdata, batch_size=1,
                        shuffle=True, num_workers=0)
    with torch.no_grad():
        
        preds, tgts = [], []
        
        for batch in loader:
            input_feat, coarse, mask, global_mask, labels, ids= batch
            


            input_feat= input_feat.cuda()
            coarse=coarse.cuda()
            mask=mask.cuda()
            global_mask=global_mask.cuda()
            labels=labels.cuda()
            
            logits = model(input_feat, coarse, mask, global_mask)
            
            pred = torch.sigmoid(logits).cpu().detach().numpy()
            
            preds.append(pred)
            tgt = labels.cpu().detach().numpy()
            tgts.append(tgt)
        
        metrics = calc_ap(tgts,preds)
        
    return metrics
            
def calc_ap(gt_list, pr_list):
    """Average Precision (AP) for scene transitions.

    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.

    Returns:
        AP, mean AP, and a dict of AP for each movie.
    """
    


    AP_dict = dict()
    gt = list()
    pr = list()
    for imdb_id in range(len(gt_list)):
        AP_dict[imdb_id] = average_precision_score(gt_list[imdb_id][0], pr_list[imdb_id][0])
        gt.append(gt_list[imdb_id][0])
        pr.append(pr_list[imdb_id][0])

    mAP = sum(AP_dict.values()) / len(AP_dict)

    gt = np.concatenate(gt)
    pr = np.concatenate(pr)
    AP = average_precision_score(gt, pr)
    
    print("AP:{} | mAP:{}".format(AP,mAP))

    return AP, mAP, AP_dict        
            
    


if __name__ == "__main__":

    with open('train.json', 'rb') as f:
        train_files = json.load(f)
    
    with open('test.json', 'rb') as f:
        test_files = json.load(f)


    MAX_EPOCHS = 20
    model = MultimodalLongFormer()
    train_data = SegDataset(train_files)
    test_data = SegDataset(test_files)

    metric = train(MAX_EPOCHS,model,train_data,test_data)