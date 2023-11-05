import torch 
import numpy as np

from models import MatrixFactorization
from dataset import load_dataset, get_dataloaders
from utils import get_loss_curve, get_output_files


def train(model, train_dataloader, val_dataloader, lr=0.001, num_epochs=30, device='cpu'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        epoch_train_losses = []
        epoch_val_losses = []
        
        model.train()
        
        for user_batch, item_batch, rating_batch in train_dataloader:
            user_batch = user_batch.to(device, dtype=torch.long)
            item_batch = item_batch.to(device, dtype=torch.long)
            rating_batch = rating_batch.to(device, dtype=torch.float)
            
            preds = model(user_batch, item_batch)
            loss = criterion(preds, rating_batch)
            epoch_train_losses.append(loss.item())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        model.eval()
        for user_batch, item_batch, rating_batch in val_dataloader:
            user_batch = user_batch.to(device, dtype=torch.long)
            item_batch = item_batch.to(device, dtype=torch.long)
            rating_batch = rating_batch.to(device, dtype=torch.float)
            
            preds = model(user_batch, item_batch)
            loss = criterion(preds, rating_batch)
            epoch_val_losses.append(loss.item())
            
        mean_epoch_train_loss = np.mean(epoch_train_losses)
        mean_epoch_val_loss = np.mean(epoch_val_losses)
        train_losses.append(mean_epoch_train_loss)
        val_losses.append(mean_epoch_val_loss)
        print(f'Epoch: {i}, Train Loss: {mean_epoch_train_loss}, Val Loss:{mean_epoch_val_loss}')

    get_loss_curve(train_losses=train_losses, val_losses=val_losses, outfile='init.png')


def get_test_df_preds(model, test):

    test_preds = []

    for i, row in test.iterrows():
        test_preds.append(model(torch.tensor(row.user_id), 
                                torch.tensor(row.item_id)).item())

    test['preds'] = test_preds


if __name__ == "__main__":
    train_df, val_df, test_df = load_dataset(train_csv='train_split.csv', val_csv='val_split.csv', test_csv='test_split.csv')
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_df, val_df, test_df, batch_size=512)
    
    # instantiate model
    num_factors = 50
    num_users = 943
    num_items = 1682

    model = MatrixFactorization(num_factors, num_users, num_items)
    train(model, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    get_test_df_preds(model, test_df)
    get_output_files(test_df, qrels_out='qrels', run_out='run', threshold=4.0)