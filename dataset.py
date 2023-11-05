import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MovieDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.users = df['user_id']
        self.items = df['item_id']
        self.ratings = df['rating']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        user = self.users[index]
        item = self.items[index]
        rating = self.ratings[index]
        
        return user, item, rating


def load_dataset(train_csv='train_split.csv', val_csv='val_split.csv', test_csv='test_split.csv'):

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    return train, val, test
    

def get_dataloaders(train, val, test, batch_size=512):
    train_dataset = MovieDataset(train)
    val_dataset = MovieDataset(val)
    test_dataset = MovieDataset(test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader