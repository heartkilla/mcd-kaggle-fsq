import torch
from torch.utils.data import Dataset, DataLoader

import config


class FourSquareDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.fulltext = df['fulltext'].values
        self.latitudes = df['latitude'].values
        self.longitudes = df['longitude'].values
        self.coord_x = df['coord_x'].values
        self.coord_y = df['coord_y'].values
        self.coord_z = df['coord_z'].values
        self.labels = df['point_of_interest'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.fulltext)
    
    def __getitem__(self, index):
        fulltext = self.fulltext[index]
        latitude = self.latitudes[index]
        longitude = self.longitudes[index]
        label = self.labels[index]
        coord_x = self.coord_x[index]
        coord_y = self.coord_y[index]
        coord_z = self.coord_z[index]
        
        inputs = self.tokenizer(
            fulltext,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )

        return {
            'ids': inputs['input_ids'][0],
            'mask': inputs['attention_mask'][0],
            'latitude': torch.tensor(latitude, dtype=torch.float),
            'longitude': torch.tensor(longitude, dtype=torch.float),
            'coord_x': torch.tensor(coord_x),
            'coord_y': torch.tensor(coord_y),
            'coord_z': torch.tensor(coord_z),
            'label': torch.tensor(label, dtype=torch.long)
        }


def prepare_loaders(df_train, df_valid):
    # df_train = df[df.kfold != fold].reset_index(drop=True)
    # df_valid = df[df.kfold == fold].reset_index(drop=True)
    # df_train, df_valid = train_test_split(df, random_state=CFG.seed, shuffle=True, test_size=0.3)
    # df_valid = df_valid[df_valid.point_of_interest.isin(df_train.point_of_interest.unique())]
    
    train_dataset = FourSquareDataset(df_train, tokenizer=config.tokenizer, max_length=config.max_length)
    valid_dataset = FourSquareDataset(df_valid, tokenizer=config.tokenizer, max_length=config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader
