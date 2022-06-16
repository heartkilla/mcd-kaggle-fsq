import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
import models
import utils


warnings.filterwarnings("ignore")
# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run():
    if not os.path.exists(config.OUT_DIR):
        os.makedirs(config.OUT_DIR)

    df = pd.read_csv(f"{config.ROOT_DIR}/train.csv")

    if config.DEBUG:
        kf = GroupKFold(n_splits=11)
        for fold, (_, valid_index) in enumerate(kf.split(df, df['id'], df['point_of_interest'])):
            if fold == 5:  # pick only fold 5 for debug
                df = df.iloc[valid_index].reset_index(drop=True)
                
    for col in ["name", "address", "city", "state", "zip", "country", "url", "phone", "categories"]:
        df[col] = df[col].fillna("")

    df["fulltext"] = (
        df["name"] + " " + df["address"] + " " + df["city"] + " " + df["state"] + " "  + df["country"] + " " + df["categories"]
    ).replace(r'\s+', ' ', regex=True)

    # preprocess of string
    df["fulltext"] = df["fulltext"].str.lower()  # lowercase
    df["fulltext"] = df["fulltext"].str.replace(r'[^\w\s]+', '')  # remove punctuation

    # Standardization of coordinates.
    # https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature
    df["coord_x"] = np.cos(df["latitude"]) * np.cos(df["longitude"])
    df["coord_y"] = np.cos(df["latitude"]) * np.sin(df["longitude"])
    df["coord_z"] = np.sin(df["latitude"])


    gkf = GroupKFold(n_splits=config.n_splits)
    for fold, ( _, val_) in enumerate(gkf.split(X=df, y=df.point_of_interest, groups=df.point_of_interest)):
          df.loc[val_ , "kfold"] = fold

    df_train = df.query("kfold != @config.fold_to_train").copy().reset_index()
    df_valid = df.query("kfold == @config.fold_to_train").copy().reset_index()

    #df_train, df_valid = train_test_split(df, random_state=config.seed, shuffle=True, test_size=0.2)
    #df_train = df  # 訓練にデータ全部使う
    #df_valid = df_valid[df_valid.point_of_interest.isin(df_train.point_of_interest.unique())]

    config.n_classes = df_train.point_of_interest.nunique()
    print("Number of classes", config.n_classes)

    print(df.shape)
    print(df_train.shape)
    print(df_valid.shape)

    print(len(set(df_train.point_of_interest)))
    print(len(set(df_valid.point_of_interest)))


    print(len(set(df_train.point_of_interest) & set(df_valid.point_of_interest)))

    encoder = LabelEncoder()
    df_train['point_of_interest'] = encoder.fit_transform(df_train['point_of_interest'])
    encoder = LabelEncoder()
    df_valid['point_of_interest'] = encoder.fit_transform(df_valid['point_of_interest'])

    utils.set_seed(42)

    model = models.FSMultiModalNet(config.model_name)
    model.to(config.device)

    train_loader, valid_loader = dataset.prepare_loaders(df_train, df_valid)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                       
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_loader) * 2, 
        num_training_steps=len(train_loader) * config.epochs
    )

    model, history = engine.run_training(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        device=config.device,
        num_epochs=config.epochs
    )

if __name__ == '__main__':
    run()
