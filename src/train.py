import os
import warnings

import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import get_linear_schedule_with_warmup

import dataset
import engine
import models
import utils
from configs import CFG

warnings.filterwarnings("ignore")
# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run():
    if not os.path.exists(CFG.OUT_DIR):
        os.makedirs(CFG.OUT_DIR)

    df = pd.read_csv(f"{CFG.ROOT_DIR}/{CFG.INPUT}")

    if CFG.DEBUG:
        kf = GroupKFold(n_splits=11)
        for fold, (_, valid_index) in enumerate(kf.split(df, df['id'], df['point_of_interest'])):
            if fold == 5:  # pick only fold 5 for debug
                df = df.iloc[valid_index].reset_index(drop=True)
        CFG.n_classes = df["point_of_interest"].nunique()
                
    for col in ["name", "address", "city", "state", "zip", "country", "url", "phone", "categories"]:
        df[col] = df[col].fillna("")

    df["fulltext"] = (
        df["name"] + " " + df["address"] + " " + df["city"] + " " + df["state"] + " "  + df["country"] + " " + df["categories"]
    ).replace(r'\s+', ' ', regex=True)

    # preprocess of string
    # df["fulltext"] = df["fulltext"].str.lower()  # lowercase
    # df["fulltext"] = df["fulltext"].str.replace(r'[^\w\s]+', '')  # remove punctuation

    # Standardization of coordinates.
    # https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature
    df["coord_x"] = np.cos(df["latitude"]) * np.cos(df["longitude"])
    df["coord_y"] = np.cos(df["latitude"]) * np.sin(df["longitude"])
    df["coord_z"] = np.sin(df["latitude"])

    df_train, df_valid = train_test_split(df, random_state=CFG.seed, shuffle=True, test_size=0.2)
    df_train = df  # 訓練にデータ全部使う
    df_valid = df_valid[df_valid.point_of_interest.isin(df_train.point_of_interest.unique())]

    CFG.n_classes = df_train.point_of_interest.nunique()
    print("Number of classes", CFG.n_classes)

    encoder = LabelEncoder()
    df_train['point_of_interest'] = encoder.fit_transform(df_train['point_of_interest'])
    df_valid['point_of_interest'] = encoder.transform(df_valid['point_of_interest'])

    utils.set_seed(CFG.seed)

    model = models.FSMultiModalNet(CFG.model_name)
    model.to(CFG.device)

    train_loader, valid_loader = dataset.prepare_loaders(df_train, df_valid)

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
                       
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_loader) * CFG.warmup_epochs, 
        num_training_steps=len(train_loader) * CFG.epochs
    )

    model, history = engine.run_training(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        device=CFG.device,
        num_epochs=CFG.epochs
    )

if __name__ == '__main__':
    run()
