import torch
import torch.nn as nn
from MTST.Loss.Custom_loss import R2BoostedLoss2
from MTST.Models.transformer_basis import TransformerBlock
from MTST.Models.Core_node import MTST
from MTST.Models.full_model import MTST_full
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from MTST.Processing.data_engineering import final_data_processing
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(input_dim=6, embed_dim=64, heads=16, dropout=0.1, n_layers=2, output_len=1, max_len=5000):
    model = MTST_full(
        input_dim=input_dim,
        embed_dim=embed_dim,
        heads=heads,
        dropout=dropout,
        n_layers=n_layers,
        output_len=output_len,
        max_len=max_len,
    ).to(device)
    return model

def load_data(data_path='Data/powerconsumption.csv'):
    energie_raw = pd.read_csv(data_path, sep=',')
    energie = energie_raw.iloc[:, 1:]
    data = pd.DataFrame(energie)
    return data, energie_raw


def train_model(model, data, high_res=15, mid_res=50, low_res=125, input_len=300, output_len=1, batch_size=32, epochs=30, lr=5e-4):
    x_data_train, y_data_train = final_data_processing(data.loc[:4000], input_len, output_len, stride=1, fillna_method='ffill')
    dataset = TensorDataset(x_data_train, y_data_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb, high_res, mid_res, low_res )
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model


def run_inference(model, test_data, energie_raw,high_res=15, mid_res=50, low_res=125, 
                  input_len=300, output_len=1,START_FORECAST='10/30/2017 23:00', 
                  END_FORCAST='12/30/2017 23:00'):
    
    energie_raw['Datetime'] = pd.to_datetime(energie_raw['Datetime'])
    start_dt = pd.to_datetime(START_FORECAST)
    end_dt = pd.to_datetime(END_FORCAST)
    start_date_index = energie_raw.index[energie_raw['Datetime'] == start_dt].to_list()[0]
    end_date_index = energie_raw.index[energie_raw['Datetime'] == end_dt].to_list()[0]
    test_slice = test_data.loc[start_date_index:end_date_index+1]
    x_data_test, y_data_test = final_data_processing(test_slice, input_len, output_len, stride=output_len, fillna_method='ffill')

    #x_data_test, y_data_test = final_data_processing(test_data.loc[start_date_index :end_date_index+1], input_len, output_len, stride=output_len, fillna_method='ffill')
    with torch.no_grad():
        pred = model(x_data_test, high_res, mid_res, low_res)
        pred = pred.squeeze(0).cpu().numpy()
    return pred, y_data_test


if __name__ == "__main__":
    # Exécution classique sans Streamlit
    model = build_model()
    data, energie_raw = load_data()
    model = train_model(model, data)
    pred, true = run_inference(model, data.loc[4000+1:])
    print("Inference done.")
