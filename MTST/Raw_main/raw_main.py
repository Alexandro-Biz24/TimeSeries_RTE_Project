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

START_FORECAST='10/30/2017 23:00'
END_FORCAST='12/30/2017 23:00'

# Model Hyperparameters
input_dim = 6
embed_dim = 64
heads = 16
dropout = 0.1
n_layers = 2
output_len = 1
input_len = 300
max_len = 5000

# Resolution Factors
high_res = 15
mid_res = 50
low_res = 125

batch_size = 32
epochs = 30
lr = 5e-4
print('___________parmaètres bien initialisés___________')
# fixed parameter
model = MTST_full(
    input_dim=input_dim,
    embed_dim=embed_dim,
    heads=heads,
    dropout=dropout,   
    n_layers=n_layers,
    output_len=output_len,
    max_len=max_len,
    ).to(device)

criterion = nn.MSELoss() # R2BoostedLoss2(alpha=1.0, beta=0.8, gamma=0.5) # # R2BoostedLoss DynamicLoss CombinedLoss AmplifiedMSELoss
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

print('___________model parametré___________')

data_path= 'MTST/Data/powerconsumption.csv'
print('___________Début du data processing___________')
#placer la target en premiere colonne (index=0)
energie_raw=pd.read_csv(data_path,sep=',')
energie=energie_raw.iloc[:,1:]

# Indexer sans les timestamp 
zone='PowerConsumption_Zone1'
data=pd.DataFrame(energie[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', zone]])

train=data.loc[:4000]
test=data.loc[4000+1:]  # start : 1/28/2017 18:50
print('___________data fin du data processing___________')

x_data_train, y_data_train = final_data_processing(train, input_len, output_len, stride=1, fillna_method='ffill')
dataset = TensorDataset(x_data_train, y_data_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print('___________vectorisation réussi___________')
loss_tracker = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        preds = model(xb, high_res, mid_res, low_res)  # [B, output_len, 1]
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_tracker.append(avg_loss)
print('___________fin entrainement', f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}___________")

train_R2_SCORE=r2_score(yb[:,:,5].detach().cpu().numpy().squeeze(), preds[:,:,5].detach().cpu().numpy().squeeze()) 

start_date_index = energie_raw.index[energie_raw['Datetime'] == START_FORECAST].to_list()[0]

end_date_index = energie_raw.index[energie_raw['Datetime'] == END_FORCAST].to_list()[0]

print('_________début inférence___________')
x_data_test, y_data_test= final_data_processing(test[start_date_index:end_date_index+1], input_len, output_len, stride=output_len, fillna_method='ffill')
dataset_tst = TensorDataset(x_data_test, y_data_test)
test_loader = DataLoader(dataset_tst, batch_size=batch_size, shuffle=True)

model.eval()
with torch.no_grad():
    pred = model(x_data_test, high_res, mid_res, low_res)
    pred = pred.squeeze(0).cpu().numpy()

pred_r2_score=r2_score(y_data_test[:,:,5].detach().cpu().numpy(), pred[:,:,5])
print(f'___________fin inférence score r2 {pred_r2_score}___________')
plt.figure(figsize=(14, 4))

true=y_data_test[:,:,5].detach().cpu().numpy().squeeze()
plt.plot(true, label='True', alpha=0.7)
plt.plot(pred[:,:,5], label='Pred', alpha=0.7)
plt.title(f'Feature ')
plt.show()













# main.py


col_graph1, col_stats1 = st.columns([3, 1])
with col_graph1:
    st.markdown("### Prévisions chronologiques")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pred_np, mode="lines", name="Prévision"))
    st.plotly_chart(fig, use_container_width=True)

with col_stats1:
    st.markdown("### Détails du point sélectionné")
    st.info("Fonction interactive à venir (hover sur le graphe)")





    # --- Disposition Streamlit (graph + détails) ---
    col_graph1, col_stats1 = st.columns([3, 1])

    with col_graph1:
        st.markdown("### Prévisions chronologiques")

        # 1️⃣ Rendu visuel du graphe
        st.plotly_chart(fig, use_container_width=True)

        # 2️⃣ Capture d’événements hover
        selected_points = plotly_events(
            fig,
            click_event=False,
            hover_event=True,
            override_height=450,
            key="pred_hover"
        )

    with col_stats1:
        st.markdown("### 🔍 Détails du point sélectionné")

        # Initialise des placeholders pour chaque case
        placeholders = {
            name: st.empty() for name in feature_names
        }

        if selected_points:
            hovered_index = int(selected_points[0]["pointIndex"])
            vals = {name: other_feats[name][hovered_index] for name in feature_names}

            # Met à jour les cases avec les nouvelles valeurs
            for name, val in vals.items():
                placeholders[name].markdown(
                    f"""
                    <div style='padding: 10px; margin-bottom: 8px; border-radius: 10px;
                                background-color:#f7f9fb; border:1px solid #dce3e8;'>
                        <b>{name}</b><br>
                        <span style='font-size:18px;'>{val:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Survolez un point du graphe pour afficher les données explicatives : "
                    "Température, Humidité, Vent, Flux diffus, etc.")
            


# Mémorise l'index sélectionné (persistant)
    if events:
        idx = int(events[0].get("pointIndex", events[0].get("pointNumber", 0)))
        st.session_state.last_idx = idx

    last_idx = st.session_state.get("last_idx", None)

    with col_stats1:
        st.markdown("### 🔍 Détails du point sélectionné")
        if last_idx is None:
            st.info("Cliquez un point sur la courbe pour afficher les données explicatives : "
                    "Température, Humidité, Vent, Flux diffus, etc.")
        else:
            for name in feature_names:
                val = float(other_feats[name][last_idx])
                st.markdown(
                    f"""
                    <div style='padding: 10px; margin-bottom: 8px; border-radius: 10px;
                                background-color:#f7f9fb; border:1px solid #dce3e8;'>
                        <b>{name}</b><br>
                        <span style='font-size:18px;'>{val:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
