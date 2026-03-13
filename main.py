import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import IsolationForest
from networkx.algorithms import community

# --------------------------------------------------
# STEP 1: Load Dataset
# --------------------------------------------------

data = pd.read_csv("data/train_transaction.csv", nrows=5000)

print("Dataset loaded successfully!\n")

data = data[['card1','TransactionAmt','isFraud']]
data = data.dropna()

data['card1'] = data['card1'].astype(int)

print("Cleaned dataset shape:", data.shape)

# --------------------------------------------------
# STEP 2: Build Transaction Graph
# --------------------------------------------------

G = nx.Graph()

for i,row in data.iterrows():

    card = str(row['card1'])
    amount = row['TransactionAmt']

    G.add_node(card)

    txn_node = "txn_" + str(i)

    G.add_node(txn_node)

    G.add_edge(card,txn_node,weight=amount)

print("\nGraph Created")

print("Number of Nodes:", G.number_of_nodes())
print("Number of Edges:", G.number_of_edges())

# --------------------------------------------------
# STEP 3: Extract Graph Features
# --------------------------------------------------

degree = dict(G.degree())
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G)

features = []
nodes = []

for node in G.nodes():

    if "txn_" not in node:

        nodes.append(node)

        features.append([
            degree[node],
            pagerank[node],
            betweenness[node],
            clustering[node]
        ])

feature_df = pd.DataFrame(
    features,
    columns=["degree","pagerank","betweenness","clustering"]
)

print("\nFeature Data:")
print(feature_df.head())

# --------------------------------------------------
# STEP 4: Fraud Detection
# --------------------------------------------------

model = IsolationForest(contamination=0.05)

model.fit(feature_df)

predictions = model.predict(feature_df)

feature_df['anomaly'] = predictions

print("\nFraud Prediction:")
print(feature_df.head())

# --------------------------------------------------
# STEP 5: Identify Suspicious Accounts
# --------------------------------------------------

fraud_nodes = []

for node,pred in zip(nodes,predictions):

    if pred == -1:
        fraud_nodes.append(node)

print("\nSuspicious Accounts:")
print(fraud_nodes)

# --------------------------------------------------
# STEP 6: Save Suspicious Accounts
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

fraud_df = pd.DataFrame(
    fraud_nodes,
    columns=["Suspicious_Account"]
)

fraud_df.to_csv("outputs/suspicious_accounts.csv", index=False)

print("\nSuspicious accounts saved to outputs/suspicious_accounts.csv")

# --------------------------------------------------
# STEP 7: Display Fraud Accounts Clearly
# --------------------------------------------------

plt.figure(figsize=(12,10))

fraud_subgraph_nodes = set()

for fraud in fraud_nodes:

    fraud_subgraph_nodes.add(fraud)

    neighbors = list(G.neighbors(fraud))

    for n in neighbors:
        fraud_subgraph_nodes.add(n)

H = G.subgraph(fraud_subgraph_nodes)

pos = nx.spring_layout(H, k=0.5, seed=42)

normal_nodes = []
fraud_nodes_graph = []

for node in H.nodes():

    if node in fraud_nodes:
        fraud_nodes_graph.append(node)
    else:
        normal_nodes.append(node)

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=normal_nodes,
    node_color="skyblue",
    node_size=300
)

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=fraud_nodes_graph,
    node_color="red",
    node_size=900
)

nx.draw_networkx_edges(
    H,
    pos,
    edge_color="gray",
    alpha=0.6
)

nx.draw_networkx_labels(
    H,
    pos,
    labels={node:node for node in fraud_nodes_graph},
    font_size=9
)

plt.title("Fraud Accounts and Their Transaction Connections")

plt.axis("off")

plt.show()

# --------------------------------------------------
# STEP 8: Detect Fraud Rings (Communities)
# --------------------------------------------------

UG = G.to_undirected()

communities = list(
    community.greedy_modularity_communities(UG)
)

print("\nTotal Communities Detected:", len(communities))

fraud_communities = []

for i,comm in enumerate(communities):

    fraud_in_comm = [node for node in comm if node in fraud_nodes]

    if len(fraud_in_comm) > 0:

        fraud_communities.append(comm)

        print(f"\nFraud Community {i+1}")
        print("Fraud accounts:", fraud_in_comm)

# --------------------------------------------------
# STEP 9: Visualize Fraud Rings
# --------------------------------------------------

plt.figure(figsize=(12,10))

fraud_ring_nodes = set()

for comm in fraud_communities:
    fraud_ring_nodes.update(comm)

H = G.subgraph(fraud_ring_nodes)

pos = nx.spring_layout(H, k=0.4)

normal_nodes = []
fraud_nodes_graph = []

for node in H.nodes():

    if node in fraud_nodes:
        fraud_nodes_graph.append(node)
    else:
        normal_nodes.append(node)

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=normal_nodes,
    node_color="skyblue",
    node_size=300
)

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=fraud_nodes_graph,
    node_color="red",
    node_size=800
)

nx.draw_networkx_edges(
    H,
    pos,
    alpha=0.5
)

nx.draw_networkx_labels(
    H,
    pos,
    labels={node:node for node in fraud_nodes_graph},
    font_size=9
)

plt.title("Detected Fraud Rings")

plt.axis("off")

plt.show()