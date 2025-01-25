import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import time
import community as community_louvain
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns

def load_data(file_path):
    """Load movie dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, rating_threshold, revenue_threshold):
    """Filter dataset and preprocess."""
    scaler = MinMaxScaler()

    # Processar receita para valores numéricos
    df['gross_earn'] = (
        df['gross_earn']
        .str.replace('[\$,]', '', regex=True)
        .str.replace('M', 'e6', regex=False)
        .str.replace('K', 'e3', regex=False)
        .astype(float)
    )

    # Aplicar filtros baseados em nota e receita
    df = df[(df['rating'] >= rating_threshold) & (df['gross_earn'] >= revenue_threshold)]

    # Tratar valores faltantes
    df['runtime'] = df['runtime'].replace({'N/A': None, 'Unknown': None})
    df['runtime'] = df['runtime'].str.extract(r'(\d+)').astype(float)
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    df['gross_earn'] = df['gross_earn'].fillna(df['gross_earn'].median())
    df['genre'] = df['genre'].fillna("Unknown")

    # Normalizar colunas numéricas
    df[['rating', 'runtime', 'gross_earn']] = scaler.fit_transform(df[['rating', 'runtime', 'gross_earn']])

    # Codificar gêneros
    genre_encoded = pd.get_dummies(df['genre'], prefix='genre')
    df = pd.concat([df, genre_encoded], axis=1)

    return df

def create_graph(df, similarity_threshold=0.3):
    """Create a graph where movies are nodes and edges represent similarity."""
    G = nx.Graph()

    # Add nodes
    for index, row in df.iterrows():
        G.add_node(row['title'], rating=row['rating'])

    # Calculate similarity
    feature_cols = ['rating', 'runtime', 'gross_earn'] + [col for col in df.columns if col.startswith('genre_')]
    feature_matrix = df[feature_cols].values

    similarity_matrix = cosine_similarity(feature_matrix)

    # Add edges based on similarity threshold
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > similarity_threshold:
                G.add_edge(df.iloc[i]['title'], df.iloc[j]['title'], weight=similarity_matrix[i, j])

    return G

def analyze_graph(G):
    """Perform basic graph analysis."""
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Calculate centrality
    centrality = nx.degree_centrality(G)
    top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 central nodes:", top_central_nodes)

def analyze_centrality(G, metric='degree'):
    """Analyze centrality of the graph based on the specified metric."""
    if metric == 'degree':
        centrality = nx.degree_centrality(G)
    elif metric == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif metric == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    else:
        raise ValueError("Métrica não reconhecida.")
    
    # Top 3 nós mais centrais
    top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    return top_central_nodes

def visualize_graph(G, title, partition, k=0.4):
    """Visualize the graph using NetworkX and Matplotlib."""
    pos = nx.spring_layout(G, k=k, iterations=20)
    plt.figure(figsize=(12, 8))

    # Colorir nós de acordo com a comunidade
    cmap = plt.get_cmap('viridis')
    communities = set(partition.values())
    colors = [cmap(partition[node] / len(communities)) for node in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors, alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray')
    labels = {node: node for node in G.nodes}  # Exibe todos os rótulos
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    plt.title(title)
    plt.axis('off')

    # Adicionar legenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / len(communities)), markersize=10) for i in range(len(communities))]
    labels = [f'Comunidade {i + 1}' for i in range(len(communities))]
    plt.legend(handles, labels, loc='best', title='Comunidades')

    plt.show()

def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
    return result


def plot_info_table(num_nodes, num_edges, top_degree_nodes, top_closeness_nodes, top_betweenness_nodes):
    """Plot a table with important graph information."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Dados para a tabela
    data = [
        ["Número de Nós", num_nodes],
        ["Número de Arestas", num_edges],
        ["Top 3 Filmes por Grau de Centralidade", ", ".join([node for node, _ in top_degree_nodes])],
        ["Top 3 Filmes por Centralidade de Proximidade", ", ".join([node for node, _ in top_closeness_nodes])],
        ["Top 3 Filmes por Centralidade de Intermediação", ", ".join([node for node, _ in top_betweenness_nodes])]
    ]

    # Criar tabela
    table = ax.table(cellText=data, colLabels=["Métrica", "Valor"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Informações Importantes do Grafo")
    plt.show()

def cluster_graph(G):
    """Cluster the graph using the greedy modularity method."""
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    return partition

def analyze_communities(df, partition):
    """Analyze and print information about each community."""
    communities = {}
    for node, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    
    for community, nodes in communities.items():
        print(f"Comunidade {community + 1}:")
        print(f"Número de filmes: {len(nodes)}")
        print(f"Filmes: {', '.join(nodes[:10])}...")  # Exibir os primeiros 10 filmes de cada comunidade
        print()

def analyze_shared_features(df, G):
    """Analyze shared features between connected movies."""
    shared_features = {}
    for edge in G.edges(data=True):
        movie1, movie2, data = edge
        shared_features[(movie1, movie2)] = {
            'similarity': data['weight'],
            'shared_genres': list(set(df[df['title'] == movie1].iloc[0].filter(like='genre_').index) & set(df[df['title'] == movie2].iloc[0].filter(like='genre_').index)),
            'runtime_diff': abs(df[df['title'] == movie1]['runtime'].values[0] - df[df['title'] == movie2]['runtime'].values[0]),
            'rating_diff': abs(df[df['title'] == movie1]['rating'].values[0] - df[df['title'] == movie2]['rating'].values[0]),
            'gross_earn_diff': abs(df[df['title'] == movie1]['gross_earn'].values[0] - df[df['title'] == movie2]['gross_earn'].values[0])
        }
    return shared_features

def feature_importance_analysis(df, target):
    """Train a model and calculate feature importance for a given target."""
    # Selecionar features e target
    feature_cols = ['runtime'] + [col for col in df.columns if col.startswith('genre_')]
    X = df[feature_cols]
    y = df[target]

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calcular a importância das features
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    return feature_importance

def plot_feature_importance(feature_importance, title):
    """Plot a bar chart of the top 5 most important features."""
    top_features = feature_importance.head(5)  # Selecionar as top 5 características mais importantes

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')

    plt.xlabel('Importância', fontsize=14)
    plt.ylabel('Características', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adicionar rótulos de valor nas barras
    for index, value in enumerate(top_features['importance']):
        plt.text(value, index, f'{value:.2f}', va='center', ha='right', fontsize=12, color='black')

    plt.gca().invert_yaxis()  # Inverter o eixo y para mostrar a característica mais importante no topo
    plt.show()

def main():
    file_path = 'movies2.csv'

    print("Loading dataset...")
    df = measure_execution_time(load_data, file_path)

    # Aplicar filtros por nota e receita
    print("Preprocessing data (filtered by rating and revenue)...")
    df_filtered = measure_execution_time(preprocess_data, df.copy(), 7.3, 1e8)

    print("Creating graph (filtered by rating and revenue)...")
    G_filtered = measure_execution_time(create_graph, df_filtered, similarity_threshold=0.5)

    print("Analyzing graph (filtered by rating and revenue)...")
    measure_execution_time(analyze_graph, G_filtered)

    print("Clustering graph...")
    partition = measure_execution_time(cluster_graph, G_filtered)

    print("Analyzing communities...")
    analyze_communities(df_filtered, partition)

    print("Visualizing graph (filtered by rating and revenue)...")
    measure_execution_time(visualize_graph, G_filtered, "Graph Filtered by Rating and Revenue", partition, k=0.5)

    # Analisar centralidade
    print("Analyzing centrality (degree)...")
    top_degree_nodes = measure_execution_time(analyze_centrality, G_filtered, metric='degree')
    print("Top 3 filmes por grau de centralidade:", top_degree_nodes)

    print("Analyzing centrality (closeness)...")
    top_closeness_nodes = measure_execution_time(analyze_centrality, G_filtered, metric='closeness')
    print("Top 3 filmes por centralidade de proximidade:", top_closeness_nodes)

    print("Analyzing centrality (betweenness)...")
    top_betweenness_nodes = measure_execution_time(analyze_centrality, G_filtered, metric='betweenness')
    print("Top 3 filmes por centralidade de intermediação:", top_betweenness_nodes)

    # Coletar informações importantes
    num_nodes = G_filtered.number_of_nodes()
    num_edges = G_filtered.number_of_edges()

    # Plotar tabela com informações importantes
    plot_info_table(num_nodes, num_edges, top_degree_nodes, top_closeness_nodes, top_betweenness_nodes)

    # Analisar importância das características para receita
    print("Analyzing feature importance for revenue...")
    revenue_importance = measure_execution_time(feature_importance_analysis, df_filtered, target='gross_earn')
    plot_feature_importance(revenue_importance, 'Top 5 Características para Receita do Filme')

    # Analisar importância das características para nota
    print("Analyzing feature importance for rating...")
    rating_importance = measure_execution_time(feature_importance_analysis, df_filtered, target='rating')
    plot_feature_importance(rating_importance, 'Top 5 Características para Nota do Filme')

if __name__ == "__main__":
    main()
