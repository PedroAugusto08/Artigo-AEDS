import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import pandas as pd
import time
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def create_graph(df, similarity_threshold=0.7, max_connections_per_node=10):
    """Create a graph where movies are nodes and edges represent similarity."""
    G = nx.Graph()

    # Add nodes
    for index, row in df.iterrows():
        G.add_node(row['Title'], rating=row['Rating'])

    # Calcular a matriz de similaridade
    feature_cols = ['Rating', 'Runtime (Minutes)', 'Revenue (Millions)', 'Budget (Million)'] + [col for col in df.columns if col.startswith('genre_')]
    feature_matrix = df[feature_cols].values
    similarity_matrix = cosine_similarity(feature_matrix)

    # Criar arestas baseadas na similaridade
    for i in range(len(df)):
        similarities = []  # Lista para armazenar pares (similaridade, índice)
        for j in range(len(df)):
            if i != j:  # Evitar auto-conexões
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    similarities.append((similarity, j))

        # Ordenar por similaridade (descendente) e limitar as conexões
        similarities = sorted(similarities, reverse=True)[:max_connections_per_node]

        # Criar arestas para os nós mais semelhantes
        for similarity, j in similarities:
            G.add_edge(df.iloc[i]['Title'], df.iloc[j]['Title'], weight=similarity, color='blue')

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
    if (metric == 'degree'):
        centrality = nx.degree_centrality(G)
    elif (metric == 'closeness'):
        centrality = nx.closeness_centrality(G)
    elif (metric == 'betweenness'):
        centrality = nx.betweenness_centrality(G)
    else:
        raise ValueError("Métrica não reconhecida.")
    
    # Top 3 nós mais centrais
    top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    return top_central_nodes

def calculate_node_size(G, min_size=200, max_size=1000):
    """Calculate node sizes based on their degree."""
    degrees = dict(G.degree())
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
    
    # Normalizar tamanhos dos nós
    node_sizes = {}
    for node, degree in degrees.items():
        if max_degree == min_degree:
            node_sizes[node] = min_size
        else:
            node_sizes[node] = min_size + (degree - min_degree) / (max_degree - min_degree) * (max_size - min_size)
    
    return node_sizes

def visualize_graph(G, title, partition, k=0.5):
    """Visualize the graph using NetworkX and Matplotlib.""" 
    pos = nx.spring_layout(G, k=k, iterations=100, seed=42)
    plt.figure(figsize=(14, 10))

    # Colorir nós de acordo com a comunidade
    cmap = plt.get_cmap('viridis')
    communities = set(partition.values())
    colors = [cmap(partition[node] / len(communities)) for node in G.nodes]

    # Calcular tamanhos dos nós com base no grau
    node_sizes = calculate_node_size(G)

    # Desenhar nós com menos transparência
    nx.draw_networkx_nodes(G, pos, node_size=[node_sizes[node] for node in G.nodes], node_color=colors, alpha=0.9)

    # Desenhar arestas com uma única cor
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=2, edge_color='gray')

    # Adicionar rótulos
    labels = {node: node for node in G.nodes}
    texts = []
    
    for node, (x, y) in pos.items():
        text = plt.text(x, y, labels[node], fontsize=7, color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.1))
        texts.append(text)

    # Ajustar automaticamente os rótulos
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.1))
    
    # Adicionar título
    plt.title(title, fontsize=16)
    plt.axis('off')

    # Adicionar legenda para os nós
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

def calculate_average_degree(G):
    """Calculate the average degree of nodes in the graph."""
    degrees = dict(G.degree())
    average_degree = sum(degrees.values()) / len(degrees)
    return average_degree

def plot_info_table(num_nodes, num_edges, top_degree_nodes, top_closeness_nodes, top_betweenness_nodes, average_degree):
    """Plot a table with important graph information.""" 
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Dados para a tabela
    data = [
        ["Número de Nós", num_nodes],
        ["Número de Arestas", num_edges],
        ["Grau Médio dos Nós", f"{average_degree:.2f}"],
        ["Top 3 Filmes por Grau de Centralidade", ", ".join([node for node, _ in top_degree_nodes])],
        ["Top 3 Filmes por Centralidade de Proximidade", ", ".join([node for node, _ in top_closeness_nodes])],
        ["Top 3 Filmes por Centralidade de Intermediação", ", ".join([node for node, _ in top_betweenness_nodes])],
    ]

    # Criar tabela
    table = ax.table(cellText=data, colLabels=["Métrica", "Valor"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Informações Importantes do Grafo")
    plt.show()

def cluster_graph_louvain(G):
    """Cluster the graph using the Louvain method."""
    partition = community_louvain.best_partition(G)
    return partition

def analyze_communities(df, partition):
    """Analyze and print information about each community.""" 
    communities = {}
    for node, community in partition.items():
        if (community not in communities):
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
    for movie1, movie2, data in G.edges(data=True):
        shared_features[(movie1, movie2)] = {
            'similarity': data['weight'],
            'shared_genres': list(set(df[df['Title'] == movie1].iloc[0].filter(like='genre_').index) & set(df[df['Title'] == movie2].iloc[0].filter(like='genre_').index)),
            'runtime_diff': abs(df[df['Title'] == movie1]['Runtime (Minutes)'].values[0] - df[df['Title'] == movie2]['Runtime (Minutes)'].values[0]),
            'rating_diff': abs(df[df['Title'] == movie1]['Rating'].values[0] - df[df['Title'] == movie2]['Rating'].values[0]),
            'revenue_diff': abs(df[df['Title'] == movie1]['Revenue (Millions)'].values[0] - df[df['Title'] == movie2]['Revenue (Millions)'].values[0])
        }
    return shared_features

def feature_importance_analysis(df, target):
    """Train a model and calculate feature importance for a given target.""" 
    # Selecionar features e target
    feature_cols = ['Runtime (Minutes)', 'Budget (Million)', 'Director'] + [col for col in df.columns if col.startswith('genre_')]
    
    # Codificar a coluna "Director"
    df['Director'] = df['Director'].astype('category').cat.codes
    
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

def feature_importance_analysis(df, target):
    """Train a model and calculate feature importance for a given target.""" 
    # Selecionar features e target
    feature_cols = ['Runtime (Minutes)', 'Budget (Million)', 'Director'] + [col for col in df.columns if col.startswith('genre_')]
    
    # Codificar a coluna "Director"
    df['Director'] = df['Director'].astype('category').cat.codes
    
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
    """Plot a bar chart of the top 5 importance of features based on their importance score.""" 
    feature_importance_df = pd.DataFrame(feature_importance)
    feature_importance_df['importance'] = feature_importance_df['importance'].apply(lambda x: x if isinstance(x, (int, float)) else float('nan'))
    feature_importance_df = feature_importance_df.dropna(subset=['importance'])
    feature_importance_df['importance'] = feature_importance_df['importance'].astype(float)
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(5)  # Selecionar as top 5 características

    # Definir um mapeamento de cores consistente
    color_mapping = {
        'genre': 'blue',
        'Director': 'green',
        'Runtime (Minutes)': 'purple',
        'Budget (Million)': 'orange'
    }

    # Aplicar as cores correspondentes
    feature_importance_df['color'] = feature_importance_df['feature'].apply(lambda x: color_mapping.get(x.split('_')[0], 'gray'))

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette=feature_importance_df['color'].tolist())

    plt.xlabel('Importância', fontsize=14)
    plt.ylabel('Características', fontsize=14)
    plt.title('Top 5 Características Mais Importantes para a Receita', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adicionar rótulos de valor nas barras
    for index, value in enumerate(feature_importance_df['importance']):
        plt.text(value, index, f'{value:.2f}', va='center', ha='right', fontsize=12, color='black')

    plt.gca().invert_yaxis()  # Inverter o eixo y para mostrar a característica mais importante no topo
    plt.show()  # Certifique-se de que o gráfico seja exibido

def remove_nodes_without_connections(G):
    """Remove nodes that have no connections from the graph.""" 
    nodes_to_remove = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)
    return G