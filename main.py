import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from adjustText import adjust_text
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import girvan_newman
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load movie dataset from CSV.""" 
    df = pd.read_csv(file_path, delimiter=';')
    return df

def preprocess_data(df, rating_threshold, revenue_threshold):
    """Filter dataset and preprocess.""" 
    scaler = MinMaxScaler()

    # Substituir vírgulas por pontos e converter colunas numéricas para float
    df['Revenue (Millions)'] = df['Revenue (Millions)'].str.replace(',', '.').astype(float)
    df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)
    if 'Budget (Million)' in df.columns:
        df['Budget (Million)'] = df['Budget (Million)'].astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)').astype(float)
    else:
        df['Budget (Million)'] = np.nan  # Adiciona a coluna como NaN caso não exista

    # Converter colunas numéricas para float
    df['Runtime (Minutes)'] = df['Runtime (Minutes)'].replace({'N/A': None, 'Unknown': None}).astype(float)

    # Aplicar filtros baseados em nota e receita
    df = df[(df['Rating'] >= rating_threshold) & (df['Revenue (Millions)'] >= revenue_threshold)]

    # Tratar valores faltantes
    df['Runtime (Minutes)'] = df['Runtime (Minutes)'].fillna(df['Runtime (Minutes)'].median())
    df['Revenue (Millions)'] = df['Revenue (Millions)'].fillna(df['Revenue (Millions)'].median())
    df['Budget (Million)'] = df['Budget (Million)'].fillna(df['Budget (Million)'].median())

    # Normalizar colunas numéricas
    df[['Runtime (Minutes)', 'Revenue (Millions)', 'Budget (Million)']] = scaler.fit_transform(df[['Runtime (Minutes)', 'Revenue (Millions)', 'Budget (Million)']])

    # Codificar gêneros (apenas o Primary Genre)
    genre_encoded = pd.get_dummies(df['Primary Genre'], prefix='genre')
    df = pd.concat([df, genre_encoded], axis=1)

    return df, scaler

def denormalize_data(df, scaler, feature_cols):
    """Denormalize the data using the original scaler.""" 
    df_denormalized = df.copy()
    df_denormalized[feature_cols] = scaler.inverse_transform(df[feature_cols])
    return df_denormalized

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
            
        # Print depurador para exibir o grau de cada nó
        print(f"Nó: {node}, Grau: {degree}, Tamanho: {node_sizes[node]}")
    
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
        text = plt.text(x, y, labels[node], fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.3))
        texts.append(text)

    # Ajustar automaticamente os rótulos
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
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
        ["Top 3 Filmes por Centralidade de Intermediação", ", ".join([node for node, _ in top_betweenness_nodes])],
    ]

    # Criar tabela
    table = ax.table(cellText=data, colLabels=["Métrica", "Valor"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Informações Importantes do Grafo")
    plt.show()

def cluster_graph_girvan_newman(G, k=2):
    """Cluster the graph using the Girvan-Newman method.""" 
    comp = girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    communities = list(limited)[-1]
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
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

def calculate_statistics(df):
    """Calculate descriptive statistics for the most successful movies.""" 
    statistics = {
        'runtime_mean': df['Runtime (Minutes)'].mean(),
        'runtime_median': df['Runtime (Minutes)'].median(),
        'runtime_std': df['Runtime (Minutes)'].std(),
        'revenue_mean': df['Revenue (Millions)'].mean(),  # Já está em milhões
        'revenue_median': df['Revenue (Millions)'].median(),  # Já está em milhões
        'revenue_std': df['Revenue (Millions)'].std(),  # Já está em milhões
        'budget_mean': df['Budget (Million)'].mean(),  # Já está em milhões
        'budget_median': df['Budget (Million)'].median(),  # Já está em milhões
        'budget_std': df['Budget (Million)'].std(),  # Já está em milhões
        'rating_mean': df['Rating'].mean(),
        'rating_median': df['Rating'].median(),
        'rating_std': df['Rating'].std()
    }
    return statistics


def display_statistics(statistics):
    """Display descriptive statistics for the most successful movies in a table.""" 
    data = [
        ["Tempo Médio de Duração", f"{statistics['runtime_mean']:.2f} minutos"],
        ["Mediana da Duração", f"{statistics['runtime_median']:.2f} minutos"],
        ["Desvio Padrão da Duração", f"{statistics['runtime_std']:.2f} minutos"],
        ["Receita Média", f"${statistics['revenue_mean']:.2f}M"],
        ["Mediana da Receita", f"${statistics['revenue_median']:.2f}M"],
        ["Desvio Padrão da Receita", f"${statistics['revenue_std']:.2f}M"],
        ["Orçamento Médio", f"${statistics['budget_mean']:.2f}M"],
        ["Mediana do Orçamento", f"${statistics['budget_median']:.2f}M"],
        ["Desvio Padrão do Orçamento", f"${statistics['budget_std']:.2f}M"],
        ["Nota Média", f"{statistics['rating_mean']:.2f}"],
        ["Mediana da Nota", f"{statistics['rating_median']:.2f}"],
        ["Desvio Padrão da Nota", f"{statistics['rating_std']:.2f}"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=data, colLabels=["Métrica", "Valor"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Estatísticas Descritivas para os Filmes de Maior Sucesso")
    plt.show()  # Certifique-se de que o gráfico seja exibido

def find_most_successful_movie(df):
    """Find the most successful movie based on revenue.""" 
    most_successful_movie = df.loc[df['Revenue (Millions)'].idxmax()]
    return most_successful_movie

def find_connected_movies(G, df, movie_title):
    """Find movies connected to the given movie title in the graph and ensure they share at least one characteristic.""" 
    connected_movies = list(G.neighbors(movie_title))
    valid_connected_movies = []

    movie_data = df[df['Title'] == movie_title].iloc[0]

    for connected_movie in connected_movies:
        connected_movie_data = df[df['Title'] == connected_movie].iloc[0]
        shared_genres = movie_data['Primary Genre'] == connected_movie_data['Primary Genre']
        shared_director = movie_data['Director'] == connected_movie_data['Director']
        similar_runtime = abs(movie_data['Runtime (Minutes)'] - connected_movie_data['Runtime (Minutes)']) < 0.1
        similar_budget = abs(movie_data['Budget (Million)'] - connected_movie_data['Budget (Million)']) < 0.1

        if shared_genres or shared_director or similar_runtime or similar_budget:
            valid_connected_movies.append(connected_movie)

    return valid_connected_movies

def analyze_connection_factors(df, movie_title, connected_movies):
    """Analyze the factors that connect the given movie to its connected movies.""" 
    factors = []
    factor_counts = {
        'shared_genres': 0,
        'shared_director': 0,
        'similar_runtime': 0,
        'similar_budget': 0
    }
    movie_data = df[df['Title'] == movie_title].iloc[0]
    
    for connected_movie in connected_movies:
        connected_movie_data = df[df['Title'] == connected_movie].iloc[0]
        shared_genres = movie_data['Primary Genre'] == connected_movie_data['Primary Genre']
        shared_director = movie_data['Director'] == connected_movie_data['Director']
        similar_runtime = abs(movie_data['Runtime (Minutes)'] - connected_movie_data['Runtime (Minutes)']) < 0.1
        similar_budget = abs(movie_data['Budget (Million)'] - connected_movie_data['Budget (Million)']) < 0.1

        factors.append({
            'connected_movie': connected_movie,
            'shared_genres': shared_genres,
            'shared_director': shared_director,
            'similar_runtime': similar_runtime,
            'similar_budget': similar_budget
        })

        # Atualizar contagem de fatores
        if shared_genres:
            factor_counts['shared_genres'] += 1
        if shared_director:
            factor_counts['shared_director'] += 1
        if similar_runtime:
            factor_counts['similar_runtime'] += 1
        if similar_budget:
            factor_counts['similar_budget'] += 1
    
    return factors, factor_counts

def plot_successful_movie_connections(G, df, movie_title, connected_movies, connection_factors, k=1.0):
    """Plot the most successful movie and its connected movies, and analyze the connection factors.""" 
    pos = nx.spring_layout(G, k=k, iterations=20)
    plt.figure(figsize=(12, 8))

    # Colorir nós
    node_colors = []
    for node in G.nodes:
        if node == movie_title:
            node_colors.append('red')  # Filme de maior sucesso em vermelho
        elif node in connected_movies:
            node_colors.append('blue')  # Filmes conectados em azul
        else:
            node_colors.append('gray')  # Outros filmes em cinza

    # Definir um tamanho fixo para todos os nós
    node_size = 300

    # Desenhar nós
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.6)

    # Desenhar arestas com cores diferentes
    edges = G.edges(data=True)
    edge_colors = [edge[2]['color'] for edge in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color=edge_colors)

    # Desenhar rótulos
    labels = {node: node for node in G.nodes if node == movie_title or node in connected_movies}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    plt.title(f"Filme de Maior Sucesso e Filmes Conectados: {movie_title}")
    plt.axis('off')

    # Adicionar legenda para os nós
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Filme de Maior Sucesso'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Filmes Conectados'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Outros Filmes')
    ]

    # Adicionar legenda para as arestas
    edge_legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Conexão por Gênero'),
        plt.Line2D([0], [0], color='green', lw=2, label='Conexão por Diretor'),
        plt.Line2D([0], [0], color='purple', lw=2, label='Conexão por Runtime'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Conexão por Budget')
    ]

    plt.legend(handles=legend_elements + edge_legend_elements, loc='best', title='Legenda')

    plt.show()

    # Analisar fatores de conexão
    for factor in connection_factors:
        print(f"Connected movie: {factor['connected_movie']}, Shared genres: {factor['shared_genres']}, Shared director: {factor['shared_director']}, Similar runtime: {factor['similar_runtime']}, Similar budget: {factor['similar_budget']}")

def add_missing_connections(G, df, similarity_threshold=0.3):
    """Add connections for movies that have no connections.""" 
    feature_cols = ['Rating', 'Runtime (Minutes)', 'Revenue (Millions)', 'Director'] + [col for col in df.columns if col.startswith('genre_')]
    feature_matrix = df[feature_cols].values
    similarity_matrix = cosine_similarity(feature_matrix)

    for node in G.nodes:
        if G.degree(node) == 0:
            # Encontrar o filme mais similar que não seja ele mesmo
            node_index = df[df['Title'] == node].index[0]
            similarities = similarity_matrix[node_index]
            sorted_indices = similarities.argsort()[::-1]  # Ordenar índices pela similaridade em ordem decrescente

            # Encontrar o filme mais similar que não seja ele mesmo e esteja dentro dos limites
            most_similar_index = None
            for idx in sorted_indices:
                if idx != node_index and idx < len(df):
                    most_similar_index = idx
                    break

            if most_similar_index is not None:
                most_similar_movie = df.iloc[most_similar_index]['Title']
                similarity_score = similarities[most_similar_index]

                # Adicionar conexão se a similaridade for maior que o limiar
                if similarity_score > similarity_threshold:
                    G.add_edge(node, most_similar_movie, weight=similarity_score)
    return G

def remove_nodes_without_connections(G):
    """Remove nodes that have no connections from the graph.""" 
    nodes_to_remove = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)
    return G

def plot_connection_factors_table(factor_counts):
    """Plot a table showing the count of connections by factor.""" 
    data = [
        ["Fator", "Contagem"],
        ["Gênero Compartilhado", factor_counts['shared_genres']],
        ["Diretor Compartilhado", factor_counts['shared_director']],
        ["Runtime Similar", factor_counts['similar_runtime']],
        ["Orçamento Similar", factor_counts['similar_budget']]
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Contagem de Conexões por Fator")
    plt.show()



def main():
    file_path = 'filmes.csv'

    print("Loading dataset...")
    df = measure_execution_time(load_data, file_path)

    # Aplicar filtros por nota e receita
    print("Preprocessing data (filtered by rating and revenue)...")
    df_filtered, scaler = measure_execution_time(preprocess_data, df.copy(), 0, 200)  # Ajuste os limiares conforme necessário

    print("Creating graph (filtered by rating and revenue)...")
    G_filtered = measure_execution_time(create_graph, df_filtered, similarity_threshold=0.5)

    # Remover nós sem conexões
    print("Removing nodes without connections...")
    G_filtered = remove_nodes_without_connections(G_filtered)

    print("Clustering graph using Girvan-Newman...")
    partition = measure_execution_time(cluster_graph_girvan_newman, G_filtered, k=5)

    print("Analyzing communities...")
    community_info = measure_execution_time(analyze_communities, df_filtered, partition)

    print("Visualizing graph (filtered by rating and revenue)...")
    measure_execution_time(visualize_graph, G_filtered, "Graph Filtered by Rating and Revenue", partition, k=0.6)
    
    # Analisar importância das características para receita
    print("Analyzing feature importance based on importance score...")
    feature_importance = measure_execution_time(feature_importance_analysis, df_filtered, 'Revenue (Millions)')
    plot_feature_importance(feature_importance, 'Top 5 Características Mais Importantes para a Receita')
    
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

    # Encontrar o filme de maior sucesso
    print("Finding the most successful movie...")
    most_successful_movie = find_most_successful_movie(df_filtered)
    print(f"Most successful movie: {most_successful_movie['Title']} with revenue ${most_successful_movie['Revenue (Millions)']}M")

    # Encontrar filmes conectados ao filme de maior sucesso
    print("Finding movies connected to the most successful movie...")
    connected_movies = find_connected_movies(G_filtered, df_filtered, most_successful_movie['Title'])
    print(f"Movies connected to {most_successful_movie['Title']}: {connected_movies}")

    # Analisar fatores de conexão
    print("Analyzing connection factors...")
    connection_factors, factor_counts = analyze_connection_factors(df_filtered, most_successful_movie['Title'], connected_movies)
    for factor in connection_factors:
        print(f"Connected movie: {factor['connected_movie']}, Shared genres: {factor['shared_genres']}, Shared director: {factor['shared_director']}, Similar runtime: {factor['similar_runtime']}, Similar budget: {factor['similar_budget']}")

    # Plotar o filme de maior sucesso e os filmes conectados a ele
    plot_successful_movie_connections(G_filtered, df_filtered, most_successful_movie['Title'], connected_movies, connection_factors, k=0.6)

    # Plotar tabela de contagem de fatores de conexão
    plot_connection_factors_table(factor_counts)

    # Desnormalizar os dados filtrados antes de calcular as estatísticas
    feature_cols = ['Runtime (Minutes)', 'Revenue (Millions)', 'Budget (Million)']
    df_denormalized = denormalize_data(df_filtered, scaler, feature_cols)

    # Calcular e exibir estatísticas descritivas
    print("Calculating statistics for the most successful movies...")
    statistics = calculate_statistics(df_denormalized)
    display_statistics(statistics)

if __name__ == "__main__": 
    main()
