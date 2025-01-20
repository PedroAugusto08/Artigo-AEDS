import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import time

def load_data(file_path):
    """Load movie dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, filter_type, threshold):
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

    # Aplicar filtros baseados no tipo (nota ou receita)
    if filter_type == 'rating':
        df = df[df['rating'] >= threshold]
    elif filter_type == 'revenue':
        df = df[df['gross_earn'] >= threshold]

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

def create_graph(df, similarity_threshold=0.5):
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

def visualize_graph(G, title):
    """Visualize the graph using NetworkX and Matplotlib."""
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray')
    labels = {node: node for node in G.nodes}  # Exibe todos os rótulos
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    plt.title(title)
    plt.axis('off')
    plt.show()

def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
    return result

def main():
    file_path = 'movies2.csv'

    print("Loading dataset...")
    df = measure_execution_time(load_data, file_path)

    # Grafo 1: Filtrado por nota
    print("Preprocessing data (filtered by rating)...")
    df_rating = measure_execution_time(preprocess_data, df.copy(), 'rating', 7.5)

    print("Creating graph (filtered by rating)...")
    G_rating = measure_execution_time(create_graph, df_rating, similarity_threshold=0.5)

    print("Analyzing graph (filtered by rating)...")
    measure_execution_time(analyze_graph, G_rating)

    print("Visualizing graph (filtered by rating)...")
    measure_execution_time(visualize_graph, G_rating, "Graph Filtered by Rating")

    # Grafo 2: Filtrado por receita
    print("Preprocessing data (filtered by revenue)...")
    df_revenue = measure_execution_time(preprocess_data, df.copy(), 'revenue', 1e8)

    print("Creating graph (filtered by revenue)...")
    G_revenue = measure_execution_time(create_graph, df_revenue, similarity_threshold=0.5)

    print("Analyzing graph (filtered by revenue)...")
    measure_execution_time(analyze_graph, G_revenue)

    print("Visualizing graph (filtered by revenue)...")
    measure_execution_time(visualize_graph, G_revenue, "Graph Filtered by Revenue")

if __name__ == "__main__":
    main()
