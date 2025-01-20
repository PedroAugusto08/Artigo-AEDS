import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import time  # Import para medir o tempo de execução

matplotlib.use('TkAgg')  # Para usar o backend interativo TkAgg

# Step 1: Load Dataset
def load_data(file_path):
    """Load movie dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, revenue_threshold, rating_threshold):
    """Filter dataset and preprocess."""
    scaler = MinMaxScaler()

    # Filtrar por receita e nota
    print(f"Filtrando filmes com receita maior que {revenue_threshold} e nota maior que {rating_threshold}...")
    df['gross_earn'] = (
        df['gross_earn']
        .str.replace('[\\$,]', '', regex=True)
        .str.replace('M', 'e6', regex=False)
        .str.replace('K', 'e3', regex=False)
        .astype(float)
    )
    df = df[(df['gross_earn'] >= revenue_threshold) & (df['rating'] >= rating_threshold)]

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

# Step 3: Create Graph
def create_graph(df):
    """Create a graph where movies are nodes and edges represent similarity."""
    G = nx.Graph()

    # Add nodes
    for index, row in df.iterrows():
        G.add_node(row['title'], rating=row['rating'])

    # Calculate similarity
    feature_cols = ['rating', 'runtime', 'gross_earn'] + [col for col in df.columns if col.startswith('genre_')]
    feature_matrix = df[feature_cols].values

    # Substituir valores NaN por 0 no feature_matrix
    feature_matrix = np.nan_to_num(feature_matrix)

    similarity_matrix = cosine_similarity(feature_matrix)

    # Add edges based on similarity threshold
    threshold = 0.7
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(df.iloc[i]['title'], df.iloc[j]['title'], weight=similarity_matrix[i, j])

    return G

# Step 4: Analyze Graph
def analyze_graph(G):
    """Perform basic graph analysis."""
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    
    # Calculate centrality
    centrality = nx.degree_centrality(G)
    top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 central nodes:", top_central_nodes)

# Step 5: Visualize Graph
def visualize_graph(G):
    """Visualize the graph using NetworkX and Matplotlib."""
    pos = nx.spring_layout(G, k=0.15, iterations=20)  # Ajuste do layout para uma melhor distribuição
    plt.figure(figsize=(12, 8))
    
    # Desenha os nós
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.6)
    
    # Desenha as arestas com opacidade ajustada
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray')
    
    # Adiciona os rótulos (limita a exibição)
    labels = {node: node for node in G.nodes if np.random.rand() > 0.9}  # Exibe ~10% dos rótulos
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    
    # Salva o gráfico como uma imagem PNG
    plt.savefig("graph_output.png", format="png", dpi=300)
    
    # Exibe o título e mostra o gráfico
    plt.title("Movie Graph")
    plt.axis('off')  # Desativa os eixos para uma visualização mais limpa
    plt.show()

# Measure Execution Time
def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
    return result

# Main Function
def main():
    file_path = 'movies2.csv'  # Replace with the path to your dataset
    revenue_threshold = 1e7  # Receita mínima: 10 milhões
    rating_threshold = 7  # Nota mínima: 7
    
    print("Loading dataset...")
    df = measure_execution_time(load_data, file_path)
    
    print("Preprocessing data...")
    df = measure_execution_time(preprocess_data, df, revenue_threshold, rating_threshold)
    
    print("Creating graph...")
    G = measure_execution_time(create_graph, df)
    
    print("Analyzing graph...")
    measure_execution_time(analyze_graph, G)
    
    print("Visualizing graph...")
    measure_execution_time(visualize_graph, G)

if __name__ == "__main__":
    main()
