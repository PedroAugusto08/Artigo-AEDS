import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load Dataset
def load_data(file_path):
    """Load movie dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    """Normalize numerical features and encode genres."""
    # Normalize numerical columns
    scaler = MinMaxScaler()
    df[['revenue', 'popularity', 'runtime']] = scaler.fit_transform(df[['revenue', 'popularity', 'runtime']])
    
    # One-hot encode genres
    genres_encoded = df['genre'].str.get_dummies(sep=',')
    df = pd.concat([df, genres_encoded], axis=1)
    return df

# Step 3: Create Graph
def create_graph(df):
    """Create a graph where movies are nodes and edges represent similarity."""
    G = nx.Graph()

    # Add nodes
    for index, row in df.iterrows():
        G.add_node(row['movie_id'], title=row['title'], rating=row['rating'])

    # Calculate similarity
    feature_cols = ['revenue', 'popularity', 'runtime'] + list(df.columns[df.columns.str.contains('|'.join(df['genre'].unique()))])
    feature_matrix = df[feature_cols].values
    similarity_matrix = cosine_similarity(feature_matrix)

    # Add edges based on similarity threshold
    threshold = 0.7
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(df.iloc[i]['movie_id'], df.iloc[j]['movie_id'], weight=similarity_matrix[i, j])
    
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
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    
    # Draw edges
    edges = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    
    plt.title("Movie Graph")
    plt.show()

# Main Function
def main():
    file_path = 'movies.csv'  # Replace with the path to your dataset
    
    print("Loading dataset...")
    df = load_data(file_path)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Creating graph...")
    G = create_graph(df)
    
    print("Analyzing graph...")
    analyze_graph(G)
    
    print("Visualizing graph...")
    visualize_graph(G)

if __name__ == "__main__":
    main()
