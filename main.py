import argparse
from read import load_data, preprocess_data, denormalize_data
from graph import create_graph, analyze_centrality, visualize_graph, measure_execution_time, plot_info_table, cluster_graph_louvain, feature_importance_analysis, plot_feature_importance, remove_nodes_without_connections, calculate_average_degree

def main(file_path, similarity_threshold, revenue_threshold):
    print("Loading dataset...")
    df = measure_execution_time(load_data, file_path)

    # Aplicar filtro por receita
    print("Preprocessing data (filtered by revenue)...")
    df_filtered, scaler = measure_execution_time(preprocess_data, df.copy(), revenue_threshold=revenue_threshold)

    print("Creating graph (filtered by revenue)...")
    G_filtered = measure_execution_time(create_graph, df_filtered, similarity_threshold=similarity_threshold)

    # Remover nós sem conexões
    print("Removing nodes without connections...")
    G_filtered = remove_nodes_without_connections(G_filtered)

    print("Clustering graph using Louvain...")
    partition = measure_execution_time(cluster_graph_louvain, G_filtered)

    print("Visualizing graph (filtered by revenue)...")
    measure_execution_time(visualize_graph, G_filtered, "Graph Filtered by Revenue", partition)
    
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
    average_degree = calculate_average_degree(G_filtered)

    # Plotar tabela com informações importantes
    plot_info_table(num_nodes, num_edges, top_degree_nodes, top_closeness_nodes, top_betweenness_nodes, average_degree)

    # Desnormalizar os dados filtrados antes de calcular as estatísticas
    feature_cols = ['Runtime (Minutes)', 'Revenue (Millions)', 'Budget (Million)']
    df_denormalized = denormalize_data(df_filtered, scaler, feature_cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and analyze movie data.')
    parser.add_argument('--file_path', type=str, default='filmes.csv', help='Path to the CSV file containing movie data.')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, help='Threshold for movie similarity to create edges in the graph.')
    parser.add_argument('--revenue_threshold', type=float, default=0, help='Minimum revenue to filter movies.')

    args = parser.parse_args()
    main(args.file_path, args.similarity_threshold, args.revenue_threshold)