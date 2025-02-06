from read import load_data, preprocess_data, denormalize_data
from graph import create_graph, analyze_graph, analyze_centrality, calculate_node_size, visualize_graph, measure_execution_time, plot_info_table, cluster_graph_leiden, analyze_communities, analyze_shared_features, feature_importance_analysis, plot_feature_importance, calculate_statistics, display_statistics, find_most_successful_movie, find_connected_movies, analyze_connection_factors, plot_successful_movie_connections, add_missing_connections, remove_nodes_without_connections, plot_connection_factors_table

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

    print("Clustering graph using Leiden...")
    partition = measure_execution_time(cluster_graph_leiden, G_filtered)

    print("Analyzing communities...")
    community_info = measure_execution_time(analyze_communities, df_filtered, partition)

    print("Visualizing graph (filtered by rating and revenue)...")
    measure_execution_time(visualize_graph, G_filtered, "Graph Filtered by Rating and Revenue", partition)
    
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
    plot_successful_movie_connections(G_filtered, df_filtered, most_successful_movie['Title'], connected_movies, connection_factors)

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