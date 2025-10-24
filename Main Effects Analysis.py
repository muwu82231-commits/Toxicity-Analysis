import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from googleapiclient import discovery
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def load_reddit_data(filepaths_dict):
    all_data = []

    for community, filepath in filepaths_dict.items():
        try:
            df = pd.read_csv(filepath, encoding='iso-8859-1')
            df.columns = ['subreddit', 'author', 'date', 'post'] + [f'tfidf_{i}' for i in range(256)]
            df['community'] = community
            df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
            df = df.dropna(subset=['author', 'date', 'post'])
            all_data.append(df)
            print(f"Loaded {community}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {community}: {e}")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def calculate_toxicity_scores_with_api(df, api_key):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    
    toxicity_scores = []
    
    for idx, text in enumerate(df['post']):
        if idx % 500 == 0:
            print(f"Progress: {idx}/{len(df)}")
        
        try:
            analyze_request = {
                'comment': {'text': str(text)[:20000]},
                'requestedAttributes': {'TOXICITY': {}}
            }
            response = client.comments().analyze(body=analyze_request).execute()
            score = response['attributeScores']['TOXICITY']['summaryScore']['value']
            toxicity_scores.append(score)
        except:
            toxicity_scores.append(np.nan)
    
    toxicity_scores = pd.Series(toxicity_scores)
    toxicity_scores = toxicity_scores.fillna(toxicity_scores.median())
    return toxicity_scores


def build_user_interaction_network(df):
    G = nx.Graph()
    
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    
    for (year, week, community), group in df.groupby(['year', 'week', 'community']):
        posts = group.sort_values('date')
        authors = posts['author'].tolist()
        
        for i in range(len(authors) - 1):
            if authors[i] != authors[i+1]:
                if G.has_edge(authors[i], authors[i+1]):
                    G[authors[i]][authors[i+1]]['weight'] += 1
                else:
                    G.add_edge(authors[i], authors[i+1], weight=1)
    
    user_attributes = {}
    for user in G.nodes():
        user_data = df[df['author'] == user]
        user_attributes[user] = {
            'post_count': len(user_data),
            'communities': list(user_data['community'].unique()),
            'avg_toxicity': user_data['toxicity_score'].mean() if 'toxicity_score' in df.columns else 0
        }
    
    nx.set_node_attributes(G, user_attributes)
    
    return G, user_attributes


def calculate_centrality_measures(G):
    print("Calculating network centrality metrics...")
    
    centrality_measures = pd.DataFrame()
    
    degree_centrality = nx.degree_centrality(G)
    centrality_measures['degree_centrality'] = pd.Series(degree_centrality)
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        centrality_measures['eigenvector_centrality'] = pd.Series(eigenvector_centrality)
    except:
        print("Using alternative eigenvector centrality calculation")
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        centrality_measures['eigenvector_centrality'] = pd.Series(eigenvector_centrality)
    
    closeness_centrality = {}
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        closeness = nx.closeness_centrality(subgraph)
        closeness_centrality.update(closeness)
    centrality_measures['closeness_centrality'] = pd.Series(closeness_centrality)
    
    betweenness_centrality = nx.betweenness_centrality(G)
    centrality_measures['betweenness_centrality'] = pd.Series(betweenness_centrality)
    
    centrality_measures['author'] = centrality_measures.index
    
    return centrality_measures


def calculate_cross_community_metrics(df):
    cross_community_stats = df.groupby('author').agg({
        'community': lambda x: len(set(x)),
        'post': 'count',
        'toxicity_score': 'mean'
    }).reset_index()
    
    cross_community_stats.columns = ['author', 'community_count', 'total_posts', 'avg_toxicity']
    cross_community_stats['is_cross_community'] = cross_community_stats['community_count'] > 1
    
    return cross_community_stats


def prepare_panel_data(df, centrality_df, cross_community_df):
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    df['time_period'] = df['year'].astype(str) + '_W' + df['week'].astype(str).str.zfill(2)
    
    panel_data = df.groupby(['author', 'time_period']).agg({
        'toxicity_score': 'mean',
        'post': 'count',
        'community': lambda x: list(x.unique())
    }).reset_index()
    
    panel_data = panel_data.merge(centrality_df, on='author', how='left')
    panel_data = panel_data.merge(cross_community_df[['author', 'community_count']],
                                  on='author', how='left')
    
    first_appearance = df.groupby('author')['date'].min().reset_index()
    first_appearance.columns = ['author', 'first_post_date']
    panel_data = panel_data.merge(first_appearance, on='author', how='left')
    
    panel_data['time_numeric'] = pd.factorize(panel_data['time_period'])[0]
    panel_data['account_age'] = panel_data['time_numeric']
    panel_data['activity_level'] = panel_data['post']
    panel_data['message_length'] = df.groupby(['author', 'time_period'])['post'].transform(lambda x: x.str.len().mean())
    
    panel_data = panel_data.set_index(['author', 'time_period'])
    
    return panel_data


def run_fixed_effects_regression(panel_data):
    results = {}
    
    y = panel_data['toxicity_score']
    controls = ['account_age', 'activity_level', 'message_length']
    
    X1 = panel_data[['degree_centrality'] + controls].dropna()
    model1 = PanelOLS(y[X1.index], X1, entity_effects=True, time_effects=True)
    results['model1'] = model1.fit(cov_type='clustered', cluster_entity=True)
    
    X2 = panel_data[['eigenvector_centrality'] + controls].dropna()
    model2 = PanelOLS(y[X2.index], X2, entity_effects=True, time_effects=True)
    results['model2'] = model2.fit(cov_type='clustered', cluster_entity=True)
    
    X3 = panel_data[['closeness_centrality'] + controls].dropna()
    model3 = PanelOLS(y[X3.index], X3, entity_effects=True, time_effects=True)
    results['model3'] = model3.fit(cov_type='clustered', cluster_entity=True)
    
    X4 = panel_data[['betweenness_centrality'] + controls].dropna()
    model4 = PanelOLS(y[X4.index], X4, entity_effects=True, time_effects=True)
    results['model4'] = model4.fit(cov_type='clustered', cluster_entity=True)
    
    X_full = panel_data[['degree_centrality', 'eigenvector_centrality',
                        'closeness_centrality', 'betweenness_centrality'] + controls].dropna()
    model_full = PanelOLS(y[X_full.index], X_full, entity_effects=True, time_effects=True)
    results['model_full'] = model_full.fit(cov_type='clustered', cluster_entity=True)
    
    return results


def print_regression_results(results):
    print("\n" + "="*80)
    print("Fixed Effects Regression Results")
    print("="*80)
    
    summary_table = pd.DataFrame()
    
    for model_name, result in results.items():
        coef_summary = pd.DataFrame({
            'coefficient': result.params,
            'std_error': result.std_errors,
            'p_value': result.pvalues
        })
        
        coef_summary['significance'] = ''
        coef_summary.loc[coef_summary['p_value'] < 0.01, 'significance'] = '***'
        coef_summary.loc[(coef_summary['p_value'] >= 0.01) & 
                        (coef_summary['p_value'] < 0.05), 'significance'] = '**'
        coef_summary.loc[(coef_summary['p_value'] >= 0.05) & 
                        (coef_summary['p_value'] < 0.1), 'significance'] = '*'
        
        coef_summary['formatted'] = coef_summary.apply(
            lambda row: f"{row['coefficient']:.3f}{row['significance']}\n({row['std_error']:.3f})",
            axis=1
        )
        
        summary_table[model_name] = coef_summary['formatted']
    
    stats_row = pd.Series()
    for model_name, result in results.items():
        stats_row[model_name] = f"R²={result.rsquared_within:.3f}\nF={result.f_statistic.stat:.2f}***\nN={result.nobs}"
    
    summary_table = pd.concat([summary_table, pd.DataFrame(stats_row).T])
    
    print(summary_table)
    
    return summary_table


def analyze_network_structure(G, df):
    network_stats = {
        'Total posts': len(df),
        'Unique users': df['author'].nunique(),
        'Observation window (weeks)': df['week'].nunique()
    }
    
    community_distribution = df.groupby('community').size()
    for community, count in community_distribution.items():
        network_stats[f'r/{community} posts'] = count
        network_stats[f'r/{community} percentage'] = f"{count/len(df)*100:.1f}%"
    
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    network_stats.update({
        'Average degree centrality': np.mean(degree_values) if degree_values else 0,
        'Degree centrality std dev': np.std(degree_values) if degree_values else 0,
        'Network density': nx.density(G),
        'Clustering coefficient': nx.average_clustering(G)
    })
    
    cross_community_users = df.groupby('author')['community'].nunique()
    network_stats.update({
        'Cross-community active users': sum(cross_community_users > 1),
        'Cross-community participation ratio': f"{sum(cross_community_users > 1)/len(cross_community_users)*100:.2f}%"
    })
    
    return network_stats


def plot_centrality_distributions(centrality_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    centrality_measures = ['degree_centrality', 'eigenvector_centrality',
                          'closeness_centrality', 'betweenness_centrality']
    titles = ['Degree Centrality', 'Eigenvector Centrality',
              'Closeness Centrality', 'Betweenness Centrality']
    
    for i, (measure, title) in enumerate(zip(centrality_measures, titles)):
        axes[i].hist(centrality_df[measure].dropna(), bins=50, density=True,
                    alpha=0.7, color='gray', edgecolor='black')
        centrality_df[measure].dropna().plot.kde(ax=axes[i], color='black', linewidth=2)
        
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Centrality Score', fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        mean_val = centrality_df[measure].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[i].text(mean_val*1.1, axes[i].get_ylim()[1]*0.9,
                    f'Mean: {mean_val:.3f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('centrality_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("Starting Structural Centrality Analysis\n")
    
    api_key = "YOUR_PERSPECTIVE_API_KEY"
    
    print("1. Loading data...")
    filepaths = {
        'suicidewatch': 'suicidewatch_2024_features_tfidf_256.csv',
        'anxiety': 'anxiety_2024_features_tfidf_256.csv',
        'depression': 'depression_2024_features_tfidf_256_compressed.csv'
    }
    
    df = load_reddit_data(filepaths)
    print(f"   Total records: {len(df)}")
    
    print("\n2. Calculating toxicity scores...")
    df['toxicity_score'] = calculate_toxicity_scores_with_api(df, api_key)
    print(f"   Average toxicity score: {df['toxicity_score'].mean():.4f}")
    
    print("\n3. Building user interaction network...")
    G, user_attributes = build_user_interaction_network(df)
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    
    print("\n4. Calculating network centrality metrics...")
    centrality_df = calculate_centrality_measures(G)
    
    print("\n5. Calculating cross-community participation metrics...")
    cross_community_df = calculate_cross_community_metrics(df)
    print(f"   Cross-community users: {sum(cross_community_df['is_cross_community'])}")
    
    print("\n6. Analyzing network structure features...")
    network_stats = analyze_network_structure(G, df)
    
    print("\n7. Preparing panel data...")
    panel_data = prepare_panel_data(df, centrality_df, cross_community_df)
    print(f"   Panel data dimensions: {panel_data.shape}")
    
    print("\n8. Running fixed effects regression analysis...")
    regression_results = run_fixed_effects_regression(panel_data)
    
    print("\n9. Regression results:")
    results_table = print_regression_results(regression_results)
    
    print("\n10. Generating visualization charts...")
    plot_centrality_distributions(centrality_df)
    
    with open('network_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("Network Structure Analysis Results\n")
        f.write("="*50 + "\n\n")
        
        for key, value in network_stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n\nRegression Results:\n")
        for model_name, result in regression_results.items():
            f.write(f"\n{model_name}:\n")
            f.write(str(result.summary))
            f.write("\n" + "-"*50 + "\n")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
