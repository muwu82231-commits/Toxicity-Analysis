import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient import discovery
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.style.use('seaborn-v0_8-whitegrid')


def load_cross_community_data(filepaths_dict):
    all_data = []
    
    for community, filepath in filepaths_dict.items():
        try:
            df = pd.read_csv(filepath, encoding='iso-8859-1')
            df.columns = ['subreddit', 'author', 'date', 'post'] + [f'tfidf_{i}' for i in range(256)]
            df['community'] = community
            df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
            all_data.append(df)
            print(f"Loaded {community}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {community}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    user_communities = combined_df.groupby('author')['community'].nunique()
    cross_community_users = user_communities[user_communities > 1].index
    
    cross_community_df = combined_df[combined_df['author'].isin(cross_community_users)]
    
    print(f"\nIdentified {len(cross_community_users)} cross-community users")
    print(f"Percentage of total users: {len(cross_community_users)/combined_df['author'].nunique()*100:.2f}%")
    
    return cross_community_df, combined_df


def calculate_toxicity_with_api(df, api_key):
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
    
    df['toxicity_score'] = toxicity_scores
    df['toxicity_score'] = df['toxicity_score'].fillna(df['toxicity_score'].median())
    
    return df


def calculate_toxicity_diffusion_metrics(df):
    user_community_toxicity = df.groupby(['author', 'community'])['toxicity_score'].agg(['mean', 'std', 'count']).reset_index()
    
    toxicity_diffusion_range = user_community_toxicity.groupby('author')['mean'].std().reset_index()
    toxicity_diffusion_range.columns = ['author', 'toxicity_diffusion_range']
    
    consistency_scores = []
    for author in user_community_toxicity['author'].unique():
        author_data = user_community_toxicity[user_community_toxicity['author'] == author]
        
        if len(author_data) >= 2:
            toxicity_scores = author_data['mean'].values
            cv = np.std(toxicity_scores) / (np.mean(toxicity_scores) + 1e-6)
            consistency = 1 / (1 + cv)
            
            consistency_scores.append({
                'author': author,
                'toxicity_consistency': consistency,
                'community_count': len(author_data)
            })
    
    toxicity_consistency = pd.DataFrame(consistency_scores)
    diffusion_metrics = toxicity_diffusion_range.merge(toxicity_consistency, on='author', how='left')
    
    return diffusion_metrics


def build_cross_community_network(df):
    B = nx.Graph()
    
    users = df['author'].unique()
    communities = df['community'].unique()
    
    B.add_nodes_from(users, bipartite=0)
    B.add_nodes_from(communities, bipartite=1)
    
    user_community_activity = df.groupby(['author', 'community']).size().reset_index(name='posts')
    
    for _, row in user_community_activity.iterrows():
        B.add_edge(row['author'], row['community'], weight=row['posts'])
    
    user_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    user_projection = nx.bipartite.weighted_projected_graph(B, user_nodes)
    
    cross_community_betweenness = nx.betweenness_centrality(user_projection, weight='weight')
    
    return B, cross_community_betweenness


def calculate_topic_similarity(df):
    tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
    
    community_vectors = {}
    for community in df['community'].unique():
        community_data = df[df['community'] == community]
        community_vectors[community] = community_data[tfidf_cols].mean().values
    
    similarity_scores = {}
    communities = list(community_vectors.keys())
    
    for i in range(len(communities)):
        for j in range(i+1, len(communities)):
            sim = cosine_similarity(
                community_vectors[communities[i]].reshape(1, -1),
                community_vectors[communities[j]].reshape(1, -1)
            )[0][0]
            
            similarity_scores[f"{communities[i]}-{communities[j]}"] = sim
    
    return similarity_scores


def prepare_diffusion_panel_data(cross_df, diffusion_metrics, cross_betweenness, topic_similarity):
    cross_df['week'] = cross_df['date'].dt.isocalendar().week
    cross_df['time_period'] = cross_df['date'].dt.strftime('%Y_W%U')
    
    panel_data = cross_df.groupby(['author', 'time_period']).agg({
        'toxicity_score': 'mean',
        'community': lambda x: list(set(x))
    }).reset_index()
    
    panel_data = panel_data.merge(diffusion_metrics, on='author', how='left')
    panel_data['cross_community_betweenness'] = panel_data['author'].map(cross_betweenness)
    
    avg_similarities = []
    for _, row in panel_data.iterrows():
        communities = row['community']
        if len(communities) >= 2:
            similarities = []
            for i in range(len(communities)):
                for j in range(i+1, len(communities)):
                    key1 = f"{communities[i]}-{communities[j]}"
                    key2 = f"{communities[j]}-{communities[i]}"
                    
                    if key1 in topic_similarity:
                        similarities.append(topic_similarity[key1])
                    elif key2 in topic_similarity:
                        similarities.append(topic_similarity[key2])
            
            avg_sim = np.mean(similarities) if similarities else 0
        else:
            avg_sim = 0
        
        avg_similarities.append(avg_sim)
    
    panel_data['avg_topic_similarity'] = avg_similarities
    
    community_participation = cross_df.groupby('author')['community'].nunique().reset_index()
    community_participation.columns = ['author', 'participation_breadth']
    panel_data = panel_data.merge(community_participation, on='author', how='left')
    
    panel_data = panel_data.sort_values(['author', 'time_period'])
    panel_data['prior_toxicity'] = panel_data.groupby('author')['toxicity_score'].shift(1)
    
    panel_data['account_age'] = pd.factorize(panel_data['time_period'])[0]
    panel_data['message_length'] = cross_df.groupby(['author', 'time_period'])['post'].transform(lambda x: x.str.len().mean())
    
    panel_data = panel_data.set_index(['author', 'time_period'])
    
    return panel_data


def run_diffusion_regression_models(panel_data):
    results = {}
    
    controls = ['account_age', 'message_length']
    
    y1 = panel_data['toxicity_diffusion_range']
    X1 = panel_data[['cross_community_betweenness'] + controls].dropna()
    model1 = PanelOLS(y1[X1.index], X1, entity_effects=True, time_effects=True)
    results['model1'] = model1.fit(cov_type='clustered', cluster_entity=True)
    
    panel_data['betweenness_x_similarity'] = (
        panel_data['cross_community_betweenness'] * panel_data['avg_topic_similarity']
    )
    
    X2 = panel_data[['cross_community_betweenness', 'participation_breadth',
                     'avg_topic_similarity', 'betweenness_x_similarity'] + controls].dropna()
    model2 = PanelOLS(y1[X2.index], X2, entity_effects=True, time_effects=True)
    results['model2'] = model2.fit(cov_type='clustered', cluster_entity=True)
    
    panel_data['eigenvector_proxy'] = panel_data['cross_community_betweenness'] ** 2
    panel_data['eigenvector_x_breadth'] = (
        panel_data['eigenvector_proxy'] * panel_data['participation_breadth']
    )
    
    X3 = panel_data[['cross_community_betweenness', 'participation_breadth',
                     'prior_toxicity', 'avg_topic_similarity',
                     'betweenness_x_similarity', 'eigenvector_x_breadth'] + controls].dropna()
    y3 = y1[X3.index]
    model3 = PanelOLS(y3, X3, entity_effects=True, time_effects=True)
    results['model3'] = model3.fit(cov_type='clustered', cluster_entity=True)
    
    y2 = panel_data['toxicity_consistency']
    
    X4 = panel_data[['participation_breadth', 'avg_topic_similarity'] + controls].dropna()
    model4 = PanelOLS(y2[X4.index], X4, entity_effects=True, time_effects=True)
    results['model4'] = model4.fit(cov_type='clustered', cluster_entity=True)
    
    X5 = panel_data[['participation_breadth', 'prior_toxicity',
                     'avg_topic_similarity', 'eigenvector_x_breadth'] + controls].dropna()
    y5 = y2[X5.index]
    model5 = PanelOLS(y5, X5, entity_effects=True, time_effects=True)
    results['model5'] = model5.fit(cov_type='clustered', cluster_entity=True)
    
    return results


def plot_diffusion_network(B, cross_betweenness, toxicity_data):
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(B, k=2, iterations=50)
    
    user_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
    community_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]
    
    top_bridge_users = sorted(cross_betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    bridge_users = [user for user, _ in top_bridge_users]
    
    community_colors = {'suicidewatch': '#ff6b6b', 'anxiety': '#4ecdc4', 'depression': '#45b7d1'}
    for community in community_nodes:
        nx.draw_networkx_nodes(B, pos, nodelist=[community],
                              node_color=community_colors.get(community, 'gray'),
                              node_size=3000, node_shape='s', alpha=0.7)
    
    normal_users = [u for u in user_nodes if u not in bridge_users]
    nx.draw_networkx_nodes(B, pos, nodelist=normal_users,
                          node_color='lightgray', node_size=50, alpha=0.5)
    
    bridge_colors = ['red' if toxicity_data.get(u, 0) > 0.5 else 'green' for u in bridge_users]
    nx.draw_networkx_nodes(B, pos, nodelist=bridge_users,
                          node_color=bridge_colors, node_size=500, alpha=0.9)
    
    toxic_edges = [(u, v) for u, v in B.edges()
                   if u in bridge_users and toxicity_data.get(u, 0) > 0.5]
    nx.draw_networkx_edges(B, pos, edgelist=toxic_edges,
                          edge_color='red', width=2, style='dashed', alpha=0.7)
    
    normal_edges = [(u, v) for u, v in B.edges() if (u, v) not in toxic_edges]
    nx.draw_networkx_edges(B, pos, edgelist=normal_edges,
                          edge_color='gray', width=0.5, alpha=0.3)
    
    nx.draw_networkx_labels(B, pos, labels={n: n for n in community_nodes},
                           font_size=12, font_weight='bold')
    
    bridge_labels = {u: f'User {i+1}' for i, (u, _) in enumerate(top_bridge_users[:3])}
    nx.draw_networkx_labels(B, pos, labels=bridge_labels, font_size=10)
    
    plt.title('Toxicity Diffusion Network across Mental Health Communities',
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('toxicity_diffusion_network.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_bridge_user_behavior(cross_df, cross_betweenness, toxicity_metrics):
    user_analysis = pd.DataFrame(list(cross_betweenness.items()),
                                columns=['author', 'cross_betweenness'])
    user_analysis = user_analysis.merge(toxicity_metrics, on='author', how='left')
    
    avg_toxicity = cross_df.groupby('author')['toxicity_score'].mean().reset_index()
    avg_toxicity.columns = ['author', 'avg_toxicity']
    user_analysis = user_analysis.merge(avg_toxicity, on='author', how='left')
    
    toxicity_amplifiers = user_analysis[
        (user_analysis['cross_betweenness'] > user_analysis['cross_betweenness'].quantile(0.75)) &
        (user_analysis['toxicity_consistency'] > 0.7)
    ]
    
    context_switchers = user_analysis[
        (user_analysis['cross_betweenness'].between(
            user_analysis['cross_betweenness'].quantile(0.25),
            user_analysis['cross_betweenness'].quantile(0.75)
        )) &
        (user_analysis['toxicity_consistency'] < 0.5)
    ]
    
    positive_connectors = user_analysis[
        (user_analysis['avg_toxicity'] < 0.1) &
        (user_analysis['community_count'] > 1)
    ]
    
    results = {
        'toxicity_amplifiers': {
            'count': len(toxicity_amplifiers),
            'percentage': len(toxicity_amplifiers) / len(user_analysis) * 100,
            'avg_betweenness': toxicity_amplifiers['cross_betweenness'].mean(),
            'avg_toxicity': toxicity_amplifiers['avg_toxicity'].mean()
        },
        'context_switchers': {
            'count': len(context_switchers),
            'percentage': len(context_switchers) / len(user_analysis) * 100,
            'avg_betweenness': context_switchers['cross_betweenness'].mean(),
            'avg_toxicity': context_switchers['avg_toxicity'].mean()
        },
        'positive_connectors': {
            'count': len(positive_connectors),
            'percentage': len(positive_connectors) / len(user_analysis) * 100,
            'avg_betweenness': positive_connectors['cross_betweenness'].mean(),
            'avg_toxicity': positive_connectors['avg_toxicity'].mean()
        }
    }
    
    return results, user_analysis


def main():
    print("Starting Toxicity Diffusion Pathways Analysis\n")
    
    api_key = "YOUR_PERSPECTIVE_API_KEY"
    
    print("1. Loading cross-community user data...")
    filepaths = {
        'suicidewatch': 'suicidewatch_2024_features_tfidf_256.csv',
        'anxiety': 'anxiety_2024_features_tfidf_256.csv',
        'depression': 'depression_2024_features_tfidf_256_compressed.csv'
    }
    
    cross_df, all_df = load_cross_community_data(filepaths)
    
    print("\n2. Calculating toxicity scores...")
    cross_df = calculate_toxicity_with_api(cross_df, api_key)
    all_df = calculate_toxicity_with_api(all_df, api_key)
    
    print("\n3. Calculating toxicity diffusion metrics...")
    diffusion_metrics = calculate_toxicity_diffusion_metrics(cross_df)
    print(f"   Average toxicity diffusion range: {diffusion_metrics['toxicity_diffusion_range'].mean():.4f}")
    
    print("\n4. Building cross-community network...")
    B, cross_betweenness = build_cross_community_network(cross_df)
    print(f"   Bipartite network nodes: {B.number_of_nodes()}")
    print(f"   Bipartite network edges: {B.number_of_edges()}")
    
    print("\n5. Calculating inter-community topic similarity...")
    topic_similarity = calculate_topic_similarity(all_df)
    for pair, sim in topic_similarity.items():
        print(f"   {pair}: {sim:.4f}")
    
    print("\n6. Preparing panel data...")
    panel_data = prepare_diffusion_panel_data(cross_df, diffusion_metrics,
                                             cross_betweenness, topic_similarity)
    print(f"   Panel data dimensions: {panel_data.shape}")
    
    print("\n7. Running diffusion mechanism regression models...")
    regression_results = run_diffusion_regression_models(panel_data)
    
    print("\n8. Regression results:")
    for model_name, result in regression_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"R² (within): {result.rsquared_within:.3f}")
        print(f"F-statistic: {result.f_statistic.stat:.2f} (p < 0.001)")
        print(f"Observations: {result.nobs}")
    
    print("\n9. Generating visualization charts...")
    user_toxicity = cross_df.groupby('author')['toxicity_score'].mean().to_dict()
    plot_diffusion_network(B, cross_betweenness, user_toxicity)
    
    print("\n10. Analyzing bridge user behavior patterns...")
    bridge_analysis, user_data = analyze_bridge_user_behavior(cross_df, cross_betweenness,
                                                             diffusion_metrics)
    
    for user_type, stats in bridge_analysis.items():
        print(f"\n{user_type}:")
        print(f"   Count: {stats['count']} ({stats['percentage']:.1f}%)")
        print(f"   Average betweenness centrality: {stats['avg_betweenness']:.4f}")
        print(f"   Average toxicity level: {stats['avg_toxicity']:.4f}")
    
    with open('diffusion_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("Toxicity Diffusion Analysis Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Diffusion Mechanism Regression Results:\n")
        for model_name, result in regression_results.items():
            f.write(f"\n{model_name}:\n")
            f.write(str(result.summary))
            f.write("\n" + "-"*50 + "\n")
        
        f.write("\n\nBridge User Analysis:\n")
        for user_type, stats in bridge_analysis.items():
            f.write(f"\n{user_type}:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
