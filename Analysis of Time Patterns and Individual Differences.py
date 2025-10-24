import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from googleapiclient import discovery
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def load_reddit_data(filepath, encoding='iso-8859-1'):
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        df.columns = ['subreddit', 'author', 'date', 'post'] + [f'tfidf_{i}' for i in range(256)]
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
        df = df.dropna(subset=['author', 'date', 'post'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_toxicity_scores(df, api_key):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    
    toxicity_scores = []
    
    for idx, text in enumerate(df['post']):
        if idx % 100 == 0:
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


def temporal_analysis(df, community_name):
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    weekly_stats = df.groupby(['year', 'week']).agg({
        'author': 'count',
        'toxicity_score': ['mean', 'std', 'median']
    }).reset_index()

    monthly_stats = df.groupby(['year', 'month']).agg({
        'author': 'count',
        'toxicity_score': ['mean', 'std', 'median']
    }).reset_index()

    return {
        'weekly_stats': weekly_stats,
        'monthly_stats': monthly_stats,
        'community': community_name
    }


def individual_trajectory_analysis(df):
    user_trajectories = df.groupby(['author', 'week'])['toxicity_score'].mean().reset_index()
    
    user_stats = user_trajectories.groupby('author').agg({
        'toxicity_score': ['mean', 'std', 'count']
    }).reset_index()

    high_toxicity_threshold = user_stats[('toxicity_score', 'mean')].quantile(0.75)
    high_variability_threshold = user_stats[('toxicity_score', 'std')].quantile(0.75)

    crisis_responsive = user_stats[
        (user_stats[('toxicity_score', 'std')] > high_variability_threshold) &
        (user_stats[('toxicity_score', 'mean')] < high_toxicity_threshold)
    ]['author'].tolist()

    persistent_antagonists = user_stats[
        (user_stats[('toxicity_score', 'mean')] > high_toxicity_threshold) &
        (user_stats[('toxicity_score', 'std')] < high_variability_threshold)
    ]['author'].tolist()

    reactive_defenders = user_stats[
        ~user_stats['author'].isin(crisis_responsive + persistent_antagonists)
    ]['author'].tolist()
    
    consistency_scores = []
    for user in user_trajectories['author'].unique():
        user_data = user_trajectories[user_trajectories['author'] == user].sort_values('week')
        
        if len(user_data) > 2:
            toxicity_values = user_data['toxicity_score'].values
            autocorr = np.corrcoef(toxicity_values[:-1], toxicity_values[1:])[0, 1]
            consistency_scores.append({
                'author': user,
                'temporal_consistency': autocorr,
                'activity_weeks': len(user_data)
            })

    return {
        'trajectories': user_trajectories,
        'user_types': {
            'crisis_responsive': crisis_responsive,
            'persistent_antagonists': persistent_antagonists,
            'reactive_defenders': reactive_defenders
        },
        'consistency': pd.DataFrame(consistency_scores)
    }


def plot_temporal_trends(temporal_results, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for community, results in temporal_results.items():
        weekly_data = results['weekly_stats']
        ax1.plot(range(len(weekly_data)),
                 weekly_data[('author', 'count')],
                 label=f'r/{community}',
                 marker='o',
                 linewidth=2)

    ax1.set_xlabel('Week Number (2024)', fontsize=12)
    ax1.set_ylabel('Number of Posts', fontsize=12)
    ax1.set_title('Weekly Temporal Trends in Mental Health Community Posting Activity',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    months = ['January 2024', 'February 2024', 'March 2024']
    for i, (community, results) in enumerate(temporal_results.items()):
        monthly_data = results['monthly_stats']
        if len(monthly_data) >= 3:
            values = monthly_data[('author', 'count')].values[:3]
            ax2.plot(months, values,
                     label=f'r/{community}',
                     marker='o' if i == 0 else ('s' if i == 1 else '^'),
                     linewidth=2,
                     linestyle='-' if i == 0 else ('--' if i == 1 else ':'))

    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Number of Posts', fontsize=12)
    ax2.set_title('Monthly Aggregated Trend Comparison',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_toxicity_distribution(toxicity_data, save_path=None):
    plt.figure(figsize=(10, 6))

    colors = {'suicidewatch': '#ff4444', 'anxiety': '#66bb6a', 'depression': '#ffb74d'}

    for community, data in toxicity_data.items():
        sns.kdeplot(data=data, label=f'r/{community}',
                    color=colors.get(community, 'gray'),
                    linewidth=2.5)

    plt.xlabel('Toxicity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Kernel Density Estimation of Toxicity Scores',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def perform_statistical_tests(data_dict):
    results = {}

    communities = list(data_dict.keys())
    toxicity_values = [data_dict[c]['toxicity_scores'] for c in communities]
    f_stat, p_value = stats.f_oneway(*toxicity_values)

    results['anova'] = {
        'F_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    return results


def main():
    print("Starting Temporal and Individual Differences Analysis\n")
    
    api_key = "YOUR_PERSPECTIVE_API_KEY"

    print("1. Loading data...")
    data_files = {
        'suicidewatch': 'suicidewatch_2024_features_tfidf_256.csv',
        'anxiety': 'anxiety_2024_features_tfidf_256.csv',
        'depression': 'depression_2024_features_tfidf_256_compressed.csv'
    }

    all_data = {}
    for community, filepath in data_files.items():
        df = load_reddit_data(filepath)
        if df is not None:
            df['toxicity_score'] = calculate_toxicity_scores(df, api_key)
            all_data[community] = df
            print(f"   - {community}: {len(df)} records")

    print("\n2. Performing temporal analysis...")
    temporal_results = {}
    for community, df in all_data.items():
        temporal_results[community] = temporal_analysis(df, community)

    print("\n3. Analyzing individual behavior trajectories...")
    trajectory_results = {}
    for community, df in all_data.items():
        trajectory_results[community] = individual_trajectory_analysis(df)
        user_types = trajectory_results[community]['user_types']
        print(f"   - {community}:")
        print(f"     * Crisis-responsive: {len(user_types['crisis_responsive'])} users")
        print(f"     * Persistent antagonists: {len(user_types['persistent_antagonists'])} users")
        print(f"     * Reactive defenders: {len(user_types['reactive_defenders'])} users")

    print("\n4. Generating visualization charts...")
    plot_temporal_trends(temporal_results, 'temporal_trends.png')

    toxicity_data = {
        community: df['toxicity_score']
        for community, df in all_data.items()
    }
    plot_toxicity_distribution(toxicity_data, 'toxicity_distribution.png')

    print("\n5. Performing statistical tests...")
    data_for_stats = {
        community: {
            'toxicity_scores': df['toxicity_score'].values,
            'temporal_results': temporal_results[community]
        }
        for community, df in all_data.items()
    }

    statistical_results = perform_statistical_tests(data_for_stats)

    print("\nStatistical Test Results:")
    print(f"   - ANOVA F-statistic: {statistical_results['anova']['F_statistic']:.4f}")
    print(f"   - ANOVA p-value: {statistical_results['anova']['p_value']:.4f}")

    with open('temporal_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("Temporal and Individual Differences Analysis Report\n")
        f.write("="*50 + "\n\n")

        f.write("1. Basic Statistics\n")
        for community, df in all_data.items():
            toxicity_mean = df['toxicity_score'].mean()
            toxicity_std = df['toxicity_score'].std()
            f.write(f"   r/{community}:\n")
            f.write(f"     - Mean toxicity score: {toxicity_mean:.4f} (SD={toxicity_std:.4f})\n")
            f.write(f"     - Total posts: {len(df)}\n")
            f.write(f"     - Unique users: {df['author'].nunique()}\n\n")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
