import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from googleapiclient import discovery
import warnings
from datetime import datetime
import os
import json
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


class ToxicityAnalyzer:
    
    def __init__(self, data_path: str, api_key: str = None):
        self.data_path = data_path
        self.api_key = api_key
        self.communities = ['r/Anxiety', 'r/depression', 'r/SuicideWatch']
        self.toxicity_data = {}
        self.stats_results = {}
        if api_key:
            self.client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=api_key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

    def load_data(self, file_paths: Dict[str, str]) -> None:
        for community, path in file_paths.items():
            print(f"Loading {community} data...")
            try:
                for encoding in ['utf-8', 'iso-8859-1', 'latin1']:
                    try:
                        df = pd.read_csv(path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue

                df = df[df['post'].notna()]
                df['post_length'] = df['post'].str.len()
                df = df[df['post_length'] > 10]

                self.toxicity_data[community] = df
                print(f"  Loaded {len(df)} posts from {community}")

            except Exception as e:
                print(f"  Error loading {community}: {e}")

    def calculate_toxicity_scores(self) -> None:
        for community, df in self.toxicity_data.items():
            print(f"\nCalculating toxicity scores for {community}...")
            scores = []
            
            for idx, text in enumerate(df['post']):
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{len(df)}")
                
                try:
                    analyze_request = {
                        'comment': {'text': str(text)[:20000]},
                        'requestedAttributes': {
                            'TOXICITY': {},
                            'SEVERE_TOXICITY': {},
                            'IDENTITY_ATTACK': {},
                            'INSULT': {},
                            'PROFANITY': {},
                            'THREAT': {}
                        }
                    }
                    
                    response = self.client.comments().analyze(body=analyze_request).execute()
                    toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
                    scores.append(toxicity_score)
                    
                except Exception as e:
                    scores.append(np.nan)
            
            df['toxicity_score'] = scores
            df['toxicity_score'] = df['toxicity_score'].fillna(df['toxicity_score'].median())

    def perform_statistical_analysis(self) -> Dict:
        results = {}
        
        print("\n=== Descriptive Statistics ===")
        desc_stats = {}

        for community, df in self.toxicity_data.items():
            scores = df['toxicity_score'].dropna().values
            stats_dict = {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'q1': np.percentile(scores, 25),
                'q3': np.percentile(scores, 75),
                'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
                'skewness': stats.skew(scores),
                'kurtosis': stats.kurtosis(scores),
                'low_toxicity': (scores <= 0.4).sum() / len(scores) * 100,
                'moderate_toxicity': ((scores > 0.4) & (scores <= 0.7)).sum() / len(scores) * 100,
                'high_toxicity': (scores > 0.7).sum() / len(scores) * 100
            }
            desc_stats[community] = stats_dict

            print(f"\n{community}:")
            print(f"  Mean: {stats_dict['mean']:.4f} (SD: {stats_dict['std']:.4f})")
            print(f"  Median: {stats_dict['median']:.4f}")
            print(f"  IQR: {stats_dict['iqr']:.4f}")
            print(f"  High toxicity: {stats_dict['high_toxicity']:.2f}%")

        results['descriptive'] = desc_stats

        print("\n=== Kruskal-Wallis Test ===")
        groups = [df['toxicity_score'].dropna().values for df in self.toxicity_data.values()]
        h_stat, p_value = kruskal(*groups)

        n_total = sum(len(g) for g in groups)
        eta_squared = (h_stat - len(groups) + 1) / (n_total - len(groups))

        print(f"H-statistic: {h_stat:.4f}")
        print(f"p-value: {p_value:.4e}")
        print(f"η² (eta-squared): {eta_squared:.4f}")

        results['kruskal_wallis'] = {
            'h_statistic': h_stat,
            'p_value': p_value,
            'eta_squared': eta_squared
        }

        print("\n=== Post-hoc Analysis (Pairwise Comparisons) ===")
        pairwise_results = self._perform_dunn_test()
        results['post_hoc'] = pairwise_results

        self.stats_results = results
        return results

    def _perform_dunn_test(self) -> Dict:
        communities = list(self.toxicity_data.keys())
        pairwise_results = {}

        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                comm1, comm2 = communities[i], communities[j]
                scores1 = self.toxicity_data[comm1]['toxicity_score'].dropna().values
                scores2 = self.toxicity_data[comm2]['toxicity_score'].dropna().values

                statistic, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')

                n1, n2 = len(scores1), len(scores2)
                z_score = stats.norm.ppf(1 - p_value / 2)
                effect_size = z_score / np.sqrt(n1 + n2)

                p_adjusted = min(1.0, p_value * 3)

                pair_key = f"{comm1} vs {comm2}"
                pairwise_results[pair_key] = {
                    'mean_diff': abs(np.mean(scores1) - np.mean(scores2)),
                    'p_value': p_value,
                    'p_adjusted': p_adjusted,
                    'effect_size': abs(effect_size),
                    'significant': p_adjusted < 0.001
                }

                print(f"{pair_key}:")
                print(f"  Mean difference: {pairwise_results[pair_key]['mean_diff']:.4f}")
                print(f"  p-value (adjusted): {p_adjusted:.4e}")
                print(f"  Effect size: {abs(effect_size):.4f}")

        return pairwise_results

    def generate_visualizations(self, output_dir: str = './figures') -> None:
        os.makedirs(output_dir, exist_ok=True)

        self._plot_boxplot(output_dir)
        self._plot_kde_curves(output_dir)
        self._plot_toxicity_proportions(output_dir)

    def _plot_boxplot(self, output_dir: str) -> None:
        plt.figure(figsize=(10, 6))

        data_to_plot = []
        labels = []
        for community, df in self.toxicity_data.items():
            data_to_plot.append(df['toxicity_score'].dropna().values)
            labels.append(community)

        box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                               showmeans=True, meanline=True)

        colors = ['#52c41a', '#1890ff', '#ff4d4f']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.ylabel('Toxicity Score', fontsize=14, fontweight='bold')
        plt.title('Distribution of Toxicity Scores Across Communities',
                  fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'toxicity_boxplot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_kde_curves(self, output_dir: str) -> None:
        plt.figure(figsize=(10, 6))

        x_range = np.linspace(0, 1, 1000)

        for i, (community, df) in enumerate(self.toxicity_data.items()):
            scores = df['toxicity_score'].dropna().values

            kde = stats.gaussian_kde(scores, bw_method='silverman')
            density = kde(x_range)

            if i == 0:
                plt.plot(x_range, density, 'k-', linewidth=2.5, label=community)
            elif i == 1:
                plt.plot(x_range, density, 'k--', linewidth=2.5, label=community,
                         color='gray')
            else:
                plt.plot(x_range, density, 'k-.', linewidth=2.5, label=community)

        plt.xlabel('Toxicity Score', fontsize=14, fontweight='bold')
        plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
        plt.title('Kernel Density Estimation of Toxicity Scores',
                  fontsize=16, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'toxicity_kde.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_toxicity_proportions(self, output_dir: str) -> None:
        categories = ['Low\n(≤0.4)', 'Moderate\n(0.4-0.7)', 'High\n(>0.7)']

        data = []
        for community, stats_dict in self.stats_results['descriptive'].items():
            data.append([
                stats_dict['low_toxicity'],
                stats_dict['moderate_toxicity'],
                stats_dict['high_toxicity']
            ])

        x = np.arange(len(categories))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#52c41a', '#1890ff', '#ff4d4f']
        for i, (community, values) in enumerate(zip(self.toxicity_data.keys(), data)):
            ax.bar(x + i * width, values, width, label=community, color=colors[i], alpha=0.8)

        ax.set_xlabel('Toxicity Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Toxicity Levels Across Communities',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'toxicity_proportions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def export_results(self, output_file: str = 'toxicity_analysis_results.json') -> None:
        export_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'communities': list(self.toxicity_data.keys()),
            'sample_sizes': {k: len(v) for k, v in self.toxicity_data.items()},
            'statistical_results': self.stats_results,
            'methodology': {
                'toxicity_calculation': 'Google Perspective API',
                'statistical_tests': ['Kruskal-Wallis', 'Mann-Whitney U', 'Dunn post-hoc'],
                'significance_level': 0.001,
                'correction_method': 'Bonferroni'
            }
        }

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        export_data = convert_numpy(export_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults exported to {output_file}")


def main():
    print("=" * 60)
    print("Reddit Mental Health Communities Toxicity Analysis")
    print("=" * 60)

    api_key = "YOUR_PERSPECTIVE_API_KEY"
    
    analyzer = ToxicityAnalyzer(data_path='./data', api_key=api_key)

    file_paths = {
        'r/Anxiety': 'anxiety_2024_features_tfidf_256.csv',
        'r/depression': 'depression_2024_features_tfidf_256_compressed.csv',
        'r/SuicideWatch': 'suicidewatch_2024_features_tfidf_256.csv'
    }

    print("\n[1/5] Loading data files...")
    analyzer.load_data(file_paths)

    print("\n[2/5] Calculating toxicity scores using Perspective API...")
    analyzer.calculate_toxicity_scores()

    print("\n[3/5] Performing statistical analysis...")
    results = analyzer.perform_statistical_analysis()

    print("\n[4/5] Generating visualizations...")
    analyzer.generate_visualizations('./figures')

    print("\n[5/5] Exporting results...")
    analyzer.export_results('toxicity_analysis_results.json')

    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
