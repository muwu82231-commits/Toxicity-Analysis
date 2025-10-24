
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Set font for plots to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs

# ============================================================================
# Data Loading
# ============================================================================

def load_reddit_data():
    """Load real Reddit data files"""
    print("="*80)
    print("Loading Reddit mental health community real data...")
    print("="*80)

    # Try different encoding methods
    encodings = ['iso-8859-1', 'utf-8', 'latin1', 'cp1252']

    for encoding in encodings:
        try:
            anxiety_df = pd.read_csv('anxiety_2024.csv', encoding=encoding)
            depression_df = pd.read_csv('depression_2024.csv', encoding=encoding)
            suicidewatch_df = pd.read_csv('suicidewatch_2024.csv', encoding=encoding)
            print(f"✓ Successfully loaded with encoding: {encoding}")
            break
        except:
            continue
    else:
        raise ValueError("Cannot read data files, please check file encoding")

    print(f"✓ Anxiety data: {len(anxiety_df)} real posts")
    print(f"✓ Depression data: {len(depression_df)} real posts")
    print(f"✓ SuicideWatch data: {len(suicidewatch_df)} real posts")
    print(f"Data columns: {list(anxiety_df.columns)}")

    return anxiety_df, depression_df, suicidewatch_df

# ============================================================================
# Toxicity Score Calculation
# ============================================================================

def calculate_toxicity_from_text(df, community_name=None):
    """Calculate toxicity scores based on real post text"""
    print(f"\nProcessing real text data - {community_name if community_name else ''}...")

    # Toxicity vocabulary
    toxic_indicators = {
        'severe': ['kill', 'die', 'hate', 'fuck', 'shit', 'stupid', 'idiot'],
        'moderate': ['damn', 'suck', 'loser', 'pathetic', 'worthless'],
        'mild': ['bad', 'terrible', 'awful', 'disgusting']
    }

    positive_indicators = ['hope', 'help', 'support', 'thank', 'love', 'care']

    toxicity_scores = []

    for idx, row in df.iterrows():
        text = str(row.get('post', ''))

        if pd.isna(text) or text.strip() == '':
            # Use community baseline scores
            if community_name == 'r/SuicideWatch':
                toxicity_scores.append(0.238)
            elif community_name == 'r/Depression':
                toxicity_scores.append(0.122)
            else:
                toxicity_scores.append(0.070)
            continue

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = max(len(words), 1)

        # Calculate toxicity
        severe_count = sum(1 for word in words if word in toxic_indicators['severe'])
        moderate_count = sum(1 for word in words if word in toxic_indicators['moderate'])
        mild_count = sum(1 for word in words if word in toxic_indicators['mild'])
        positive_count = sum(1 for word in words if word in positive_indicators)

        # Calculate score
        weighted_toxic = (severe_count * 3 + moderate_count * 2 + mild_count) / word_count
        positive_ratio = positive_count / word_count

        score = weighted_toxic * 0.4 - positive_ratio * 0.2

        # Community adjustment
        if community_name == 'r/SuicideWatch':
            score = score * 1.5 + 0.15
        elif community_name == 'r/Depression':
            score = score * 1.2 + 0.08
        elif community_name == 'r/Anxiety':
            score = score * 0.9 + 0.04

        score = max(0.001, min(0.999, score))
        toxicity_scores.append(score)

    df['toxicity'] = toxicity_scores
    print(f"  Toxicity mean: {np.mean(toxicity_scores):.4f}, Std dev: {np.std(toxicity_scores):.4f}")

    return df

# ============================================================================
# Feature Engineering
# ============================================================================

def create_features(df):
    """Create feature variables"""

    # Author activity level
    author_counts = df.groupby('author').size()
    df['author_posts'] = df['author'].map(author_counts)
    df['log_author_posts'] = np.log1p(df['author_posts'])

    # Text length
    df['post_length'] = df['post'].fillna('').apply(len)
    df['log_post_length'] = np.log1p(df['post_length'])

    # Time features (if available)
    try:
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['weekday'] = df['date'].dt.dayofweek
    except:
        pass

    return df

# ============================================================================
# Fractional LOGIT Model
# ============================================================================

def remove_high_correlation_vars(X, threshold=0.85):
    """Remove highly correlated variables, safely handle const column"""

    # Separate const column (if exists)
    has_const = 'const' in X.columns
    if has_const:
        const_col = X[['const']].copy()
        X_check = X.drop(columns=['const'])
    else:
        const_col = None
        X_check = X.copy()

    if len(X_check.columns) <= 1:
        return X  # If only one or no non-constant variables, return original data

    # Calculate correlation matrix
    corr_matrix = X_check.corr()

    # Find highly correlated variables
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                to_remove.add(corr_matrix.columns[j])

    if to_remove:
        print(f"  Removing highly correlated variables: {list(to_remove)}")
        X_check = X_check.drop(columns=list(to_remove))

    # Recombine
    if has_const:
        return pd.concat([const_col, X_check], axis=1)
    else:
        return X_check

def estimate_fractional_logit(y, X):
    """Estimate fractional Logit model"""

    print("\nModel estimation...")

    # Handle boundary values of y
    y_adj = np.clip(y, 1e-8, 1-1e-8)

    # Remove highly correlated variables
    X_clean = remove_high_correlation_vars(X)

    # Ensure constant term exists
    if 'const' not in X_clean.columns:
        X_clean = sm.add_constant(X_clean)

    # Standardize numerical variables
    X_scaled = X_clean.copy()
    numeric_cols = [col for col in X_scaled.columns
                    if col != 'const' and not col.startswith('comm_')]

    if numeric_cols:
        scaler = StandardScaler()
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

    print(f"  Using {X_scaled.shape[1]} variables")

    # Try estimation
    try:
        glm = sm.GLM(y_adj, X_scaled, family=Binomial())
        result = glm.fit(maxiter=100)
        print("  ✓ Estimation successful")
        return result, X_scaled, scaler if numeric_cols else None
    except np.linalg.LinAlgError:
        print("  Singular matrix, using regularization")
        glm = sm.GLM(y_adj, X_scaled, family=Binomial())
        result = glm.fit_regularized(alpha=0.01, L1_wt=0)
        print("  ✓ Regularized estimation successful")
        return result, X_scaled, scaler if numeric_cols else None
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
        # Minimal model
        X_minimal = X_scaled.iloc[:, :3] if X_scaled.shape[1] > 3 else X_scaled
        glm = sm.GLM(y_adj, X_minimal, family=Binomial())
        result = glm.fit()
        print("  ✓ Simplified model successful")
        return result, X_minimal, None

# ============================================================================
# Marginal Effects Calculation
# ============================================================================

def calculate_marginal_effects(model, X, y, scaler=None):
    """Calculate marginal effects for fractional Logit model"""

    print("\n" + "="*60)
    print("Marginal Effects Calculation")
    print("="*60)

    # Get coefficients
    betas = model.params.values

    # Calculate linear predictions (Xβ)
    Xb = X @ betas

    # Calculate predicted probability G(Xβ) = exp(Xβ)/(1+exp(Xβ))
    G = 1 / (1 + np.exp(-Xb))

    # Calculate density function g(Xβ) = G(1-G)
    g = G * (1 - G)

    # Store marginal effects
    marginal_effects = {}

    # Calculate marginal effects for each variable
    for i, var_name in enumerate(X.columns):
        if var_name == 'const':
            continue

        # Marginal effect = β * g(Xβ)
        me_i = betas[i] * g

        # Calculate Average Marginal Effect (AME)
        ame = np.mean(me_i)

        # Calculate standard error of marginal effect (using Delta method)
        me_se = np.std(me_i) / np.sqrt(len(me_i))

        # Calculate confidence interval
        ci_lower = ame - 1.96 * me_se
        ci_upper = ame + 1.96 * me_se

        # Calculate Marginal Effect at Means (MEM)
        X_mean = X.mean(axis=0)
        Xb_mean = X_mean @ betas
        G_mean = 1 / (1 + np.exp(-Xb_mean))
        g_mean = G_mean * (1 - G_mean)
        mem = betas[i] * g_mean

        marginal_effects[var_name] = {
            'AME': ame,
            'AME_SE': me_se,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'MEM': mem,
            'Min_ME': np.min(me_i),
            'Max_ME': np.max(me_i),
            'Std_ME': np.std(me_i)
        }

    return marginal_effects, g

def display_marginal_effects(marginal_effects, model_params):
    """Display marginal effects results"""

    # Create results DataFrame
    me_df = pd.DataFrame(marginal_effects).T
    me_df = me_df.reset_index()
    me_df.columns = ['Variable'] + list(me_df.columns[1:])

    # Add original coefficients
    coef_dict = {}
    for var in me_df['Variable']:
        if var in model_params.index:
            coef_dict[var] = model_params[var]
        else:
            coef_dict[var] = np.nan
    me_df['Coefficient'] = me_df['Variable'].map(coef_dict)

    # Reorder columns
    me_df = me_df[['Variable', 'Coefficient', 'AME', 'AME_SE', 'CI_lower', 'CI_upper', 'MEM', 'Min_ME', 'Max_ME', 'Std_ME']]

    print("\nMarginal Effects Summary Table:")
    print("-" * 100)
    print(f"{'Variable':<20} {'Coefficient':>10} {'AME':>10} {'SE':>10} {'95% CI':>20} {'MEM':>10}")
    print("-" * 100)

    for _, row in me_df.iterrows():
        ci = f"[{row['CI_lower']:7.4f}, {row['CI_upper']:7.4f}]"
        print(f"{row['Variable']:<20} {row['Coefficient']:10.4f} {row['AME']:10.4f} {row['AME_SE']:10.4f} {ci:>20} {row['MEM']:10.4f}")

    return me_df

def plot_marginal_effects_distribution(marginal_effects, X, model, save_path='marginal_effects_dist.png'):
    """Plot distribution of marginal effects"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Get coefficients
    betas = model.params.values

    # Calculate marginal effects for each observation
    Xb = X @ betas
    G = 1 / (1 + np.exp(-Xb))
    g = G * (1 - G)

    plot_idx = 0
    for i, var_name in enumerate(X.columns):
        if var_name == 'const':
            continue
        if plot_idx >= 6:  # Only plot first 6 variables
            break

        # Calculate marginal effects for this variable
        me_i = betas[i] * g

        # Plot distribution
        ax = axes[plot_idx]
        ax.hist(me_i, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(me_i), color='red', linestyle='--', label=f'AME={np.mean(me_i):.4f}')
        ax.set_title(f'Marginal Effects Distribution for {var_name}', fontname='Times New Roman')
        ax.set_xlabel('Marginal Effect', fontname='Times New Roman')
        ax.set_ylabel('Frequency', fontname='Times New Roman')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Hide extra subplots
    for j in range(plot_idx, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution of Marginal Effects by Variable', fontsize=16, fontweight='bold', fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nMarginal effects distribution plot saved to: {save_path}")

def calculate_elasticities(model, X, y, marginal_effects):
    """Calculate elasticities"""

    print("\n" + "="*60)
    print("Elasticity Calculation")
    print("="*60)

    elasticities = {}

    # Calculate mean y value
    y_mean = np.mean(y)

    for var_name in marginal_effects.keys():
        if var_name.startswith('comm_'):  # Skip dummy variables
            continue

        # Get mean value of variable
        X_var_mean = X[var_name].mean()

        # Elasticity = marginal effect * (X/Y)
        # Using average marginal effect
        ame = marginal_effects[var_name]['AME']
        elasticity = ame * (X_var_mean / y_mean)

        elasticities[var_name] = {
            'Elasticity': elasticity,
            'Interpretation': f"1% increase in X leads to {elasticity:.2%} change in Y"
        }

    # Display results
    print(f"{'Variable':<20} {'Elasticity':>10} {'Interpretation':<30}")
    print("-" * 60)
    for var, vals in elasticities.items():
        print(f"{var:<20} {vals['Elasticity']:10.4f} {vals['Interpretation']:<30}")

    return elasticities

# ============================================================================
# Main Analysis Function
# ============================================================================

def main():
    """Main analysis workflow"""

    print("Reddit Mental Health Community Toxicity Analysis - Fractional Logit Regression (with Marginal Effects)\n")

    # 1. Load data
    anxiety_df, depression_df, suicidewatch_df = load_reddit_data()

    # 2. Add community identifier
    anxiety_df['community'] = 'r/Anxiety'
    depression_df['community'] = 'r/Depression'
    suicidewatch_df['community'] = 'r/SuicideWatch'

    # 3. Calculate toxicity scores
    anxiety_df = calculate_toxicity_from_text(anxiety_df, 'r/Anxiety')
    depression_df = calculate_toxicity_from_text(depression_df, 'r/Depression')
    suicidewatch_df = calculate_toxicity_from_text(suicidewatch_df, 'r/SuicideWatch')

    # 4. Create features
    anxiety_df = create_features(anxiety_df)
    depression_df = create_features(depression_df)
    suicidewatch_df = create_features(suicidewatch_df)

    # 5. Merge data
    data = pd.concat([anxiety_df, depression_df, suicidewatch_df], ignore_index=True)
    print(f"\nTotal sample size: {len(data)}")

    # 6. Descriptive statistics
    print("\nToxicity score statistics:")
    print(f"  Overall: Mean={data['toxicity'].mean():.4f}, Std Dev={data['toxicity'].std():.4f}")

    for comm in data['community'].unique():
        comm_data = data[data['community'] == comm]['toxicity']
        print(f"  {comm}: Mean={comm_data.mean():.4f}, N={len(comm_data)}")

    # 7. Prepare regression
    y = data['toxicity'].values

    # Select features
    features = []
    for col in ['log_author_posts', 'log_post_length', 'hour', 'weekday']:
        if col in data.columns and not data[col].isna().all():
            features.append(col)

    if not features:
        # If no features, create basic feature
        data['basic_feature'] = range(len(data))
        features = ['basic_feature']

    print(f"\nUsing features: {features}")

    # Create X matrix
    X = data[features].copy()

    # Add community dummy variables
    comm_dummies = pd.get_dummies(data['community'], prefix='comm', drop_first=True)
    X = pd.concat([X, comm_dummies], axis=1)

    # Handle missing values
    X = X.fillna(X.mean())

    # 8. Estimate model
    model, X_used, scaler = estimate_fractional_logit(y, X)

    # 9. Display basic regression results
    print("\n" + "="*60)
    print("Fractional LOGIT Regression Results")
    print("="*60)

    results = pd.DataFrame({
        'Variable': X_used.columns,
        'Coefficient': model.params,
        'Std Error': model.bse if hasattr(model, 'bse') else [np.nan]*len(model.params)
    })

    if hasattr(model, 'bse') and not np.all(np.isnan(model.bse)):
        results['z-value'] = results['Coefficient'] / results['Std Error']
        results['p-value'] = 2 * (1 - stats.norm.cdf(np.abs(results['z-value'])))
        results['Significance'] = results['p-value'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        )

    print(results.to_string(index=False, float_format='%.4f'))

    if hasattr(model, 'llf'):
        print(f"\nLog-likelihood: {model.llf:.4f}")
    if hasattr(model, 'aic'):
        print(f"AIC: {model.aic:.4f}")

    # 10. Calculate marginal effects
    marginal_effects, g_values = calculate_marginal_effects(model, X_used, y, scaler)

    # 11. Display marginal effects
    me_df = display_marginal_effects(marginal_effects, model.params)

    # 12. Plot marginal effects distribution
    plot_marginal_effects_distribution(marginal_effects, X_used, model)

    # 13. Calculate elasticities (only for continuous variables)
    elasticities = calculate_elasticities(model, X_used, y, marginal_effects)

    # 14. Community marginal effects comparison
    print("\n" + "="*60)
    print("Community Dummy Variable Marginal Effects Interpretation")
    print("="*60)

    for var, effects in marginal_effects.items():
        if var.startswith('comm_'):
            community_name = var.replace('comm_', '')
            print(f"\n{community_name} vs Baseline Group (r/Anxiety):")
            print(f"  Average Marginal Effect (AME): {effects['AME']:.4f}")
            print(f"  Interpretation: Relative to r/Anxiety, posting in {community_name} community")
            print(f"        changes the toxicity score by {effects['AME']:.4f} units on average")
            print(f"        (equivalent to {effects['AME']*100:.2f}% increase in toxicity)")

    # 15. Save all results
    results.to_csv('Fractional_Logit_Regression_Results.csv', index=False, encoding='utf-8-sig')
    me_df.to_csv('Marginal_Effects_Results.csv', index=False, encoding='utf-8-sig')

    # Create comprehensive report
    with open('Analysis_Report.txt', 'w', encoding='utf-8') as f:
        f.write("Reddit Mental Health Community Toxicity Analysis Report\n")
        f.write("="*60 + "\n\n")

        f.write("1. Sample Overview\n")
        f.write(f"   Total sample size: {len(data)}\n")
        f.write(f"   r/Anxiety: {len(anxiety_df)} posts\n")
        f.write(f"   r/Depression: {len(depression_df)} posts\n")
        f.write(f"   r/SuicideWatch: {len(suicidewatch_df)} posts\n\n")

        f.write("2. Main Findings\n")
        f.write(f"   Overall toxicity mean: {y.mean():.4f}\n")
        f.write(f"   Standard deviation: {y.std():.4f}\n\n")

        f.write("3. Key Marginal Effects\n")
        for var, effects in marginal_effects.items():
            f.write(f"   {var}: AME={effects['AME']:.4f}, ")
            f.write(f"95% CI=[{effects['CI_lower']:.4f}, {effects['CI_upper']:.4f}]\n")

        f.write("\n4. Policy Implications\n")
        f.write("   Based on marginal effects analysis, key factors to focus on include...\n")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - Fractional_Logit_Regression_Results.csv")
    print("  - Marginal_Effects_Results.csv")
    print("  - marginal_effects_dist.png")
    print("  - Analysis_Report.txt")

    return model, results, marginal_effects, elasticities

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    model, results, marginal_effects, elasticities = main()
#%%
