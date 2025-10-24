import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)



anxiety_df = load_data('anxiety_2024_features_tfidf_256.csv')
depression_df = load_data('depression_2024_features_tfidf_256.csv')
suicidewatch_df = load_data('suicidewatch_2024_features_tfidf_256.csv')


df = pd.concat([anxiety_df, depression_df, suicidewatch_df], ignore_index=True)


print(df.columns)



def create_edges(df):
    edges = []


    for idx, row in df.iterrows():
        author = row['author']
        subreddit = row['subreddit']


        author_posts = df[(df['author'] == author) & (df['subreddit'] == subreddit)]
        for i in range(len(author_posts) - 1):
            edges.append((author_posts.iloc[i]['post'], author_posts.iloc[i + 1]['post']))

    return edges



edges = create_edges(df)


G = nx.DiGraph()
G.add_edges_from(edges)


degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)


centrality_df = pd.DataFrame({
    'degree': pd.Series(degree_centrality),
    'betweenness': pd.Series(betweenness_centrality),
    'closeness': pd.Series(closeness_centrality)
})


centrality_df.to_csv('network_centrality.csv', index=True)


plt.figure(figsize=(8, 6))
plt.hist(list(degree_centrality.values()), bins=50, color='skyblue', edgecolor='black')
plt.title("Degree Centrality Distribution")
plt.xlabel("Degree Centrality")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('degree_centrality_distribution.png')
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(list(betweenness_centrality.values()), bins=50, color='salmon', edgecolor='black')
plt.title("Betweenness Centrality Distribution")
plt.xlabel("Betweenness Centrality")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('betweenness_centrality_distribution.png')
plt.show()

#%%
