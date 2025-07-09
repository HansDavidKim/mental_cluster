import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

### t-SNE is a popular dimensionality reduction technique that is particularly well-suited for visualizing high-dimensional data.
from sklearn.manifold import TSNE

def visualize_reduced_logits(data: pd.DataFrame, n_components: int = 2, perplexity: int = 30):
    """
    Visualize the reduced logits using t-SNE.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing 'reduced_logits' column with high-dimensional data.
    n_components (int): Number of dimensions for t-SNE output.
    perplexity (int): Perplexity parameter for t-SNE.
    
    Returns:
    None: Displays a scatter plot of the reduced logits.
    """
    # Extract the reduced logits
    if isinstance(data['reduced_logits'].iloc[0], str):
        data['reduced_logits'] = data['reduced_logits'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))

    reduced_logits = np.vstack(data['reduced_logits'].to_numpy())
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(reduced_logits)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization of Reduced Logits')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    #plt.show()

    ### save the plot as an image file
    plt.savefig('tsne_reduced_logits.png')

def visualize_prob_logits(data: pd.DataFrame):
    """
    Visualize the probability logits in a scatter plot.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing 'prob_logits' column with high-dimensional data.
    
    Returns:
    None: Displays a scatter plot of the probability logits.
    """
    # Extract the probability logits
    if isinstance(data['prob_logits'].iloc[0], str):
        data['prob_logits'] = data['prob_logits'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))

    logits_matrix = np.vstack(data['prob_logits'].to_numpy())
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(logits_matrix[:, 0], logits_matrix[:, 1], alpha=0.5)
    plt.title('Scatter Plot of Probability Logits')
    plt.xlabel('Logit Dimension 1')
    plt.ylabel('Logit Dimension 2')
    plt.grid(True)
    #plt.show()

    ### save the plot as an image file
    plt.savefig('scatter_prob_logits.png')