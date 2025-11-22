import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_distribution(data, title='Data Distribution', ax =None):
    type_counts = pd.Series(data).value_counts()
    df_plot = pd.DataFrame({'Category': type_counts.index.astype(str), 'Count': type_counts.values})

    # Use hue and legend=False to address FutureWarning
    sns.barplot(x='Category', y='Count', hue='Category', data=df_plot, palette='viridis', legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)