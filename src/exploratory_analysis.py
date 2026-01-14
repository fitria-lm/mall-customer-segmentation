"""
Module untuk exploratory data analysis (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import dari utils
from .utils import log_message, validate_dataframe, check_missing_values

def describe_dataset(df, include_all=False):
    """
    Generate comprehensive dataset description
    
    Args:
        df: DataFrame untuk dideskripsikan
        include_all: Include semua statistik (termasuk yang jarang)
    
    Returns:
        dict: Dictionary berisi statistik dataset
    """
    log_message("Generating dataset description", "INFO")
    
    # Validasi dataframe terlebih dahulu
    validate_dataframe(df)
    
    description = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": dict(df.dtypes),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    # Statistik numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        description["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        # Additional stats
        for col in numeric_cols:
            description[f"{col}_skewness"] = df[col].skew()
            description[f"{col}_kurtosis"] = df[col].kurtosis()
    
    # Statistik kategorikal
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        description["categorical_stats"] = {}
        for col in categorical_cols:
            description["categorical_stats"][col] = {
                "unique_count": df[col].nunique(),
                "unique_values": df[col].unique().tolist()[:10],  # First 10
                "top_value": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "top_count": (df[col] == df[col].mode().iloc[0]).sum() if not df[col].mode().empty else 0
            }
    
    # Missing values analysis
    missing_info = check_missing_values(df)
    description["missing_values"] = missing_info
    
    log_message("Dataset description generated", "SUCCESS")
    return description

def plot_distributions(df, columns=None, figsize=(15, 10), save_path=None):
    """
    Plot distribusi untuk kolom-kolom tertentu
    
    Args:
        df: DataFrame
        columns: List kolom untuk diplot (None = semua numeric)
        figsize: Ukuran figure
        save_path: Path untuk menyimpan plot
    """
    log_message("Plotting feature distributions", "INFO")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes jika perlu
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            ax = axes[idx]
            
            # Histogram dengan KDE
            sns.histplot(df[col], kde=True, ax=ax, bins=30, color='skyblue', edgecolor='black')
            
            # Tambah garis mean dan median
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Median: {median_val:.2f}')
            
            # Hitung skewness
            skewness = df[col].skew()
            skewness_text = f"Skewness: {skewness:.2f}\n"
            if abs(skewness) < 0.5:
                skewness_text += "(Nearly symmetric)"
            elif abs(skewness) < 1:
                skewness_text += "(Moderately skewed)"
            else:
                skewness_text += "(Highly skewed)"
            
            # Anotasi statistik
            stats_text = f"Count: {df[col].count():.0f}\nMean: {mean_val:.2f}\nStd: {df[col].std():.2f}\n{skewness_text}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Sembunyikan axes yang tidak terpakai
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"Distribution plots saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, axes

def plot_correlation_heatmap(df, method='pearson', figsize=(12, 10), save_path=None):
    """
    Plot correlation heatmap untuk kolom numerik
    
    Args:
        df: DataFrame
        method: Metode korelasi ('pearson', 'spearman', 'kendall')
        figsize: Ukuran figure
        save_path: Path untuk menyimpan plot
    """
    log_message(f"Plotting correlation heatmap (method: {method})", "INFO")
    
    # Hanya kolom numerik
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        log_message("Need at least 2 numeric columns for correlation heatmap", "WARNING")
        return None
    
    # Hitung matriks korelasi
    corr_matrix = numeric_df.corr(method=method)
    
    # Buat mask untuk segitiga atas
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap dengan diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr_matrix, 
                mask=mask, 
                cmap=cmap,
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                annot=True,
                fmt=".2f",
                annot_kws={"size": 9},
                ax=ax)
    
    ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"Correlation heatmap saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, ax, corr_matrix

def plot_boxplots_by_category(df, numeric_col, categorical_col, figsize=(12, 6), save_path=None):
    """
    Plot boxplot untuk kolom numerik dikelompokkan oleh kolom kategorikal
    
    Args:
        df: DataFrame
        numeric_col: Kolom numerik
        categorical_col: Kolom kategorikal
        figsize: Ukuran figure
        save_path: Path untuk menyimpan plot
    """
    log_message(f"Plotting boxplot of {numeric_col} by {categorical_col}", "INFO")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Boxplot
    df.boxplot(column=numeric_col, by=categorical_col, ax=ax1, grid=False)
    ax1.set_title(f'{numeric_col} by {categorical_col}', fontsize=14, fontweight='bold')
    ax1.set_xlabel(categorical_col, fontsize=12)
    ax1.set_ylabel(numeric_col, fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot untuk distribusi yang lebih detail
    categories = df[categorical_col].unique()
    colors = plt.cm.Set2(np.arange(len(categories)) / len(categories))
    
    for i, category in enumerate(categories):
        data = df[df[categorical_col] == category][numeric_col]
        parts = ax2.violinplot(data, positions=[i], showmeans=True, showmedians=True)
        
        # Warna violin plot
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
    
    ax2.set_title(f'Distribution of {numeric_col} by {categorical_col}', fontsize=14, fontweight='bold')
    ax2.set_xlabel(categorical_col, fontsize=12)
    ax2.set_ylabel(numeric_col, fontsize=12)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('', fontsize=0)  # Hapus automatic suptitle
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"Boxplot saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, (ax1, ax2)

def analyze_correlations(df, method='pearson', threshold=0.5):
    """
    Analisis korelasi mendalam antara variabel numerik
    
    Args:
        df: DataFrame
        method: Metode korelasi
        threshold: Threshold untuk korelasi kuat
    
    Returns:
        dict: Hasil analisis korelasi
    """
    log_message(f"Analyzing correlations (method: {method})", "INFO")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return {"error": "Need at least 2 numeric columns"}
    
    # Hitung matriks korelasi
    corr_matrix = numeric_df.corr(method=method)
    
    # Identifikasi korelasi kuat
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                strength = "Strong positive" if corr_value > 0 else "Strong negative"
                strong_correlations.append({
                    "variable1": col1,
                    "variable2": col2,
                    "correlation": corr_value,
                    "strength": strength,
                    "abs_correlation": abs(corr_value)
                })
    
    # Urutkan berdasarkan absolut korelasi
    strong_correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
    
    # Hitung p-values untuk korelasi signifikan
    p_values = {}
    for i in range(len(numeric_df.columns)):
        for j in range(i+1, len(numeric_df.columns)):
            col1 = numeric_df.columns[i]
            col2 = numeric_df.columns[j]
            
            if method == 'pearson':
                corr, p_value = stats.pearsonr(numeric_df[col1].dropna(), 
                                               numeric_df[col2].dropna())
            elif method == 'spearman':
                corr, p_value = stats.spearmanr(numeric_df[col1].dropna(), 
                                                numeric_df[col2].dropna())
            else:
                p_value = np.nan
            
            p_values[f"{col1}_{col2}"] = p_value
    
    results = {
        "correlation_matrix": corr_matrix,
        "strong_correlations": strong_correlations,
        "p_values": p_values,
        "method": method,
        "threshold": threshold
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"Threshold for strong correlation: {threshold}")
    print(f"\nStrong correlations found: {len(strong_correlations)}")
    
    for corr in strong_correlations[:5]:  # Tampilkan 5 teratas
        print(f"\n  {corr['variable1']} vs {corr['variable2']}:")
        print(f"    Correlation: {corr['correlation']:.3f} ({corr['strength']})")
    
    log_message("Correlation analysis completed", "SUCCESS")
    
    return results

def generate_eda_report(df, save_dir='reports/eda'):
    """
    Generate comprehensive EDA report dengan visualisasi
    
    Args:
        df: DataFrame
        save_dir: Directory untuk menyimpan report
    """
    log_message("Generating comprehensive EDA report", "INFO")
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Dataset description
    description = describe_dataset(df)
    
    # 2. Plot distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plot_distributions(
            df, 
            columns=numeric_cols,
            save_path=f"{save_dir}/distributions.png"
        )
    
    # 3. Correlation analysis
    if len(numeric_cols) >= 2:
        plot_correlation_heatmap(
            df,
            save_path=f"{save_dir}/correlation_heatmap.png"
        )
        
        # Detailed correlation analysis
        corr_results = analyze_correlations(df)
    
    # 4. Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols:
            for num_col in numeric_cols[:2]:  # Hanya 2 pertama
                try:
                    plot_boxplots_by_category(
                        df,
                        numeric_col=num_col,
                        categorical_col=cat_col,
                        save_path=f"{save_dir}/boxplot_{num_col}_by_{cat_col}.png"
                    )
                except Exception as e:
                    log_message(f"Could not plot {num_col} by {cat_col}: {str(e)}", "WARNING")
    
    # 5. Missing values visualization
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
        ax.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/missing_values_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Save text report
    report_path = f"{save_dir}/eda_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset Shape: {description['shape']}\n")
        f.write(f"Memory Usage: {description['memory_usage']:.2f} MB\n\n")
        
        f.write("Columns and Data Types:\n")
        for col, dtype in description['dtypes'].items():
            f.write(f"  {col}: {dtype}\n")
        
        f.write("\nMissing Values Summary:\n")
        f.write(f"  Total missing cells: {description['missing_values']['missing_cells']}\n")
        f.write(f"  Percentage: {description['missing_values']['missing_percentage']:.2f}%\n")
        f.write(f"  Status: {description['missing_values']['status']}\n")
    
    log_message(f"EDA report saved to: {save_dir}", "SUCCESS")
    
    return {
        "description": description,
        "save_dir": save_dir,
        "report_path": report_path
    }

# Test fungsi
if __name__ == "__main__":
    # Buat test data
    test_df = pd.DataFrame({
        'Age': np.random.randint(18, 70, 100),
        'Income': np.random.normal(50, 15, 100),
        'Spending': np.random.normal(50, 20, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Testing exploratory analysis functions...")
    
    # Test describe_dataset
    desc = describe_dataset(test_df)
    print(f"Dataset shape: {desc['shape']}")
    
    # Test plot_distributions
    plot_distributions(test_df, columns=['Age', 'Income', 'Spending'])
    
    # Test correlation analysis
    results = analyze_correlations(test_df[['Age', 'Income', 'Spending']])
    
    print("✅ All tests passed!")