"""
NutriPredict - Exploratory Data Analysis (EDA)
Script untuk analisis dataset sebelum training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("NUTRIPREDICT - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ==================== 1. LOAD DATA ====================
print("\n📊 STEP 1: LOAD DATA")
print("-" * 70)

try:
    df = pd.read_csv('child_data_rev.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Error: stunting_data.csv tidak ditemukan!")
    print("   Download dataset dari: https://www.kaggle.com/datasets/m282dsx1313reyhan/indonesian-children-medical-and-food-nutrition")
    exit()

# ==================== 2. BASIC INFO ====================
print("\n📋 STEP 2: BASIC INFORMATION")
print("-" * 70)

print("\n🔍 Dataset Info:")
print(df.info())

print("\n📊 First 5 Rows:")
print(df.head())

print("\n📈 Statistical Summary:")
print(df.describe())

# ==================== 3. MISSING VALUES ====================
print("\n\n❓ STEP 3: MISSING VALUES ANALYSIS")
print("-" * 70)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print("\n⚠️ Columns with Missing Values:")
    print(missing_df.to_string(index=False))
    
    # Visualize missing values
    if len(missing_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.barh(missing_df['Column'], missing_df['Percentage'], color='coral')
        plt.xlabel('Percentage Missing (%)')
        plt.title('Missing Values by Column')
        plt.tight_layout()
        plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
        print("\n💾 Saved: missing_values.png")
else:
    print("\n✅ No missing values found!")

# ==================== 4. DATA TYPES ====================
print("\n\n📝 STEP 4: DATA TYPES")
print("-" * 70)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n📝 Categorical Columns ({len(categorical_cols)}):")
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"   - {col}: {unique_count} unique values")

print(f"\n🔢 Numerical Columns ({len(numerical_cols)}):")
for col in numerical_cols:
    print(f"   - {col}: range [{df[col].min():.2f}, {df[col].max():.2f}]")

# ==================== 5. TARGET VARIABLE ANALYSIS ====================
print("\n\n🎯 STEP 5: TARGET VARIABLE ANALYSIS")
print("-" * 70)

# Identify target column (adjust if needed)
target_candidates = ['Stunting_Status', 'Status', 'Target', 'Label', 'Class']
target_col = None

for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    # Try to find by unique values (3 classes for stunting)
    for col in df.columns:
        if df[col].nunique() == 3:
            print(f"\n🔍 Potential target column found: {col}")
            print(f"   Unique values: {df[col].unique()}")
            confirm = input("   Is this the target column? (y/n): ")
            if confirm.lower() == 'y':
                target_col = col
                break

if target_col:
    print(f"\n✅ Target Column: {target_col}")
    
    # Class distribution
    class_dist = df[target_col].value_counts().sort_index()
    class_pct = (class_dist / len(df)) * 100
    
    print("\n📊 Class Distribution:")
    for cls, count in class_dist.items():
        pct = (count / len(df)) * 100
        print(f"   {cls}: {count} ({pct:.2f}%)")
    
    # Check imbalance
    max_pct = class_pct.max()
    min_pct = class_pct.min()
    imbalance_ratio = max_pct / min_pct
    
    print(f"\n⚖️ Imbalance Analysis:")
    print(f"   Imbalance Ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2:
        print(f"   ⚠️ IMBALANCED! Max class is {imbalance_ratio:.2f}x larger than min class")
        print(f"   📌 Recommendation: Use SMOTE or class weighting")
    else:
        print(f"   ✅ Relatively balanced dataset")
    
    # Visualize class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    class_dist.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_title(f'Class Distribution - {target_col}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#f39c12', '#e74c3c'], startangle=90)
    axes[1].set_title(f'Class Proportion - {target_col}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    print("\n💾 Saved: target_distribution.png")
    
else:
    print("\n❌ Target column not found! Please specify manually.")

# ==================== 6. CATEGORICAL FEATURES ANALYSIS ====================
print("\n\n📝 STEP 6: CATEGORICAL FEATURES ANALYSIS")
print("-" * 70)

if len(categorical_cols) > 0 and target_col:
    cat_cols_to_plot = [col for col in categorical_cols if col != target_col][:6]  # Max 6 plots
    
    if len(cat_cols_to_plot) > 0:
        n_cols = 3
        n_rows = (len(cat_cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(cat_cols_to_plot):
            if df[col].nunique() <= 10:  # Only plot if reasonable number of categories
                pd.crosstab(df[col], df[target_col]).plot(
                    kind='bar',
                    ax=axes[idx],
                    color=['#2ecc71', '#f39c12', '#e74c3c']
                )
                axes[idx].set_title(f'{col} vs {target_col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].legend(title=target_col)
        
        # Hide empty subplots
        for idx in range(len(cat_cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        print("\n💾 Saved: categorical_analysis.png")

# ==================== 7. NUMERICAL FEATURES ANALYSIS ====================
print("\n\n🔢 STEP 7: NUMERICAL FEATURES ANALYSIS")
print("-" * 70)

if len(numerical_cols) > 0:
    # Remove target if it's numerical
    num_cols_to_plot = [col for col in numerical_cols if col != target_col][:6]
    
    if len(num_cols_to_plot) > 0:
        # Distribution plots
        n_cols = 3
        n_rows = (len(num_cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(num_cols_to_plot):
            df[col].hist(bins=30, ax=axes[idx], color='skyblue', edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        # Hide empty subplots
        for idx in range(len(num_cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
        print("\n💾 Saved: numerical_distributions.png")
        
        # Boxplots by target
        if target_col:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for idx, col in enumerate(num_cols_to_plot):
                df.boxplot(column=col, by=target_col, ax=axes[idx])
                axes[idx].set_title(f'{col} by {target_col}')
                axes[idx].set_xlabel(target_col)
                axes[idx].set_ylabel(col)
            
            # Hide empty subplots
            for idx in range(len(num_cols_to_plot), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            plt.savefig('numerical_by_target.png', dpi=300, bbox_inches='tight')
            print("💾 Saved: numerical_by_target.png")

# ==================== 8. CORRELATION ANALYSIS ====================
print("\n\n🔗 STEP 8: CORRELATION ANALYSIS")
print("-" * 70)

if len(numerical_cols) > 2:
    # Compute correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    print("\n📊 Top 10 Positive Correlations:")
    # Get upper triangle
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlations = upper_tri.stack().sort_values(ascending=False)
    print(correlations.head(10))
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n💾 Saved: correlation_matrix.png")

# ==================== 9. OUTLIER DETECTION ====================
print("\n\n📉 STEP 9: OUTLIER DETECTION")
print("-" * 70)

if len(numerical_cols) > 0:
    num_cols_to_check = [col for col in numerical_cols if col != target_col][:6]
    
    outlier_summary = []
    
    for col in num_cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_summary.append({
            'Column': col,
            'Outlier Count': outlier_count,
            'Percentage': f'{outlier_pct:.2f}%',
            'Lower Bound': f'{lower_bound:.2f}',
            'Upper Bound': f'{upper_bound:.2f}'
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print("\n📊 Outlier Summary (IQR Method):")
    print(outlier_df.to_string(index=False))

# ==================== 10. SUMMARY & RECOMMENDATIONS ====================
print("\n\n" + "=" * 70)
print("📋 SUMMARY & RECOMMENDATIONS")
print("=" * 70)

print("\n✅ Dataset Overview:")
print(f"   - Total Samples: {len(df)}")
print(f"   - Total Features: {len(df.columns)}")
print(f"   - Categorical Features: {len(categorical_cols)}")
print(f"   - Numerical Features: {len(numerical_cols)}")

if len(missing_df) > 0:
    print(f"\n⚠️ Data Quality Issues:")
    print(f"   - {len(missing_df)} columns with missing values")
    print(f"   - Total missing: {missing['Missing Count'].sum()} ({missing_pct.sum():.2f}%)")
    print(f"   📌 Recommendation: Handle missing values (drop or impute)")

if target_col and imbalance_ratio > 2:
    print(f"\n⚠️ Class Imbalance Detected:")
    print(f"   - Imbalance ratio: {imbalance_ratio:.2f}x")
    print(f"   📌 Recommendation: Use SMOTE + Class Weighting")

print("\n🎯 Next Steps:")
print("   1. ✅ Handle missing values")
print("   2. ✅ Encode categorical variables")
print("   3. ✅ Scale numerical features")
print("   4. ✅ Handle class imbalance (SMOTE)")
print("   5. ✅ Train ANN/MLP model")
print("   6. ✅ Evaluate with Confusion Matrix")

print("\n💡 Ready to train? Run:")
print("   python train_model.py")

print("\n" + "=" * 70)
print("EDA COMPLETED! 🎉")
print("=" * 70)