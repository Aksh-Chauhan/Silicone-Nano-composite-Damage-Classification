import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# --- 1. CONFIGURATION & DATA MAP ---
DATA_MAP = {
    'testo_c0_50': (0, 50), 'testo_65_virgin': (0, 65), 'testo_80_virgin': (0, 80), 'testo_100_virgin': (0, 95),
    'testo_c10_50': (10, 50), 'testo_65_c_10': (10, 65), 'testo_80_c_10': (10, 80), 'testo_100_c_10': (10, 95),
    'testo_c15_50': (15, 50), 'testo_65_c_15': (15, 65), 'testo_80_c_15': (15, 80), 'testo_100_c_15': (15, 95)
}

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# --- 2. FEATURE EXTRACTION ---
def extract_frame_features(grid, first_grid, time_seconds):
    f_min, f_max = np.nanmin(grid), np.nanmax(grid)
    f_mean = np.nanmean(grid)
    
    ambient = 32
    temp_now_diff = max(f_mean - ambient, 1e-6)
    temp_start_diff = max(np.nanmean(first_grid) - ambient, 1e-6)
    k_instant = -np.log(temp_now_diff / temp_start_diff) / (time_seconds + 1e-6)

    gx, gy = np.gradient(grid)
    t_grad = np.nanmean(np.sqrt(gx**2 + gy**2))
    
    raw_contrast = max(f_max - f_min, 1e-6)
    norm = ((grid - f_min) / raw_contrast * 255).astype(np.uint8)
    glcm = graycomatrix(norm, [1], [0], 256, symmetric=True, normed=True)

    return {
        'Mean_T': f_mean,
        'Std': np.nanstd(grid),
        'Skew': skew(grid.flatten(), nan_policy='omit'),
        'K_instant': k_instant,
        'Gradient': t_grad,
        'GLCM_con': graycoprops(glcm, 'contrast')[0, 0],
        'GLCM_hom': graycoprops(glcm, 'homogeneity')[0, 0]
    }

# --- 3. DATA LOADING ---
all_data = []
for folder, (cell_pct, start_temp) in DATA_MAP.items():
    if os.path.exists(folder):
        print(f"Processing Folder: {folder}...")
        files = natural_sort([f for f in os.listdir(folder) if f.endswith('.xlsx')])
        first_grid = pd.read_excel(os.path.join(folder, files[0]), header=None).values
        
        for i, f in enumerate(files):
            try:
                orig_grid = pd.read_excel(os.path.join(folder, f), header=None).values
                time_sec = i * 30 
                features = extract_frame_features(orig_grid, first_grid, time_sec)
                features.update({'Filler_Pct': cell_pct, 'Start_Temp': start_temp})
                all_data.append(features)
            except: continue

df = pd.DataFrame(all_data)

# --- 4. CLUSTERING & LABEL MAPPING ---
print("\nClustering data into Damage Levels...")
X_phys = df.drop(['Filler_Pct', 'Start_Temp'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_phys)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(X_pca)

cluster_order = df.groupby('Cluster_ID')['Gradient'].mean().sort_values().index
label_mapping = {cluster_order[0]: 'Healthy', cluster_order[1]: 'Moderate Damaged', cluster_order[2]: 'Severely Damaged'}
df['Damage_Level'] = df['Cluster_ID'].map(label_mapping)

# --- 5. MODEL TRAINING & CROSS-VALIDATION ---
X_model = X_scaled
y_model = df['Damage_Level']

# A. Setup Models
rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# B. Case 1: Cross-Validation (Stability Check)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = cross_val_score(rf, X_model, y_model, cv=skf)
svm_cv = cross_val_score(svm, X_model, y_model, cv=skf)

# C. Case 2: Train-Test Split (Detailed Metrics)
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, stratify=y_model, random_state=42)

models = {'Random Forest': rf, 'SVM (RBF)': svm}
split_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=list(label_mapping.values()))
    
    split_results[name] = {
        'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'CM': cm
    }

# --- 6. OUTPUT RESULTS ---
print("\n" + "="*85)
print("CASE 1: CROSS-VALIDATION RESULTS (Whole Dataset Stability)")
print("-" * 85)
print(f"{'Random Forest Mean Accuracy':<30} : {rf_cv.mean()*100:.2f}% (Std: {rf_cv.std()*100:.2f}%)")
print(f"{'SVM (RBF) Mean Accuracy':<30} : {svm_cv.mean()*100:.2f}% (Std: {svm_cv.std()*100:.2f}%)")

print("\n" + "="*85)
print("CASE 2: TEST SPLIT METRICS (20% Hold-out Evaluation)")
print("-" * 85)
print(f"{'Model':<20} | {'Acc':<8} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<8}")
print("-" * 85)
for name, m in split_results.items():
    print(f"{name:<20} | {m['Accuracy']*100:.2f}% | {m['Precision']:.4f}   | {m['Recall']:.4f} | {m['F1 Score']:.4f}")
print("="*85)

# --- 7. VISUALIZATIONS ---
# A. Metrics Bar Chart with Values
metrics_only = pd.DataFrame(split_results).T.drop(columns='CM')
ax = metrics_only.plot(kind='bar', figsize=(12, 6), rot=0, color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'])
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.ylim(0, 1.2) # Extra room for labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)
plt.legend(loc='upper right', ncol=4)
plt.show()

# B. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (name, m) in enumerate(split_results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=m['CM'], display_labels=list(label_mapping.values()))
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f'Confusion Matrix: {name}')
plt.tight_layout()
plt.show()

# C. Correlation Matrix
plt.figure(figsize=(10, 8))
# Drop columns that aren't training features to keep the matrix clean
corr_features = df.drop(['Damage_Level', 'Cluster_ID', 'Filler_Pct', 'Start_Temp'], axis=1).corr()
sns.heatmap(corr_features, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix (Thermal & Texture Features)")
plt.show()

# D. CV Stability Boxplot (Compatible with Matplotlib 3.9+)
plt.figure(figsize=(8, 5))
plt.boxplot([rf_cv, svm_cv], tick_labels=['Random Forest', 'SVM'])
plt.ylabel('Accuracy')
plt.title('Accuracy Stability (5-Fold Cross-Validation)')
plt.grid(axis='y', alpha=0.3)
plt.show()

# PCA Cluster Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Damage_Level'], palette='viridis')
plt.title("Damage Levels discovered via PCA & K-Means")
plt.show()
