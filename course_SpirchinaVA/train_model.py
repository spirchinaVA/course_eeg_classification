import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import urllib.request
import warnings
warnings.filterwarnings('ignore')

# Машинное обучение
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score, roc_curve)

# Модели
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Сохранение модели
import joblib
import os

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def _print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)


def create_directories():
    dirs = ['data', 'models', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✓ Директории созданы")


def download_and_load_data():
    _print_header("1. ЗАГРУЗКА ДАННЫХ")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
    data_path = 'data/eeg_eye_state.arff'
    
    if not os.path.exists(data_path):
        print("Загрузка датасета из UCI Repository...")
        urllib.request.urlretrieve(url, data_path)
        print("✓ Датасет загружен")
    else:
        print("✓ Датасет уже существует")
    
    # Чтение ARFF файла
    print("Чтение данных...")
    data, meta = arff.loadarff(data_path)
    df = pd.DataFrame(data)
    
    # Конвертация типов
    df['eyeDetection'] = df['eyeDetection'].astype(int)
    
    print(f"✓ Данные загружены: {df.shape[0]} наблюдений, {df.shape[1]} признаков")
    print(f"\nПризнаки: 14 каналов ЭЭГ + целевая переменная (eyeDetection)")
    print(f"  - 0 = глаза открыты")
    print(f"  - 1 = глаза закрыты")
    
    return df


def exploratory_data_analysis(df):
    _print_header("2. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")
    
    # Базовая информация
    print(f"\nРазмер датасета: {df.shape}")
    print(f"Пропущенные значения: {df.isnull().sum().sum()}")
    
    # Распределение классов
    print(f"\nРаспределение целевой переменной:")
    class_dist = df['eyeDetection'].value_counts()
    print(f"  Класс 0 (открыты): {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%)")
    print(f"  Класс 1 (закрыты): {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)")
    
    # Визуализация распределения классов
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    class_dist.plot(kind='bar', ax=ax[0], color=['skyblue', 'lightcoral'])
    ax[0].set_title('Распределение классов', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Состояние глаз')
    ax[0].set_xticklabels(['Открыты (0)', 'Закрыты (1)'], rotation=0)
    ax[0].set_ylabel('Количество')
    ax[0].grid(True, alpha=0.3)
    
    class_dist.plot(kind='pie', ax=ax[1], autopct='%1.1f%%', 
                     colors=['skyblue', 'lightcoral'])
    ax[1].set_title('Процентное соотношение', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Визуализация сохранена: results/class_distribution.png")
    plt.close()
    
    # Корреляционная матрица
    feature_cols = [col for col in df.columns if col != 'eyeDetection']
    plt.figure(figsize=(14, 12))
    correlation_matrix = df[feature_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Корреляционная матрица ЭЭГ каналов', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Корреляционная матрица сохранена: results/correlation_matrix.png")
    plt.close()
    
    return feature_cols


def prepare_data(df, feature_cols):
    _print_header("3. ПОДГОТОВКА ДАННЫХ")
    
    # Разделение на признаки и целевую переменную
    X = df[feature_cols].values
    y = df['eyeDetection'].values
    
    print(f"Размерность признаков X: {X.shape}")
    print(f"Размерность целевой переменной y: {y.shape}")
    
    # Разделение на train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nРазделение данных:")
    print(f"  Train: {X_train.shape[0]} наблюдений ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} наблюдений ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Данные нормализованы (StandardScaler)")
    
    # Сохранение scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    print("✓ Scaler и названия признаков сохранены")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def prepare_data_split(X_train, X_test, y_train, y_test):
    _print_header("ПОДГОТОВКА ДАННЫХ (НОРМАЛИЗАЦИЯ)")
    
    print(f"\nПолучены уже разделенные данные:")
    print(f"  Train: {X_train.shape[0]} наблюдений")
    print(f"  Test: {X_test.shape[0]} наблюдений")
    
    # Нормализация
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Данные нормализованы (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}")
    print(f"Обучение модели: {name}")
    print(f"{'='*60}")
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\nРезультаты:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Открыты (0)', 'Закрыты (1)']))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'predictions_proba': y_pred_proba
    }


def train_models(X_train, X_test, y_train, y_test):
    _print_header("4. ОБУЧЕНИЕ МОДЕЛЕЙ")
    
    results = {}
    
    # 1. Logistic Regression
    print("\n[1/4] Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    results['Logistic Regression'] = evaluate_model(
        'Logistic Regression', lr_model, X_train, X_test, y_train, y_test
    )
    
    # 2. Random Forest
    print("\n[2/4] Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    results['Random Forest'] = evaluate_model(
        'Random Forest', rf_model, X_train, X_test, y_train, y_test
    )
    
    # 3. Gradient Boosting
    print("\n[3/4] Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    results['Gradient Boosting'] = evaluate_model(
        'Gradient Boosting', gb_model, X_train, X_test, y_train, y_test
    )
    
    # 4. XGBoost
    print("\n[4/4] XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, 
                              eval_metric='logloss', use_label_encoder=False)
    results['XGBoost'] = evaluate_model(
        'XGBoost', xgb_model, X_train, X_test, y_train, y_test
    )
    
    return results


def compare_models(results):
    _print_header("5. СРАВНЕНИЕ МОДЕЛЕЙ")
    
    # Таблица сравнения
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'F1-Score': result['f1_score'],
            'ROC-AUC': result['roc_auc'],
            'CV Mean': result['cv_mean'],
            'CV Std': result['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    comparison_df.plot(x='Model', y='Accuracy', kind='bar', ax=axes[0], 
                       legend=False, color='skyblue')
    axes[0].set_title('Accuracy по моделям', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Модель')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # F1-Score
    comparison_df.plot(x='Model', y='F1-Score', kind='bar', ax=axes[1], 
                       legend=False, color='lightcoral')
    axes[1].set_title('F1-Score по моделям', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_xlabel('Модель')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # ROC-AUC
    comparison_df.plot(x='Model', y='ROC-AUC', kind='bar', ax=axes[2], 
                       legend=False, color='lightgreen')
    axes[2].set_title('ROC-AUC по моделям', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('ROC-AUC')
    axes[2].set_xlabel('Модель')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Визуализация сохранена: results/model_comparison.png")
    plt.close()
    
    return comparison_df


def save_final_model(results, comparison_df, scaler=None, feature_cols=None):
    _print_header("6. СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    
    # Определяем лучшую модель
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = results[best_model_name]
    final_model = best_result['model']
    
    print(f"\nЛучшая модель: {best_model_name}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1-Score: {best_result['f1_score']:.4f}")
    print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
    
    # Сохранение модели
    model_path = 'models/final_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"\n✓ Модель сохранена: {model_path}")
    
    # Сохранение scaler и feature_names если переданы
    if scaler is not None:
        joblib.dump(scaler, 'models/scaler.pkl')
        print(f"✓ Scaler сохранен: models/scaler.pkl")
    
    if feature_cols is not None:
        joblib.dump(feature_cols, 'models/feature_names.pkl')
        print(f"✓ Названия признаков сохранены: models/feature_names.pkl")
    
    # Сохранение информации о модели
    model_info = {
        'model_name': best_model_name,
        'accuracy': best_result['accuracy'],
        'f1_score': best_result['f1_score'],
        'roc_auc': best_result['roc_auc'],
        'cv_mean': best_result['cv_mean'],
        'cv_std': best_result['cv_std'],
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_info, 'models/model_info.pkl')
    print(f"✓ Информация о модели сохранена: models/model_info.pkl")
    
    return best_model_name, best_result
