import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def _print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)


def load_model_and_scaler():
    _print_header("ТЕСТИРОВАНИЕ МОДЕЛИ КЛАССИФИКАЦИИ СОСТОЯНИЯ ГЛАЗ")
    print()
    
    # Проверка наличия файлов
    required_files = [
        'models/final_model.pkl',
        'models/scaler.pkl',
        'models/feature_names.pkl',
        'models/model_info.pkl'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ ОШИБКА: Файл {file_path} не найден!")
            print("Сначала запустите train_model.py для обучения модели.")
            exit(1)
    
    # Загрузка модели и данных
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    model_info = joblib.load('models/model_info.pkl')
    
    print("✓ Модель и компоненты успешно загружены")
    print(f"\nИнформация о модели:")
    print(f"  Название: {model_info['model_name']}")
    print(f"  Accuracy: {model_info['accuracy']:.4f}")
    print(f"  F1-Score: {model_info['f1_score']:.4f}")
    print(f"  ROC-AUC: {model_info['roc_auc']:.4f}")
    print(f"  Дата обучения: {model_info['training_date']}")
    print(f"  Количество признаков: {len(feature_names)}")
    
    return model, scaler, feature_names


def load_test_data():
    from scipy.io import arff
    
    data_path = 'data/eeg_eye_state.arff'
    
    if not os.path.exists(data_path):
        print("\n❌ ОШИБКА: Файл данных не найден!")
        print("Сначала запустите train_model.py для загрузки данных.")
        return None, None
    
    # Загрузка данных
    data, meta = arff.loadarff(data_path)
    df = pd.DataFrame(data)
    df['eyeDetection'] = df['eyeDetection'].astype(int)
    
    # Разделение на признаки и метки
    feature_cols = [col for col in df.columns if col != 'eyeDetection']
    X = df[feature_cols].values
    y = df['eyeDetection'].values
    
    return X, y


def test_on_dataset(model, scaler, X_test, y_test):
    _print_header("ТЕСТИРОВАНИЕ НА ДАТАСЕТЕ")
    
    # Выбираем последние 20% для тестирования 
    test_size = int(len(X_test) * 0.2)
    X_test_subset = X_test[-test_size:]
    y_test_subset = y_test[-test_size:]
    
    print(f"\nРазмер тестовой выборки: {len(X_test_subset)} наблюдений")
    
    # Нормализация и предсказание
    X_test_scaled = scaler.transform(X_test_subset)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Метрики
    accuracy = accuracy_score(y_test_subset, y_pred)
    
    print(f"\nРезультаты:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_subset, y_pred, 
                                target_names=['Открыты (0)', 'Закрыты (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_subset, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Открыты', 'Закрыты'],
                yticklabels=['Открыты', 'Закрыты'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion Matrix сохранена: results/test_confusion_matrix.png")
    plt.close()
    
    return y_pred, y_pred_proba


def test_on_test_set(model, scaler, X_test, y_test):
    _print_header("ТЕСТИРОВАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ")
    
    print(f"\nРазмер тестовой выборки: {len(X_test)} наблюдений")
    
    # Нормализация и предсказание
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nРезультаты:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Открыты (0)', 'Закрыты (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nConfusion Matrix:")
    print(f"                 Предсказано: 0  |  Предсказано: 1")
    print(f"Истинно: 0            {cm[0,0]:>5}     |       {cm[0,1]:>5}")
    print(f"Истинно: 1            {cm[1,0]:>5}     |       {cm[1,1]:>5}")
    
    # Визуализация Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Открыты', 'Закрыты'],
                yticklabels=['Открыты', 'Закрыты'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion Matrix сохранена: results/test_confusion_matrix.png")
    plt.close()
    
    return y_pred, y_pred_proba, cm


def show_model_info(model_info, feature_names):
    _print_header("ИНФОРМАЦИЯ О МОДЕЛИ")
    
    print(f"\nНазвание модели: {model_info['model_name']}")
    print(f"Дата обучения: {model_info['training_date']}")
    
    print(f"\nМетрики на тестовой выборке:")
    print(f"  Accuracy:  {model_info['accuracy']:.4f} ({model_info['accuracy']*100:.2f}%)")
    print(f"  F1-Score:  {model_info['f1_score']:.4f}")
    print(f"  ROC-AUC:   {model_info['roc_auc']:.4f}")
    print(f"  CV Score:  {model_info['cv_mean']:.2%} ± {model_info['cv_std']:.2%}")
    
    print(f"\nДатасет: EEG Eye State (UCI ML Repository)")
    print(f"Количество признаков: {len(feature_names)} каналов ЭЭГ")
    print(f"Классы: 0 = глаза открыты, 1 = глаза закрыты")


def test_random_samples(model, scaler, X, y, n_samples=10):
    _print_header(f"ТЕСТИРОВАНИЕ НА {n_samples} СЛУЧАЙНЫХ ПРИМЕРАХ")
    
    # Выбор случайных индексов
    np.random.seed(42)
    random_indices = np.random.choice(len(X), n_samples, replace=False)
    
    X_samples = X[random_indices]
    y_samples = y[random_indices]
    
    # Нормализация и предсказание
    X_samples_scaled = scaler.transform(X_samples)
    y_pred = model.predict(X_samples_scaled)
    y_pred_proba = model.predict_proba(X_samples_scaled)
    
    # Вывод результатов
    print(f"\n{'№':<4} {'Истинный класс':<20} {'Предсказание':<20} {'Вероятность':<15} {'Результат':<10}")
    print("-"*80)
    
    class_names = {0: 'Открыты', 1: 'Закрыты'}
    correct = 0
    
    for i in range(n_samples):
        true_label = y_samples[i]
        pred_label = y_pred[i]
        prob = y_pred_proba[i][pred_label]
        is_correct = "✓" if true_label == pred_label else "✗"
        
        if true_label == pred_label:
            correct += 1
        
        print(f"{i+1:<4} {class_names[true_label]:<20} {class_names[pred_label]:<20} "
              f"{prob:.4f} ({prob*100:.1f}%){' '*2} {is_correct:<10}")
    
    accuracy = correct / n_samples
    print("-"*80)
    print(f"Точность на случайных примерах: {correct}/{n_samples} ({accuracy*100:.1f}%)")


def test_manual_input(model, scaler, feature_names):
    _print_header("ТЕСТИРОВАНИЕ С РУЧНЫМ ВВОДОМ")
    
    print(f"\nДля предсказания необходимо ввести {len(feature_names)} значений")
    print("(по одному для каждого ЭЭГ канала)")
    print("\nПризнаки:", ", ".join(feature_names))
    print("\nВведите 'skip' для пропуска этого шага")
    
    user_input = input("\nХотите ввести данные вручную? (yes/skip): ").strip().lower()
    
    if user_input == 'skip' or user_input == 'n' or user_input == 'no':
        print("Пропуск ручного ввода.")
        return
    
    print("\nВведите значения для каждого признака:")
    values = []
    
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"  {feature}: "))
                values.append(value)
                break
            except ValueError:
                print("    ❌ Ошибка: введите числовое значение")
    
    # Создание массива и предсказание
    X_manual = np.array(values).reshape(1, -1)
    X_manual_scaled = scaler.transform(X_manual)
    
    y_pred = model.predict(X_manual_scaled)[0]
    y_pred_proba = model.predict_proba(X_manual_scaled)[0]
    
    class_names = {0: 'Открыты', 1: 'Закрыты'}
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ")
    print("="*60)
    print(f"\nПредсказанный класс: {class_names[y_pred]} ({y_pred})")
    print(f"\nВероятности:")
    print(f"  Открыты (0): {y_pred_proba[0]:.4f} ({y_pred_proba[0]*100:.2f}%)")
    print(f"  Закрыты (1): {y_pred_proba[1]:.4f} ({y_pred_proba[1]*100:.2f}%)")


def interactive_menu(model, scaler, feature_names, X, y):
    while True:
        print("\n" + "="*80)
        print("МЕНЮ ТЕСТИРОВАНИЯ")
        print("="*80)
        print("\n1. Тестировать на части датасета")
        print("2. Тестировать на случайных примерах (10 шт)")
        print("3. Ручной ввод данных")
        print("4. Выход")
        
        choice = input("\nВыберите опцию (1-4): ").strip()
        
        if choice == '1':
            test_on_dataset(model, scaler, X, y)
        elif choice == '2':
            test_random_samples(model, scaler, X, y, n_samples=10)
        elif choice == '3':
            test_manual_input(model, scaler, feature_names)
        elif choice == '4':
            print("\nВыход из программы.")
            break
        else:
            print("\n❌ Неверный выбор. Попробуйте снова.")
