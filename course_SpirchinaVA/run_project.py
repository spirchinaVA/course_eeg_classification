import sys
import importlib.util
import numpy as np
from sklearn.model_selection import train_test_split

# Импорт функций из train_model.py и test_model.py
import train_model
import test_model


def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)


def check_dependencies():
    print_header("ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    
    required_packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'scipy',
        'joblib',
        'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
            print(f"❌ {package} - не установлен")
        else:
            print(f"✓ {package} - установлен")
    
    if missing_packages:
        print(f"\n❌ Отсутствуют зависимости: {', '.join(missing_packages)}")
        print("\nУстановите их с помощью:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ Все зависимости установлены")
    return True


def load_and_split_data():
    print_header("ЗАГРУЗКА И РАЗДЕЛЕНИЕ ДАННЫХ")
    
    # 1. Создание директорий
    train_model.create_directories()
    
    # 2. Загрузка данных
    df = train_model.download_and_load_data()
    
    # 3. Разделение на признаки и метки
    feature_cols = [col for col in df.columns if col != 'eyeDetection']
    X = df[feature_cols].values
    y = df['eyeDetection'].values
    
    # 4. ЕДИНОЕ разделение на train/test (80/20, random_state=42)
    print("\n" + "-"*80)
    print("РАЗДЕЛЕНИЕ ДАТАСЕТА (train_test_split)")
    print("-"*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nРазделение данных (test_size=0.2, random_state=42):")
    print(f"  Обучающая выборка: {X_train.shape[0]} наблюдений")
    print(f"  Тестовая выборка: {X_test.shape[0]} наблюдений")
    
    print(f"\nРаспределение классов в обучающей выборке:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Класс {int(label)}: {count} ({count/len(y_train)*100:.2f}%)")
    
    print(f"\nРаспределение классов в тестовой выборке:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Класс {int(label)}: {count} ({count/len(y_test)*100:.2f}%)")
    
    return df, X_train, X_test, y_train, y_test, feature_cols


def run_training(df, X_train, X_test, y_train, y_test, feature_cols):
    print_header("ОБУЧЕНИЕ МОДЕЛЕЙ")
    
    # EDA
    train_model.exploratory_data_analysis(df)
    
    # Подготовка данных (нормализация)
    X_train_scaled, X_test_scaled, scaler = train_model.prepare_data_split(
        X_train, X_test, y_train, y_test
    )
    
    # Обучение моделей
    results = train_model.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Сравнение моделей
    comparison_df = train_model.compare_models(results)
    
    # Сохранение лучшей модели
    best_model_name, best_result = train_model.save_final_model(
        results, comparison_df, scaler, feature_cols
    )
    
    return best_model_name, best_result, scaler


def run_testing(X_test, y_test):
    print_header("ТЕСТИРОВАНИЕ МОДЕЛИ")
    
    import joblib
    
    # Загрузка модели
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    model_info = joblib.load('models/model_info.pkl')
    
    print(f"\n✓ Модель загружена: {model_info['model_name']}")
    
    # Тестирование на ТОЙ ЖЕ тестовой выборке
    y_pred, y_pred_proba, cm = test_model.test_on_test_set(
        model, scaler, X_test, y_test
    )
    
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ ЗАВЕРШЕНЫ УСПЕШНО!")
    print("="*80)
    
    # Интерактивное меню для дополнительного тестирования
    while True:
        print("\n" + "="*80)
        print("МЕНЮ ДОПОЛНИТЕЛЬНОГО ТЕСТИРОВАНИЯ")
        print("="*80)
        print("1. Тестирование на случайных примерах из тестовой выборки")
        print("2. Ручной ввод данных")
        print("3. Показать информацию о модели")
        print("4. Выход")
        
        choice = input("\nВыберите опцию (1-4): ").strip()
        
        if choice == '1':
            n = input("Сколько примеров протестировать? (по умолчанию 10): ").strip()
            n_samples = int(n) if n.isdigit() and int(n) > 0 else 10
            test_model.test_random_samples(model, scaler, X_test, y_test, n_samples)
            
        elif choice == '2':
            test_model.test_manual_input(model, scaler, feature_names)
            
        elif choice == '3':
            test_model.show_model_info(model_info, feature_names)
            
        elif choice == '4':
            print("\n" + "="*80)
            print("Завершение программы.")
            print("="*80)
            break
            
        else:
            print("\n❌ Неверный выбор. Введите число от 1 до 4.")


def main():
    print("="*80)
    print("КУРСОВОЙ ПРОЕКТ: КЛАССИФИКАЦИЯ СОСТОЯНИЯ ГЛАЗ ПО ЭЭГ")
    print("Студент: Спирчина В.А.")
    print("="*80)
    
    # 1. Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    # 2. Загрузка и ЕДИНОЕ разделение данных
    df, X_train, X_test, y_train, y_test, feature_cols = load_and_split_data()
    
    # 3. Обучение моделей
    best_model_name, best_result, scaler = run_training(
        df, X_train, X_test, y_train, y_test, feature_cols
    )
    
    # 4. Тестирование на той же тестовой выборке
    run_testing(X_test, y_test)


if __name__ == "__main__":
    main()
