import joblib
import numpy as np

print("="*80)
print("ДЕМОНСТРАЦИЯ МОДЕЛИ КЛАССИФИКАЦИИ СОСТОЯНИЯ ГЛАЗ")
print("Студент: Спирчина В.А.")
print("="*80)

# Загрузка модели
try:
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    model_info = joblib.load('models/model_info.pkl')
    print("\n✅ Модель успешно загружена!")
except:
    print("\n❌ ОШИБКА: Файлы модели не найдены!")
    print("Убедитесь, что вы находитесь в папке проекта.")
    exit(1)

print(f"\nМодель: {model_info['model_name']}")
print(f"Точность: {model_info['accuracy']:.2%}")
print(f"F1-Score: {model_info['f1_score']:.2%}")
print(f"ROC-AUC: {model_info['roc_auc']:.2%}")

def show_menu():
    print("\n" + "="*80)
    print("МЕНЮ ТЕСТИРОВАНИЯ")
    print("="*80)
    print("\n1. Ручной ввод данных ЭЭГ")
    print("2. Информация о модели")
    print("3. Выход")
    print("-"*80)

while True:
    show_menu()
    choice = input("Выберите опцию (1-3): ").strip()
    
    if choice == '1':
        print("\n" + "="*80)
        print("РУЧНОЙ ВВОД ДАННЫХ")
        print("="*80)
        print("\nВам нужно ввести значения для 14 каналов ЭЭГ:")
        print("(Нормальный диапазон: 3500-5000 микровольт)\n")
        
        print("Каналы ЭЭГ:")
        for i, name in enumerate(feature_names, 1):
            print(f"  {i:2d}. {name}")
        
        print("\nВарианты ввода:")
        print("  1) Ввести все 14 значений через пробел")
        print("  2) Использовать случайные значения (для демо)")
        print("  3) Отмена")
        
        sub_choice = input("\nВыберите вариант (1-3): ").strip()
        
        if sub_choice == '1':
            try:
                print(f"\nВведите 14 значений через пробел:")
                values_str = input("Значения: ").strip()
                values = [float(x) for x in values_str.split()]
                
                if len(values) != 14:
                    print(f"\n❌ Ошибка: нужно ровно 14 значений, получено {len(values)}")
                    continue
                
                # Предсказание
                X = np.array(values).reshape(1, -1)
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0]
                
                pred_class = 'Открыты' if prediction == 0 else 'Закрыты'
                confidence = probability[prediction]
                
                print("\n" + "="*80)
                print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ")
                print("="*80)
                print(f"\nСостояние глаз: {pred_class} ({prediction})")
                print(f"Уверенность модели: {confidence:.1%}")
                print(f"\nДетальные вероятности:")
                print(f"  Открыты (0): {probability[0]:.1%}")
                print(f"  Закрыты (1): {probability[1]:.1%}")
                
            except ValueError:
                print("\n❌ Ошибка: введите числовые значения через пробел")
                
        elif sub_choice == '2':
            # Случайные значения в реалистичном диапазоне
            np.random.seed()
            values = np.random.uniform(3500, 5000, 14)
            
            print("\nСгенерированы случайные значения:")
            for name, val in zip(feature_names, values):
                print(f"  {name}: {val:.2f}")
            
            # Предсказание
            X = values.reshape(1, -1)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            pred_class = 'Открыты' if prediction == 0 else 'Закрыты'
            confidence = probability[prediction]
            
            print("\n" + "="*80)
            print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ")
            print("="*80)
            print(f"\nСостояние глаз: {pred_class} ({prediction})")
            print(f"Уверенность модели: {confidence:.1%}")
            print(f"\nДетальные вероятности:")
            print(f"  Открыты (0): {probability[0]:.1%}")
            print(f"  Закрыты (1): {probability[1]:.1%}")
    elif choice == '2':
        print("\n" + "="*80)
        print("ИНФОРМАЦИЯ О МОДЕЛИ")
        print("="*80)
        print(f"\nНазвание модели: {model_info['model_name']}")
        print(f"Дата обучения: {model_info['training_date']}")
        print(f"\nМетрики качества:")
        print(f"  Accuracy:  {model_info['accuracy']:.2%}")
        print(f"  F1-Score:  {model_info['f1_score']:.2%}")
        print(f"  ROC-AUC:   {model_info['roc_auc']:.2%}")
        print(f"  CV Score:  {model_info['cv_mean']:.2%} ± {model_info['cv_std']:.2%}")
        print(f"\nДатасет: EEG Eye State (UCI ML Repository)")
        print(f"Размер обучающей выборки: 11,984 наблюдений")
        print(f"Размер тестовой выборки: 2,996 наблюдений")
        print(f"Количество признаков: 14 каналов ЭЭГ")
        
    elif choice == '3':
        print("\n" + "="*80)
        print("Спасибо за тестирование модели!")
        print("="*80)
        break
        
    else:
        print("\n❌ Неверный выбор. Введите число от 1 до 3.")


