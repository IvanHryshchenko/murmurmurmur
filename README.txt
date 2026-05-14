# K1 AI Prediction Project

## 📌 Описание

Проект использует нейросеть для предсказания значения PLANT STANDARD
на основе параметров GCSD и ADJ.

Модель построена на PyTorch и обучается как регрессионная нейросеть.

- нейросеть на PyTorch;
- обучение на данных;
- feature engineering;
- train/validation split;
- нормализация;
- сохранение модели;
- inference (predict.py);
- regression task.

Проект использует нейросетевую модель машинного обучения для регрессионного предсказания значений PLANT STANDARD на основе производственных параметров.

---

## 🧠 Модель

- Тип: Feedforward Neural Network
- Loss: MSE
- Optimizer: Adam
- Output: непрерывное значение (regression)

---

## 📊 Признаки

- GCSD
- ADJ
- логарифмические преобразования
- взаимодействие признаков
- характеристики PART NUMBER

---

## 🚀 Запуск обучения

```bash
python train.py

## 🚀 Запуск работы нейросети

```bash
python predict.py

##  📦Требования для запуска(зависимости)
- torch
- pandas
- numpy
- scikit-learn
- openpyxl