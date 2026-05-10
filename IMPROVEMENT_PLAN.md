# План доработок фильтра Калмана

## Анализ текущего состояния

### ✅ Уже реализовано:
1. **Базовые фильтры**: KalmanZ (1D), KalmanXV (x-v), KalmanVA (v-a), Kalman2d
2. **Частичные измерения**: PassX(), PassV(), PassXV() - раздельная обработка координаты и скорости
3. **RTS Smoother**: Для пост-обработки в batch-режиме
4. **Адаптивный шум**: Динамическое обновление q и r на основе NIS
5. **Batch-режим**: PassXBatch(), PassVBatch(), PassXVBatch()
6. **Регуляризация**: Численная стабильность через LDLT и регуляризацию матриц
7. **Мониторинг**: NIS, GetLastInnovation(), IsConvergent()
8. **Управление**: SetU(), SetTimestep(dt)

---

## 🔧 Предложения по улучшению

### 1. Явное разделение Predict и Update (Приоритет: ВЫСОКИЙ)

**Проблема**: Сейчас Predict вызывается неявно внутри Pass(). При неравномерных интервалах между измерениями это неудобно.

**Решение**:
```cpp
// Новый API
void Predict(double dt);           // Явный шаг прогноза с переменным dt
void UpdatePosition(double z, double R);    // Обновление только по координате
void UpdateVelocity(double v, double R);    // Обновление только по скорости
void UpdateFull(const Vector2d& zv, const Matrix2d& RR); // Полное обновление
```

**Преимущества**:
- Гибкая обработка данных с разной частотой (координата 100Hz, скорость 50Hz)
- Возможность пропускать Predict при отсутствии новых данных
- Явный контроль над dt для каждого шага

---

### 2. Динамические матрицы шумов измерений (Приоритет: СРЕДНИЙ)

**Проблема**: R задаётся глобально через параметры фильтра, но разные сенсоры имеют разную точность.

**Решение**:
```cpp
// Передача R в метод Update
void UpdatePosition(double z, double R_override = -1.0);
void UpdateVelocity(double v, double R_override = -1.0);

// Если R_override > 0, используется оно, иначе - rx/rv из параметров
```

**Пример использования**:
```cpp
// GPS с высокой точностью
filter.UpdatePosition(gps_x, 0.5);

// Визуальная одометрия с меньшей точностью
filter.UpdatePosition(vis_x, 5.0);
```

---

### 3. Gatekeeping (валидация инноваций) (Приоритет: ВЫСОКИЙ)

**Проблема**: Выбросы в измерениях могут "сломать" фильтр.

**Решение**:
```cpp
struct FilterConfig {
    double innovationGateThreshold = 3.0;  // Порог в сигмах
    bool rejectOutliers = true;
};

bool UpdatePosition(double z, double R = -1.0);
// Возвращает false, если измерение отклонено как выброс

double GetInnovationMahalanobisDistance() const;
// Расстояние Махаланобиса для последней инновации
```

**Алгоритм**:
1. Вычислить S = H*P*H' + R
2. Вычислить d² = innovation' * S⁻¹ * innovation
3. Если d > threshold → отклонить измерение

---

### 4. Масштабирование Q в зависимости от уверенности в модели (Приоритет: СРЕДНИЙ)

**Проблема**: Фиксированное q не учитывает изменение динамики системы.

**Решение**:
```cpp
void SetProcessNoiseScale(double scale);
// scale > 1.0 → увеличить неопределённость модели (манёвр)
// scale = 1.0 → нормальный режим
// scale < 1.0 → движение по прямой

// Автоматическое масштабирование на основе ускорения
void AutoScaleQ(double estimatedAcceleration);
```

---

### 5. Улучшенная инициализация (Приоритет: НИЗКИЙ)

**Проблема**: Текущая Init() требует ручного указания начального состояния.

**Решение**:
```cpp
// Инициализация по первым измерениям
void InitFromMeasurements(const std::vector<double>& positions, 
                          const std::vector<double>& velocities = {});

// Авто-оценка начальной ковариации
void InitAuto(double firstMeasurement, double measurementVariance);
```

---

### 6. Диагностика и логирование (Приоритет: СРЕДНИЙ)

**Решение**:
```cpp
struct FilterStatistics {
    double avgNIS = 0.0;
    size_t rejectedMeasurements = 0;
    size_t totalUpdates = 0;
    double convergenceRate = 0.0;
};

FilterStatistics GetStatistics() const;
void ResetStatistics();
std::string GetStatusReport() const;  // Текстовый отчёт для отладки
```

---

### 7. Поддержка множественных моделей (IMM) (Приоритет: НИЗКИЙ)

Для сложных сценариев с переменной динамикой:
```cpp
class IMMKalman {
    std::vector<KalmanXV> models;  // Например: постоянное движение, манёвр
    std::vector<double> probabilities;
    
    void Update(double x, double v);
    double GetBestEstimate();
};
```

---

## 📋 План реализации (по этапам)

### Этап 1 (Критичный):
- [ ] Разделить Predict/Update
- [ ] Добавить Gatekeeping
- [ ] Передать R в методы Update

### Этап 2 (Важный):
- [ ] Динамическое масштабирование Q
- [ ] Статистика и диагностика
- [ ] Улучшенная инициализация

### Этап 3 (Опциональный):
- [ ] IMM фильтр
- [ ] Расширенное логирование

---

## 🧪 Тесты для новых функций

1. **Predict/Update**: Неравномерные интервалы dt
2. **Gatekeeping**: Отсев выбросов при больших инновациях
3. **Динамическое R**: Смешанные данные от сенсоров разной точности
4. **Масштабирование Q**: Трекинг маневрирующей цели
5. **Статистика**: Проверка подсчёта отклонённых измерений
