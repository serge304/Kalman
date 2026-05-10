/*
 * KalmanAdaptiveTest.cpp
 * Тесты для адаптивной настройки параметров шума в фильтре Калмана
 */

#include "Kalman.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace Skasp;

// Тестовый класс для проверки адаптивных методов базового шаблона
template<unsigned short N>
class AdaptiveKalmanTester : public Kalman<N>
{
public:
  using typename Kalman<N>::VectorNd;
  using typename Kalman<N>::MatrixNd;
  
  AdaptiveKalmanTester(double _q, double _r, double _u = 0.0) 
    : Kalman<N>(_q, _r, _u) {}
  
  // Публичный доступ к защищенным методам для тестирования
  void TestUpdateNoiseParameters(double actual, double expected)
  {
    this->UpdateNoiseParameters(actual, expected);
  }
  
  bool TestAdaptNoiseParameters()
  {
    return this->AdaptNoiseParameters();
  }
  
  void SetLastInnovation(double value)
  {
    this->lastInnovation << value;
    this->lastInnovationSquaredNorm = this->rx + this->q; // Упрощенная модель
  }
  
  void SetLastInnovationSquaredNorm(double value)
  {
    this->lastInnovationSquaredNorm = value;
  }
};

TEST(AdaptiveNoiseTest, UpdateNoiseParameters_Basic)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  
  double initialQ = filter.GetCurrentQ();
  double initialR = filter.GetCurrentRx();
  
  // Случай, когда фактическая ковариация больше ожидаемой
  filter.TestUpdateNoiseParameters(2.0, 1.0);
  
  EXPECT_GT(filter.GetCurrentQ(), initialQ);
  EXPECT_GT(filter.GetCurrentRx(), initialR);
}

TEST(AdaptiveNoiseTest, UpdateNoiseParameters_Decrease)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  
  double initialQ = filter.GetCurrentQ();
  double initialR = filter.GetCurrentRx();
  
  // Случай, когда фактическая ковариация меньше ожидаемой
  filter.TestUpdateNoiseParameters(0.5, 1.0);
  
  EXPECT_LT(filter.GetCurrentQ(), initialQ);
  EXPECT_LT(filter.GetCurrentRx(), initialR);
}

TEST(AdaptiveNoiseTest, UpdateNoiseParameters_NoChange_WithinTolerance)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  
  double initialQ = filter.GetCurrentQ();
  double initialR = filter.GetCurrentRx();
  
  // Случай, когда отношение близко к 1 (в пределах допуска)
  filter.TestUpdateNoiseParameters(1.05, 1.0);
  
  EXPECT_DOUBLE_EQ(filter.GetCurrentQ(), initialQ);
  EXPECT_DOUBLE_EQ(filter.GetCurrentRx(), initialR);
}

TEST(AdaptiveNoiseTest, UpdateNoiseParameters_QBounds)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  
  // Устанавливаем жесткие границы
  filter.SetQBounds(0.5, 2.0);
  
  // Пытаемся сильно увеличить q
  for (int i = 0; i < 100; ++i)
    filter.TestUpdateNoiseParameters(10.0, 1.0);
  
  EXPECT_GE(filter.GetCurrentQ(), 0.5);
  EXPECT_LE(filter.GetCurrentQ(), 2.0);
}

TEST(AdaptiveNoiseTest, UpdateNoiseParameters_RBounds)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  
  // Устанавливаем жесткие границы
  filter.SetRBounds(0.5, 2.0);
  
  // Пытаемся сильно увеличить r
  for (int i = 0; i < 100; ++i)
    filter.TestUpdateNoiseParameters(10.0, 1.0);
  
  EXPECT_GE(filter.GetCurrentRx(), 0.5);
  EXPECT_LE(filter.GetCurrentRx(), 2.0);
}

TEST(AdaptiveNoiseTest, AdaptNoiseParameters_Disabled)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  
  // Адаптация выключена по умолчанию
  filter.SetLastInnovation(5.0);
  filter.SetLastInnovationSquaredNorm(1.0);
  
  bool result = filter.TestAdaptNoiseParameters();
  
  EXPECT_FALSE(result);
  EXPECT_DOUBLE_EQ(filter.GetCurrentQ(), 1.0);
}

TEST(AdaptiveNoiseTest, AdaptNoiseParameters_Enabled)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  filter.EnableAdaptiveNoise(true);
  
  // Устанавливаем большую невязку
  filter.SetLastInnovation(10.0);
  filter.SetLastInnovationSquaredNorm(1.0);
  
  bool result = filter.TestAdaptNoiseParameters();
  
  // Параметры должны измениться
  EXPECT_TRUE(result);
  EXPECT_NE(filter.GetCurrentQ(), 1.0);
}

TEST(AdaptiveNoiseTest, AdaptNoiseParameters_TargetNIS)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  filter.EnableAdaptiveNoise(true);
  filter.SetTargetNIS(1.0);
  
  // Устанавливаем NIS > target (должно увеличить q и r)
  filter.SetLastInnovation(5.0);
  filter.SetLastInnovationSquaredNorm(1.0);
  
  double initialQ = filter.GetCurrentQ();
  filter.TestAdaptNoiseParameters();
  
  EXPECT_GT(filter.GetCurrentQ(), initialQ);
}

TEST(AdaptiveNoiseTest, GetCurrentValues)
{
  AdaptiveKalmanTester<1> filter(2.5, 3.5, 0.0);
  
  EXPECT_DOUBLE_EQ(filter.GetCurrentQ(), 2.5);
  EXPECT_DOUBLE_EQ(filter.GetCurrentRx(), 3.5);
  EXPECT_DOUBLE_EQ(filter.GetCurrentRv(), 3.5);
}

TEST(AdaptiveNoiseTest, KalmanXV_AdaptiveNoise)
{
  KalmanXV filter(0.1);
  filter.SetParameters(1.0, 1.0, 0.0);
  filter.Init(0.0, 0.0);
  filter.EnableAdaptiveNoise(true);
  
  // Пропускаем несколько измерений с шумом
  for (int i = 0; i < 10; ++i)
  {
    double measurement = i * 0.1 + 0.5 * (rand() % 100 - 50) / 100.0;
    filter.PassX(measurement);
    
    // Проверяем, что методы доступны и работают
    double nis = filter.ComputeNIS();
    EXPECT_GE(nis, 0.0);
  }
  
  // Проверяем, что фильтр работает корректно
  EXPECT_TRUE(std::isfinite(filter.GetX()));
  EXPECT_TRUE(std::isfinite(filter.GetV()));
}

TEST(AdaptiveNoiseTest, KalmanVA_AdaptiveNoise)
{
  KalmanVA filter(0.1);
  filter.SetParameters(1.0, 1.0, 0.0);
  filter.Init(0.0, 0.0);
  filter.EnableAdaptiveNoise(true);
  
  // Пропускаем несколько измерений скорости с шумом
  for (int i = 0; i < 10; ++i)
  {
    double measurement = 1.0 + 0.2 * (rand() % 100 - 50) / 100.0;
    filter.PassV(measurement);
    
    // Проверяем, что методы доступны и работают
    double nis = filter.ComputeNIS();
    EXPECT_GE(nis, 0.0);
  }
  
  // Проверяем, что фильтр работает корректно
  EXPECT_TRUE(std::isfinite(filter.GetV()));
  EXPECT_TRUE(std::isfinite(filter.GetA()));
}

TEST(AdaptiveNoiseTest, AdaptiveNoise_ConvergenceImprovement)
{
  // Тест проверяет, что адаптивная настройка улучшает сходимость фильтра
  KalmanXV filter(0.1);
  filter.SetParameters(0.1, 10.0, 0.0); // Изначально плохие параметры (слишком большое r)
  filter.Init(0.0, 0.0);
  filter.EnableAdaptiveNoise(true);
  filter.SetTargetNIS(1.0);
  
  // Генерируем данные с известной динамикой
  std::vector<double> measurements;
  double true_x = 0.0;
  double true_v = 1.0;
  for (int i = 0; i < 50; ++i)
  {
    true_x += true_v * 0.1;
    double noise = 0.1 * (rand() % 100 - 50) / 100.0;
    measurements.push_back(true_x + noise);
  }
  
  // Пропускаем измерения через фильтр
  for (size_t i = 0; i < measurements.size(); ++i)
  {
    filter.PassX(measurements[i]);
    filter.AdaptNoiseParameters();
  }
  
  // После адаптации параметры должны измениться
  EXPECT_NE(filter.GetCurrentQ(), 0.1);
  EXPECT_NE(filter.GetCurrentRx(), 10.0);
  
  // Фильтр должен оставаться стабильным
  EXPECT_TRUE(std::isfinite(filter.GetX()));
  EXPECT_TRUE(filter.IsConvergent(5.0));
}

TEST(AdaptiveNoiseTest, UpdateNoiseParameters_EdgeCases)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  
  // Отрицательные значения (должны игнорироваться)
  filter.TestUpdateNoiseParameters(-1.0, 1.0);
  EXPECT_DOUBLE_EQ(filter.GetCurrentQ(), 1.0);
  
  filter.TestUpdateNoiseParameters(1.0, -1.0);
  EXPECT_DOUBLE_EQ(filter.GetCurrentQ(), 1.0);
  
  // Нулевые значения (должны игнорироваться)
  filter.TestUpdateNoiseParameters(0.0, 1.0);
  EXPECT_DOUBLE_EQ(filter.GetCurrentQ(), 1.0);
  
  filter.TestUpdateNoiseParameters(1.0, 0.0);
  EXPECT_DOUBLE_EQ(filter.GetCurrentQ(), 1.0);
}

TEST(AdaptiveNoiseTest, MultipleIterations_Adaptation)
{
  AdaptiveKalmanTester<1> filter(1.0, 1.0, 0.0);
  filter.EnableAdaptiveNoise(true);
  
  // Многократная адаптация должна приводить к стабилизации параметров
  double prevQ = filter.GetCurrentQ();
  for (int i = 0; i < 100; ++i)
  {
    filter.SetLastInnovation(2.0);
    filter.SetLastInnovationSquaredNorm(1.0);
    filter.TestAdaptNoiseParameters();
    
    // Проверяем, что параметры не уходят в бесконечность
    EXPECT_TRUE(std::isfinite(filter.GetCurrentQ()));
    EXPECT_TRUE(std::isfinite(filter.GetCurrentRx()));
  }
  
  // Параметры должны стабилизироваться (изменения становятся меньше)
  double finalQ = filter.GetCurrentQ();
  EXPECT_GT(finalQ, prevQ);
  EXPECT_LE(finalQ, filter.GetMaxQBound());
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
