/*
 * Kalman.h
 *
 *  Created on: May 29, 2018
 *      Author: sergey
 */

#ifndef KALMAN_H_
#define KALMAN_H_

#include "Eigen/Eigen"
#include <stddef.h>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace Skasp {

/// Общая форма фильтра Калмана с улучшенной численной стабильностью
template<unsigned short N>
class Kalman
{
public:
  typedef Eigen::Matrix<double, 1, 1> Scalar;
  typedef Eigen::Matrix<double, N, 1> VectorNd;
  typedef Eigen::Matrix<double, 1, N> RowVectorNd;
  typedef Eigen::Matrix<double, N, N> MatrixNd;
  typedef std::vector<VectorNd, Eigen::aligned_allocator<VectorNd> > VectorArray;
  typedef std::vector<MatrixNd, Eigen::aligned_allocator<MatrixNd> > MatrixArray;

  /// Регуляризация для предотвращения вырождения матриц
  static constexpr double REGULARIZATION_EPSILON = 1e-10;

public:
  Kalman()
    : q(0.0), rx(0.0), rv(0.0), u(0.0), I(MatrixNd::Identity()), 
      regularizationEpsilon(REGULARIZATION_EPSILON),
      adaptiveNoiseEnabled(false), targetNIS(1.0),
      minQBound(1e-6), maxQBound(1e6),
      minRBound(1e-6), maxRBound(1e6) { }

  Kalman(double _q, double _r, double _u)
    : q(_q), rx(_r), rv(_r), u(_u), I(MatrixNd::Identity()),
      regularizationEpsilon(REGULARIZATION_EPSILON),
      adaptiveNoiseEnabled(false), targetNIS(1.0),
      minQBound(1e-6), maxQBound(1e6),
      minRBound(1e-6), maxRBound(1e6) { }

  Kalman(double _q, double _rx, double _rv, double _u)
    : q(_q), rx(_rx), rv(_rv), u(_u), I(MatrixNd::Identity()),
      regularizationEpsilon(REGULARIZATION_EPSILON),
      adaptiveNoiseEnabled(false), targetNIS(1.0),
      minQBound(1e-6), maxQBound(1e6),
      minRBound(1e-6), maxRBound(1e6) { }

  void SetU(double _u) { u = _u; }

  /// Установить параметр регуляризации для предотвращения вырождения матриц
  void SetRegularizationEpsilon(double eps) { regularizationEpsilon = eps; }

  /// Получить последнюю невязку (инновацию)
  Scalar GetLastInnovation() const { return lastInnovation; }

  /// Вычислить нормированную невязку (NIS) для детектирования аномалий
  double ComputeNIS() const
  {
    if (lastInnovationSquaredNorm <= 0.0)
      return 0.0;
    return lastInnovation.squaredNorm() / lastInnovationSquaredNorm;
  }

  /// Проверка сходимости фильтра по NIS
  bool IsConvergent(double threshold = 3.0) const
  {
    return ComputeNIS() < threshold * threshold;
  }

  /// Включить/выключить адаптивную настройку параметров шума
  void EnableAdaptiveNoise(bool enable) { adaptiveNoiseEnabled = enable; }
  
  /// Установить целевое значение NIS для адаптивной настройки
  void SetTargetNIS(double target) { targetNIS = target; }
  
  /// Получить текущее значение NIS
  double GetCurrentNIS() const { return ComputeNIS(); }
  
  /// Обновить параметры q и r на основе ковариации инноваций
  /// Использует метод адаптивной настройки на основе отношения фактической и ожидаемой ковариации инноваций
  void UpdateNoiseParameters(double actualInnovationCovariance, double expectedInnovationCovariance);
  
  /// Динамически обновить q и r на основе последней инновации
  /// Возвращает true, если параметры были обновлены
  bool AdaptNoiseParameters();
  
  /// Установить границы для адаптивного изменения q
  void SetQBounds(double minQ, double maxQ) { minQBound = minQ; maxQBound = maxQ; }
  
  /// Установить границы для адаптивного изменения r
  void SetRBounds(double minR, double maxR) { minRBound = minR; maxRBound = maxR; }
  
  /// Получить текущие значения q и r
  double GetCurrentQ() const { return q; }
  double GetCurrentRx() const { return rx; }
  double GetCurrentRv() const { return rv; }
  
  /// Получить границы для q и r (для тестов)
  double GetMinQBound() const { return minQBound; }
  double GetMaxQBound() const { return maxQBound; }
  double GetMinRBound() const { return minRBound; }
  double GetMaxRBound() const { return maxRBound; }

protected:
  /// Принудительная симметризация матрицы
  MatrixNd Symmetrize(const MatrixNd& M) const
  {
    return 0.5 * (M + M.transpose());
  }

  /// Регуляризация матрицы для предотвращения вырождения
  MatrixNd Regularize(const MatrixNd& M) const
  {
    return M + regularizationEpsilon * MatrixNd::Identity();
  }

  /// Безопасное вычисление обратной матрицы с использованием LDLT разложения
  MatrixNd SafeInverse(const MatrixNd& M) const
  {
    // Добавляем регуляризацию для предотвращения сингулярности
    MatrixNd M_reg = Regularize(M);
    
    // Используем LDLT разложение для более стабильного обращения
    Eigen::LDLT<MatrixNd> ldlt(M_reg);
    
    if (ldlt.info() == Eigen::Success)
      return ldlt.solve(MatrixNd::Identity());
    else
    {
      // Fallback: используем полный pivoting LU
      Eigen::FullPivLU<MatrixNd> lu(M_reg);
      return lu.solve(MatrixNd::Identity());
    }
  }

  void PassScalar(const Scalar& Z)
  {
    // Проверка входных данных
    if (!std::isfinite(Z(0)))
      throw std::invalid_argument("Kalman: входное значение не является конечным числом");
    if (q < 0.0 || rx < 0.0)
      throw std::invalid_argument("Kalman: параметры шума должны быть неотрицательными");

    // prediction
    VectorNd Y_ = F * Y + B * u;
    MatrixNd P_ = F * P * F.transpose() + q * Q;
    
    // Принудительная симметризация P_
    P_ = Symmetrize(P_);

    // correction
    Scalar S = H * P_ * H.transpose() + R;
    // Регуляризация ковариации инноваций
    S(0, 0) += regularizationEpsilon;
    
    VectorNd K = P_ * H.transpose() * SafeInverse(S);
    
    // Вычисление невязки
    lastInnovation = Z - H * Y_;
    lastInnovationSquaredNorm = S(0, 0);
    
    Y = Y_ + K * lastInnovation;
    
    // Формула Джозефа для численной стабильности
    MatrixNd IKH = I - K * H;
    P = Symmetrize(IKH * P_ * IKH.transpose() + K * R * K.transpose());
  }

  void PassVector(const VectorNd& Z)
  {
    // Проверка входных данных
    for (int i = 0; i < Z.rows(); ++i)
      if (!std::isfinite(Z(i)))
        throw std::invalid_argument("Kalman: входное значение не является конечным числом");
    if (q < 0.0 || rx < 0.0 || rv < 0.0)
      throw std::invalid_argument("Kalman: параметры шума должны быть неотрицательными");

    // prediction
    VectorNd Y_ = F * Y + B * u;
    MatrixNd P_ = F * P * F.transpose() + q * Q;
    
    // Принудительная симметризация P_
    P_ = Symmetrize(P_);

    // correction
    MatrixNd S = HH * P_ * HH.transpose() + RR;
    // Регуляризация ковариации инноваций
    S = S + regularizationEpsilon * MatrixNd::Identity();
    
    MatrixNd K = P_ * HH.transpose() * SafeInverse(S);
    
    // Вычисление невязки
    lastInnovationVector = Z - HH * Y_;
    lastInnovation = Scalar(lastInnovationVector.norm());
    lastInnovationSquaredNorm = S.trace() / S.rows();
    
    Y = Y_ + K * lastInnovationVector;
    
    // Формула Джозефа для численной стабильности
    MatrixNd IKH = I - K * HH;
    P = Symmetrize(IKH * P_ * IKH.transpose() + K * RR * K.transpose());
  }

  void RTSSmoother()
  {
    if (Ys.size() > 1)
    {
      for (size_t k = Ys.size() - 2; k > 0; --k)
      {
        MatrixNd PP = Fs[k] * Ps[k] * Fs[k].transpose() + Qs[k];
        // Принудительная симметризация
        PP = Symmetrize(PP);
        // Регуляризация для предотвращения сингулярности
        PP = PP + regularizationEpsilon * MatrixNd::Identity();
        
        MatrixNd K = Ps[k] * Fs[k].transpose() * SafeInverse(PP);
        Ys[k] += K * (Ys[k + 1] - Fs[k] * Ys[k]);
        Ps[k] += K * (Ps[k + 1] - PP) * K.transpose();
        // Принудительная симметризация
        Ps[k] = Symmetrize(Ps[k]);
      }
    }
  }

  void UpdatePropagation()
  {
      Ps.push_back(P);
      Fs.push_back(F);
      Qs.push_back(Q);
      Ys.push_back(Y);
  }

  void Reset()
  {
      P = Pinit;

      Ps.resize(0);
      Fs.resize(0);
      Qs.resize(0);
      Ys.resize(0);
      
      lastInnovation = Scalar(0.0);
      lastInnovationSquaredNorm = 0.0;
      
      // Сброс адаптивных параметров к исходным значениям не требуется,
      // так как они хранят границы и настройки, а не состояние
  }

public:
	void Reserve(size_t n)
	{
		Ps.reserve(n);
		Fs.reserve(n);
		Qs.reserve(n);
		Ys.reserve(n);
	}

protected:
  double q, rx, rv, u;
  VectorNd Y;
  MatrixNd P;
  MatrixNd F;
  VectorNd B;
  MatrixNd Q;
  MatrixNd I;

  RowVectorNd H;
  Scalar R;

  MatrixNd HH;
  MatrixNd RR;
  MatrixNd Pinit;

  VectorArray Ys;
  MatrixArray Ps, Fs, Qs;
  
  // Переменные для мониторинга сходимости
  Scalar lastInnovation;
  VectorNd lastInnovationVector;
  double lastInnovationSquaredNorm;
  double regularizationEpsilon;
  
  // Переменные для адаптивной настройки параметров шума
  bool adaptiveNoiseEnabled;
  double targetNIS;
  double minQBound, maxQBound;
  double minRBound, maxRBound;
};

/// Фильтр первого порядка
class KalmanZ : public Kalman<1>
{
public:
  typedef Kalman<1> base;

public:
  KalmanZ(double _q, double _r, double _u = 0);
  void SetParameters(double _q, double _r, double _u);
  
  /// Инициализация фильтра с заданным начальным значением
  /// Ковариационная матрица инициализируется большим значением для отражения неопределенности
  void Init(double z0);
  
  /// Инициализация фильтра с явным указанием начальной ковариации
  void Init(double z0, double p0);
  
  void Pass(double z);
  double GetZ();
  
  /// Получить текущую ковариацию (для отладки и мониторинга)
  double GetCovariance() const;
  
private:
  void SetConstants();
};

/// Фильтры второго порядка
class Kalman2d : public Kalman<2>
{
public:
    typedef Kalman<2> base;

protected:
    Kalman2d();

public:
    /// Инициализация фильтра с заданными начальными значениями
    /// Ковариационная матрица инициализируется значениями по умолчанию (большие значения)
    void Init(double y0, double y1);
    
    /// Инициализация фильтра с явным указанием начальной ковариации
    void Init(double y0, double y1, const MatrixNd& P0);

    void SetParameters(double _q, double _r, double _u);
    void SetParameters(double _q, double _rx, double _rv, double _u);

    void GetSmoothedCurve(std::vector<double>& yy0, std::vector<double>& yy1);
    
    /// Получить текущую ковариационную матрицу (для отладки и мониторинга)
    MatrixNd GetCovariance() const;

private:
    void SetPinit();
};

class KalmanXV : public Kalman2d
{
public:
  KalmanXV();
  KalmanXV(double dt);

  void SetTimestep(double dt);

  void PassX(double x);
  void PassV(double v);
  void PassXV(double x, double v);

  void PassXSmooth(double x);
  void PassVSmooth(double v);
  void PassXVSmooth(double x, double v);

  double GetX() const;
  double GetV() const;
  
  /// Получить текущую невязку (инновацию)
  double GetLastInnovationValue() const;
  
  /// Проверка сходимости фильтра
  bool IsConvergent(double threshold = 3.0) const;

  void PassXBatch(const std::vector<double>& x, std::vector<double>& xk, std::vector<double>& vk);
  void PassVBatch(const std::vector<double>& v, std::vector<double>& xk, std::vector<double>& vk);
  void PassXVBatch(const std::vector<double>& x, const std::vector<double>& v, std::vector<double>& xk, std::vector<double>& vk);

private:
  void SetForX();
  void SetForV();
  void SetForXV();
};

class KalmanVA : public Kalman2d
{
public:
  KalmanVA();
  KalmanVA(double dt);

  void SetTimestep(double dt);

  void PassV(double v);
  void PassVSmooth(double v);

  double GetV() const;
  double GetA() const;
  
  /// Получить текущую невязку (инновацию)
  double GetLastInnovationValue() const;
  
  /// Проверка сходимости фильтра
  bool IsConvergent(double threshold = 3.0) const;

  void PassVBatch(const std::vector<double>& v, std::vector<double>& vk, std::vector<double>& ak);

private:
  void SetForV();
};

} // namespace Skasp

/// Реализация методов адаптивной настройки параметров шума

namespace Skasp {

template<unsigned short N>
void Kalman<N>::UpdateNoiseParameters(double actualInnovationCovariance, double expectedInnovationCovariance)
{
  if (expectedInnovationCovariance <= 0.0 || actualInnovationCovariance <= 0.0)
    return;
  
  // Вычисляем отношение фактической ковариации к ожидаемой
  double ratio = actualInnovationCovariance / expectedInnovationCovariance;
  
  // Если отношение близко к 1, параметры не требуют корректировки
  const double tolerance = 0.1; // 10% допуск
  if (std::abs(ratio - 1.0) < tolerance)
    return;
  
  // Адаптивно обновляем q и r
  // Если ratio > 1, то фактическая неопределенность больше ожидаемой -> увеличиваем q или r
  // Если ratio < 1, то фактическая неопределенность меньше ожидаемой -> уменьшаем q или r
  
  // Коэффициент адаптации (чем больше, тем быстрее адаптация)
  const double adaptationRate = 0.1;
  
  // Обновляем q (шум процесса)
  double newQ = q * std::pow(ratio, adaptationRate);
  newQ = std::max(minQBound, std::min(maxQBound, newQ));
  q = newQ;
  
  // Обновляем r (шум измерения) - используем тот же подход
  double newR = rx * std::pow(ratio, adaptationRate);
  newR = std::max(minRBound, std::min(maxRBound, newR));
  rx = newR;
  rv = newR;
}

template<unsigned short N>
bool Kalman<N>::AdaptNoiseParameters()
{
  if (!adaptiveNoiseEnabled || lastInnovationSquaredNorm <= 0.0)
    return false;
  
  // Вычисляем текущее NIS
  double currentNIS = ComputeNIS();
  
  // Если NIS значительно отличается от целевого значения, адаптируем параметры
  const double nisTolerance = 0.5;
  if (std::abs(currentNIS - targetNIS) < nisTolerance)
    return false;
  
  // Определяем направление корректировки
  // Если NIS > targetNIS, значит невязка слишком большая -> увеличиваем q или r
  // Если NIS < targetNIS, значит фильтр слишком "уверен" -> можно уменьшить q или r
  
  double ratio = currentNIS / targetNIS;
  const double adaptationRate = 0.05; // Более медленная адаптация для стабильности
  
  // Сохраняем старые значения для проверки изменений
  double oldQ = q;
  double oldRx = rx;
  
  // Адаптируем q (шум процесса)
  if (ratio > 1.0)
  {
    // Увеличиваем q, чтобы учесть большую неопределенность в модели
    q = q * (1.0 + adaptationRate * (ratio - 1.0));
  }
  else
  {
    // Уменьшаем q
    q = q * (1.0 - adaptationRate * (1.0 - ratio));
  }
  q = std::max(minQBound, std::min(maxQBound, q));
  
  // Адаптируем r (шум измерения)
  if (ratio > 1.0)
  {
    // Увеличиваем r, чтобы меньше доверять измерениям с большим шумом
    rx = rx * (1.0 + adaptationRate * (ratio - 1.0));
    rv = rv * (1.0 + adaptationRate * (ratio - 1.0));
  }
  else
  {
    // Уменьшаем r
    rx = rx * (1.0 - adaptationRate * (1.0 - ratio));
    rv = rv * (1.0 - adaptationRate * (1.0 - ratio));
  }
  rx = std::max(minRBound, std::min(maxRBound, rx));
  rv = std::max(minRBound, std::min(maxRBound, rv));
  
  // Возвращаем true, если параметры изменились
  return (std::abs(q - oldQ) > 1e-12) || (std::abs(rx - oldRx) > 1e-12);
}

} // namespace Skasp

#endif /* KALMAN_H_ */
