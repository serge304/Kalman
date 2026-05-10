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
      regularizationEpsilon(REGULARIZATION_EPSILON) { }

  Kalman(double _q, double _r, double _u)
    : q(_q), rx(_r), rv(_r), u(_u), I(MatrixNd::Identity()),
      regularizationEpsilon(REGULARIZATION_EPSILON) { }

  Kalman(double _q, double _rx, double _rv, double _u)
    : q(_q), rx(_rx), rv(_rv), u(_u), I(MatrixNd::Identity()),
      regularizationEpsilon(REGULARIZATION_EPSILON) { }

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

} // namespace Skasp { namespace Base {

#endif /* KALMAN_H_ */
