/*
 * Kalman.cpp
 *
 *  Created on: May 29, 2018
 *      Author: sergey
 */

#include "Kalman.h"

namespace Skasp {

///************************************************************

KalmanZ::KalmanZ(double _q, double _r, double _u)
  : base(_q, _r, _u)
{
  SetConstants();
}

void KalmanZ::SetConstants()
{
    H << 1.0;
    R << rx;
    F << 1.0;
    B << 1.0;
    Q << 1.0;
}

void KalmanZ::SetParameters(double _q, double _r, double _u)
{
    q = _q; rx = _r; rv = _r; u = _u;
    SetConstants();
}

void KalmanZ::Init(double z0)
{
  Y << z0;
  // Инициализация большой ковариацией для отражения высокой неопределенности начального состояния
  P << 1.0;
}

void KalmanZ::Init(double z0, double p0)
{
  Y << z0;
  // Явная инициализация ковариации
  P << p0;
}

double KalmanZ::GetCovariance() const
{
  return P(0, 0);
}

void KalmanZ::Pass(double z)
{
  base::PassScalar(base::Scalar(z));
}

double KalmanZ::GetZ()
{
  return Y(0);
}

///************************************************************

Kalman2d::Kalman2d()
  : base()
{
    SetPinit();
}

void Kalman2d::SetPinit()
{
    // Инициализация большими значениями для отражения высокой неопределенности начального состояния
    Pinit(0,0) = 1.0;
    Pinit(0,1) = 0.0;
    Pinit(1,0) = 0.0;
    Pinit(1,1) = 1.0;
}

void Kalman2d::Init(double y0, double y1)
{
  Reset();
  Y = base::VectorNd(y0, y1);
}

void Kalman2d::Init(double y0, double y1, const MatrixNd& P0)
{
  Reset();
  Y = base::VectorNd(y0, y1);
  P = P0;
}

typename Kalman2d::MatrixNd Kalman2d::GetCovariance() const
{
  return P;
}

void Kalman2d::SetParameters(double _q, double _r, double _u)
{
  SetParameters(_q, _r, _r, _u);
}

void Kalman2d::SetParameters(double _q, double _rx, double _rv, double _u)
{
  q = _q; rx = _rx; rv = _rv; u = _u;
}

void Kalman2d::GetSmoothedCurve(std::vector<double>& yy0, std::vector<double>& yy1)
{
    yy0.resize(Ys.size());
    yy1.resize(Ys.size());

    if (Ys.size() > 2)
      RTSSmoother();

    for (size_t i = 0; i < Ys.size(); ++i)
    {
        yy0[i] = Ys[i](0);
        yy1[i] = Ys[i](1);
    }
}

///************************************************************

KalmanXV::KalmanXV()
{

}

KalmanXV::KalmanXV(double dt)
{
    SetTimestep(dt);
}

void KalmanXV::SetTimestep(double dt)
{
    double dt2 = dt*dt;
    double dt3 = dt2*dt;
    double dt4 = dt2*dt2;

    F << 1.0, dt,
         0.0, 1.0;

    B << 0.5*dt2, dt;

    Q << 0.25*dt4, 0.5*dt3,
          0.5*dt3,     dt2;
}

void KalmanXV::SetForX()
{
    H = base::RowVectorNd::Zero();
    H(0) = 1.0;
    R << rx;
}

void KalmanXV::SetForV()
{
    H = base::RowVectorNd::Zero();
    H(1) = 1.0;
    R << rv;
}

void KalmanXV::SetForXV()
{
    HH = MatrixNd::Zero();
    HH(0, 0) = 1.0;
    HH(1, 1) = 1.0;

    RR << rx, 0.0,
          0.0, rv;
}

void KalmanXV::Predict(double dt)
{
    // Обновляем матрицы для нового dt
    SetTimestep(dt);
    
    // Выполняем шаг прогнозирования
    Y = F * Y + B * u;
    P = F * P * F.transpose() + q * Q;
    
    // Принудительная симметризация P
    P = base::Symmetrize(P);
}

bool KalmanXV::UpdateX(double x)
{
    SetForX();
    base::VectorNd z;
    z << x, 0.0;
    base::HH = base::MatrixNd::Zero();
    base::HH(0, 0) = 1.0;
    base::RR << rx, 0.0, 0.0, 0.0;
    base::PassVector(z);
    return !WasLastMeasurementRejected();
}

bool KalmanXV::UpdateV(double v)
{
    SetForV();
    base::VectorNd z;
    z << 0.0, v;
    base::HH = base::MatrixNd::Zero();
    base::HH(1, 1) = 1.0;
    base::RR << 0.0, 0.0, 0.0, rv;
    base::PassVector(z);
    return !WasLastMeasurementRejected();
}

void KalmanXV::UpdateXV(double x, double v)
{
    SetForXV();
    base::PassVector(base::VectorNd(x, v));
}

void KalmanXV::PassX(double x)
{
  Predict(base::F(0, 1)); // Используем dt из текущей матрицы F
  UpdateX(x);
}

void KalmanXV::PassV(double v)
{
  Predict(base::F(0, 1)); // Используем dt из текущей матрицы F
  UpdateV(v);
}

void KalmanXV::PassXV(double x, double v)
{
  Predict(base::F(0, 1)); // Используем dt из текущей матрицы F
  UpdateXV(x, v);
}

void KalmanXV::PassXSmooth(double x)
{
    UpdateX(x);
    base::UpdatePropagation();
}

void KalmanXV::PassVSmooth(double v)
{
    UpdateV(v);
    base::UpdatePropagation();
}

void KalmanXV::PassXVSmooth(double x, double v)
{
    UpdateXV(x, v);
    base::UpdatePropagation();
}

double KalmanXV::GetX() const
{
  return Y(0);
}

double KalmanXV::GetV() const
{
  return Y(1);
}

double KalmanXV::GetLastInnovationValue() const
{
  return base::lastInnovation(0, 0);
}

bool KalmanXV::IsConvergent(double threshold) const
{
  return base::IsConvergent(threshold);
}

void KalmanXV::PassXBatch(const std::vector<double>& x, std::vector<double>& xk, std::vector<double>& vk)
{
  Reset();

  // Прямой проход с явным прогнозированием и обновлением
  for (size_t i = 0; i < x.size(); ++i)
  {
    if (i > 0)
      Predict(base::F(0, 1));
    PassXSmooth(x[i]);
  }

  // Сглаживание
  GetSmoothedCurve(xk, vk);
}

void KalmanXV::PassVBatch(const std::vector<double>& v, std::vector<double>& xk, std::vector<double>& vk)
{
    Reset();

    // Прямой проход с явным прогнозированием и обновлением
    for (size_t i = 0; i < v.size(); ++i)
    {
      if (i > 0)
        Predict(base::F(0, 1));
      PassVSmooth(v[i]);
    }

    // Сглаживание
    GetSmoothedCurve(xk, vk);
}

void KalmanXV::PassXVBatch(const std::vector<double>& x, const std::vector<double>& v, std::vector<double>& xk, std::vector<double>& vk)
{
    Reset();

    // Прямой проход с явным прогнозированием и обновлением
    for (size_t i = 0; i < x.size(); ++i)
    {
      if (i > 0)
        Predict(base::F(0, 1));
      PassXVSmooth(x[i], v[i]);
    }

    // Сглаживание
    GetSmoothedCurve(xk, vk);
}

///************************************************************

KalmanVA::KalmanVA()
{

}

KalmanVA::KalmanVA(double dt)
{
    SetTimestep(dt);
}

void KalmanVA::SetTimestep(double dt)
{
    double dt2 = dt*dt;

    F << 1.0, dt,
        0.0, 1.0;

    B << dt, 1.0;

    Q << dt2, dt,
         dt,  1.0;
}

void KalmanVA::SetForV()
{
    H = base::RowVectorNd::Zero();
    H(0) = 1.0;
    R << rv;
}

void KalmanVA::PassV(double v)
{
  SetForV();
  // For N=2, use PassVector with single-element measurement
  base::VectorNd z;
  z << v, 0.0;
  base::HH = base::MatrixNd::Zero();
  base::HH(0, 0) = 1.0;
  base::RR << rv, 0.0, 0.0, 0.0;
  base::PassVector(z);
}

void KalmanVA::PassVSmooth(double v)
{
    PassV(v);
    UpdatePropagation();
}

double KalmanVA::GetV() const
{
  return Y(0);
}

double KalmanVA::GetA() const
{
  return Y(1);
}

double KalmanVA::GetLastInnovationValue() const
{
  return base::lastInnovation(0, 0);
}

bool KalmanVA::IsConvergent(double threshold) const
{
  return base::IsConvergent(threshold);
}

void KalmanVA::PassVBatch(const std::vector<double>& v, std::vector<double>& vk, std::vector<double>& ak)
{
    Reset();

    // Прямой проход
    for (size_t i = 0; i < v.size(); ++i)
      PassVSmooth(v[i]);

    // Сглаживание
    GetSmoothedCurve(vk, ak);
}

} // namespace Skasp { namespace Base {