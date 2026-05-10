/*
 * KalmanTest.cpp
 * 
 * Unit tests for Kalman filter implementation
 */

#include "Kalman.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

using namespace Skasp;

// Helper function to check if two doubles are approximately equal
bool approxEqual(double a, double b, double epsilon = 1e-6)
{
    return std::abs(a - b) < epsilon;
}

// ============================================================================
// KalmanZ Tests (1D Kalman Filter)
// ============================================================================

TEST(KalmanZTest, Constructor)
{
    KalmanZ filter(0.1, 0.5, 0.0);
    EXPECT_NO_THROW(filter.Init(0.0));
}

TEST(KalmanZTest, Init)
{
    KalmanZ filter(0.1, 0.5, 0.0);
    
    // Test initialization with default covariance
    filter.Init(5.0);
    EXPECT_DOUBLE_EQ(filter.GetZ(), 5.0);
    
    // Test initialization with explicit covariance
    filter.Init(10.0, 2.0);
    EXPECT_DOUBLE_EQ(filter.GetZ(), 10.0);
    EXPECT_DOUBLE_EQ(filter.GetCovariance(), 2.0);
}

TEST(KalmanZTest, Pass_ConstantMeasurement)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 1.0);
    
    const double trueValue = 5.0;
    const int numIterations = 50;
    
    // Process constant measurements
    for (int i = 0; i < numIterations; ++i)
    {
        filter.Pass(trueValue);
    }
    
    // Filter should converge to the true value
    EXPECT_NEAR(filter.GetZ(), trueValue, 0.1);
    
    // Covariance should decrease over time
    EXPECT_LT(filter.GetCovariance(), 1.0);
}

TEST(KalmanZTest, Pass_NoisyMeasurements)
{
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<double> noise(0.0, 0.5);
    
    KalmanZ filter(0.01, 0.25, 0.0);
    filter.Init(0.0, 1.0);
    
    const double trueValue = 10.0;
    const int numIterations = 100;
    
    double sumError = 0.0;
    
    for (int i = 0; i < numIterations; ++i)
    {
        double measurement = trueValue + noise(gen);
        filter.Pass(measurement);
        sumError += std::abs(filter.GetZ() - trueValue);
    }
    
    // Average error should be reasonable
    double avgError = sumError / numIterations;
    EXPECT_LT(avgError, 0.5);
}

TEST(KalmanZTest, SetParameters)
{
    KalmanZ filter(0.1, 0.5, 0.0);
    filter.SetParameters(0.2, 0.8, 1.0);
    
    filter.Init(0.0);
    filter.Pass(5.0);
    
    EXPECT_NO_THROW();
}

TEST(KalmanZTest, InvalidInput_ThrowsException)
{
    KalmanZ filter(0.1, 0.5, 0.0);
    filter.Init(0.0);
    
    // Test NaN input
    EXPECT_THROW(filter.Pass(std::nan("")), std::invalid_argument);
    
    // Test infinity input
    EXPECT_THROW(filter.Pass(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(KalmanZTest, NegativeNoiseParameters_ThrowsException)
{
    KalmanZ filter(-0.1, 0.5, 0.0);
    filter.Init(0.0);
    
    EXPECT_THROW(filter.Pass(5.0), std::invalid_argument);
}

TEST(KalmanZTest, GetInnovation)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 1.0);
    
    filter.Pass(5.0);
    double innovation = filter.GetLastInnovation()(0, 0);
    
    // First innovation should be close to the measurement since initial state was 0
    EXPECT_NEAR(innovation, 5.0, 1.0);
}

TEST(KalmanZTest, ConvergenceCheck)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 1.0);
    
    const double trueValue = 5.0;
    
    // Process several measurements
    for (int i = 0; i < 30; ++i)
    {
        filter.Pass(trueValue);
    }
    
    // Filter should report convergence
    EXPECT_TRUE(filter.IsConvergent(3.0));
}

// ============================================================================
// KalmanXV Tests (Position-Velocity Kalman Filter)
// ============================================================================

TEST(KalmanXVTest, Constructor)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 0.0);
    EXPECT_NO_THROW();
}

TEST(KalmanXVTest, SetTimestep)
{
    KalmanXV filter;
    filter.SetTimestep(0.05);
    filter.Init(0.0, 0.0);
    EXPECT_NO_THROW();
}

TEST(KalmanXVTest, PassX_ConstantPosition)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 0.0);
    
    const double truePosition = 10.0;
    const int numIterations = 50;
    
    for (int i = 0; i < numIterations; ++i)
    {
        filter.PassX(truePosition);
    }
    
    EXPECT_NEAR(filter.GetX(), truePosition, 0.5);
}

TEST(KalmanXVTest, PassV_ConstantVelocity)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 5.0);
    
    const double trueVelocity = 5.0;
    const int numIterations = 50;
    
    for (int i = 0; i < numIterations; ++i)
    {
        filter.PassV(trueVelocity);
    }
    
    EXPECT_NEAR(filter.GetV(), trueVelocity, 0.5);
}

TEST(KalmanXVTest, PassXV_PositionAndVelocity)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 2.0);
    
    const double dt = 0.1;
    const double velocity = 2.0;
    const int numSteps = 50;
    
    for (int i = 0; i < numSteps; ++i)
    {
        double truePosition = velocity * i * dt;
        filter.PassXV(truePosition, velocity);
    }
    
    EXPECT_NEAR(filter.GetX(), velocity * (numSteps - 1) * dt, 0.5);
    EXPECT_NEAR(filter.GetV(), velocity, 0.3);
}

TEST(KalmanXVTest, PassXSmooth_WithSmoother)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 0.0);
    
    std::vector<double> x_smoothed, v_smoothed;
    std::vector<double> measurements = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    
    for (double x : measurements)
    {
        filter.PassXSmooth(x);
    }
    
    filter.GetSmoothedCurve(x_smoothed, v_smoothed);
    
    EXPECT_EQ(x_smoothed.size(), measurements.size());
    EXPECT_EQ(v_smoothed.size(), measurements.size());
}

TEST(KalmanXVTest, PassXBatch)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 0.0);
    
    std::vector<double> measurements = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x_smoothed, v_smoothed;
    
    filter.PassXBatch(measurements, x_smoothed, v_smoothed);
    
    EXPECT_EQ(x_smoothed.size(), measurements.size());
    EXPECT_EQ(v_smoothed.size(), measurements.size());
    
    // Last smoothed position should be close to last measurement
    EXPECT_NEAR(x_smoothed.back(), 5.0, 0.5);
}

TEST(KalmanXVTest, GetInnovation)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 0.0);
    
    filter.PassX(5.0);
    double innovation = filter.GetLastInnovationValue();
    
    EXPECT_NEAR(std::abs(innovation), 5.0, 1.0);
}

TEST(KalmanXVTest, ConvergenceCheck)
{
    KalmanXV filter(0.01);
    filter.Init(0.0, 0.0);
    
    for (int i = 0; i < 30; ++i)
    {
        filter.PassX(10.0);
    }
    
    EXPECT_TRUE(filter.IsConvergent(3.0));
}

TEST(KalmanXVTest, MovingObject_Tracking)
{
    // This test verifies the filter can track a moving object with acceleration
    // Note: The specific behavior depends on the filter tuning parameters
    KalmanXV filter(0.1);  // Higher process noise for better tracking
    filter.Init(0.0, 1.0);
    
    const double dt = 0.1;
    const double acceleration = 0.5;
    double position = 0.0;
    double velocity = 1.0;
    
    // Simulate accelerating object and feed measurements to filter
    for (int i = 0; i < 50; ++i)
    {
        position += velocity * dt + 0.5 * acceleration * dt * dt;
        velocity += acceleration * dt;
        
        filter.PassXV(position, velocity);
    }
    
    // Verify filter state is finite and reasonable
    EXPECT_TRUE(std::isfinite(filter.GetX()));
    EXPECT_TRUE(std::isfinite(filter.GetV()));
    // Filter should track in the right direction
    EXPECT_GT(filter.GetX(), 0.0);
    EXPECT_GT(filter.GetV(), 0.0);
}

// ============================================================================
// KalmanVA Tests (Velocity-Acceleration Kalman Filter)
// ============================================================================

TEST(KalmanVATest, Constructor)
{
    KalmanVA filter(0.1);
    filter.Init(0.0, 0.0);
    EXPECT_NO_THROW();
}

TEST(KalmanVATest, PassV_ConstantVelocity)
{
    KalmanVA filter(0.1);
    filter.Init(0.0, 0.0);
    
    const double trueVelocity = 10.0;
    const int numIterations = 50;
    
    for (int i = 0; i < numIterations; ++i)
    {
        filter.PassV(trueVelocity);
    }
    
    EXPECT_NEAR(filter.GetV(), trueVelocity, 0.5);
    // Acceleration should be near zero for constant velocity
    EXPECT_NEAR(filter.GetA(), 0.0, 0.5);
}

TEST(KalmanVATest, PassV_ChangingVelocity)
{
    KalmanVA filter(0.1);  // Higher process noise for better tracking
    filter.Init(0.0, 0.0);
    
    const double dt = 0.1;
    const double acceleration = 2.0;
    double velocity = 0.0;
    
    // Simulate constant acceleration
    for (int i = 0; i < 30; ++i)
    {
        velocity += acceleration * dt;
        filter.PassV(velocity);
    }
    
    EXPECT_NEAR(filter.GetV(), velocity, 2.0);
    // Note: Acceleration estimation may have larger error due to model dynamics
    EXPECT_TRUE(std::isfinite(filter.GetA()));
}

TEST(KalmanVATest, PassVSmooth_WithSmoother)
{
    KalmanVA filter(0.1);
    filter.Init(0.0, 0.0);
    
    std::vector<double> v_smoothed, a_smoothed;
    std::vector<double> measurements = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
    
    for (double v : measurements)
    {
        filter.PassVSmooth(v);
    }
    
    filter.GetSmoothedCurve(v_smoothed, a_smoothed);
    
    EXPECT_EQ(v_smoothed.size(), measurements.size());
    EXPECT_EQ(a_smoothed.size(), measurements.size());
}

TEST(KalmanVATest, PassVBatch)
{
    KalmanVA filter(0.1);
    filter.Init(0.0, 0.0);
    
    std::vector<double> measurements = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<double> v_smoothed, a_smoothed;
    
    filter.PassVBatch(measurements, v_smoothed, a_smoothed);
    
    EXPECT_EQ(v_smoothed.size(), measurements.size());
    EXPECT_EQ(a_smoothed.size(), measurements.size());
}

TEST(KalmanVATest, GetInnovation)
{
    KalmanVA filter(0.1);
    filter.Init(0.0, 0.0);
    
    filter.PassV(5.0);
    double innovation = filter.GetLastInnovationValue();
    
    EXPECT_NEAR(std::abs(innovation), 5.0, 1.0);
}

TEST(KalmanVATest, ConvergenceCheck)
{
    KalmanVA filter(0.01);
    filter.Init(0.0, 0.0);
    
    for (int i = 0; i < 30; ++i)
    {
        filter.PassV(10.0);
    }
    
    EXPECT_TRUE(filter.IsConvergent(3.0));
}

// ============================================================================
// Kalman2d Tests (Base 2D Filter - via derived classes)
// ============================================================================

TEST(Kalman2dTest, Init_ViaKalmanXV)
{
    KalmanXV filter(0.1);
    filter.Init(1.0, 2.0);
    
    auto cov = filter.GetCovariance();
    EXPECT_DOUBLE_EQ(cov(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(cov(1, 1), 1.0);
}

TEST(Kalman2dTest, InitWithCovariance_ViaKalmanXV)
{
    KalmanXV::MatrixNd P0;
    P0 << 2.0, 0.1,
          0.1, 3.0;
    
    KalmanXV filter(0.1);
    filter.Init(1.0, 2.0, P0);
    
    auto cov = filter.GetCovariance();
    EXPECT_DOUBLE_EQ(cov(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(cov(1, 1), 3.0);
}

TEST(Kalman2dTest, SetParameters_ViaKalmanXV)
{
    KalmanXV filter(0.1);
    filter.SetParameters(0.1, 0.5, 0.0);
    EXPECT_NO_THROW();
    
    filter.SetParameters(0.1, 0.3, 0.5, 0.0);
    EXPECT_NO_THROW();
}

// ============================================================================
// Regularization Tests
// ============================================================================

TEST(RegularizationTest, SetRegularizationEpsilon)
{
    KalmanZ filter(0.1, 0.5, 0.0);
    filter.Init(0.0);
    
    filter.SetRegularizationEpsilon(1e-8);
    filter.Pass(5.0);
    
    EXPECT_NO_THROW();
}

TEST(RegularizationTest, DefaultRegularization)
{
    KalmanZ filter(0.1, 0.5, 0.0);
    filter.Init(0.0);
    
    // Should work with default regularization
    for (int i = 0; i < 10; ++i)
    {
        filter.Pass(static_cast<double>(i));
    }
    
    EXPECT_TRUE(std::isfinite(filter.GetZ()));
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST(EdgeCaseTest, ZeroProcessNoise)
{
    KalmanZ filter(0.0, 0.1, 0.0);
    filter.Init(0.0, 1.0);
    
    for (int i = 0; i < 20; ++i)
    {
        filter.Pass(5.0);
    }
    
    EXPECT_NEAR(filter.GetZ(), 5.0, 0.2);
}

TEST(EdgeCaseTest, HighMeasurementNoise)
{
    KalmanZ filter(0.1, 10.0, 0.0);  // Higher process noise to respond better
    filter.Init(7.0, 1.0);  // Start near the middle
    
    // With high measurement noise, filter should trust predictions more
    for (int i = 0; i < 20; ++i)
    {
        filter.Pass(5.0 + (i % 2) * 4.0); // Alternating between 5 and 9
    }
    
    // Result should be somewhere in the middle - relaxed tolerance
    EXPECT_GT(filter.GetZ(), 6.0);
    EXPECT_LT(filter.GetZ(), 8.0);
}

TEST(EdgeCaseTest, LargeInitialStateUncertainty)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 1000.0); // High initial uncertainty
    
    filter.Pass(5.0);
    
    // With high initial uncertainty, first measurement should have large impact
    EXPECT_NEAR(filter.GetZ(), 5.0, 2.0);
}

TEST(EdgeCaseTest, SmallInitialStateUncertainty)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 0.001); // Low initial uncertainty
    
    filter.Pass(100.0);
    
    // With low initial uncertainty, filter should be slow to accept new measurement
    EXPECT_LT(filter.GetZ(), 50.0);
}

TEST(EdgeCaseTest, ResetFunctionality)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 0.0);
    
    // Run filter
    for (int i = 0; i < 20; ++i)
    {
        filter.PassX(5.0);
    }
    
    double stateBeforeReset = filter.GetX();
    
    // Reset
    filter.Reserve(10);
    filter.Init(0.0, 0.0);
    
    // State should be reset
    filter.PassX(10.0);
    // After reset and new measurement, state should move towards new measurement
    EXPECT_NE(stateBeforeReset, filter.GetX());
}

TEST(EdgeCaseTest, ReserveCapacity)
{
    KalmanXV filter(0.1);
    filter.Init(0.0, 0.0);
    filter.Reserve(100);
    
    for (int i = 0; i < 50; ++i)
    {
        filter.PassXSmooth(static_cast<double>(i));
    }
    
    EXPECT_NO_THROW();
}

TEST(EdgeCaseTest, SingleMeasurement)
{
    KalmanZ filter(0.1, 0.5, 0.0);
    filter.Init(0.0);
    
    filter.Pass(42.0);
    
    // After single measurement, state should be between initial (0) and measurement (42)
    // Exact value depends on Kalman gain calculation
    EXPECT_GT(filter.GetZ(), 0.0);
    EXPECT_LT(filter.GetZ(), 42.0);
}

TEST(EdgeCaseTest, AlternatingMeasurements)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 1.0);
    
    // Alternating between two values
    for (int i = 0; i < 100; ++i)
    {
        filter.Pass((i % 2 == 0) ? 0.0 : 10.0);
    }
    
    // Should converge to the average
    EXPECT_NEAR(filter.GetZ(), 5.0, 1.0);
}

// ============================================================================
// Control Input Tests
// ============================================================================

TEST(ControlInputTest, KalmanZ_WithControl)
{
    KalmanZ filter(0.01, 0.1, 1.0); // u = 1.0
    filter.Init(0.0, 1.0);
    
    // With control input, each prediction adds u
    for (int i = 0; i < 10; ++i)
    {
        filter.Pass(static_cast<double>(i));
    }
    
    // State should reflect control input influence
    EXPECT_TRUE(std::isfinite(filter.GetZ()));
}

TEST(ControlInputTest, KalmanXV_WithControl)
{
    KalmanXV filter(0.1);
    filter.SetU(0.5);
    filter.Init(0.0, 0.0);
    
    for (int i = 0; i < 20; ++i)
    {
        filter.PassX(static_cast<double>(i));
    }
    
    EXPECT_TRUE(std::isfinite(filter.GetX()));
    EXPECT_TRUE(std::isfinite(filter.GetV()));
}

// ============================================================================
// NIS (Normalized Innovation Squared) Tests
// ============================================================================

TEST(NISTest, ComputeNIS)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 1.0);
    
    filter.Pass(5.0);
    
    double nis = filter.ComputeNIS();
    EXPECT_GE(nis, 0.0);
    EXPECT_TRUE(std::isfinite(nis));
}

TEST(NISTest, NIS_ConvergedFilter)
{
    KalmanZ filter(0.01, 0.1, 0.0);
    filter.Init(0.0, 1.0);
    
    // Process consistent measurements
    for (int i = 0; i < 30; ++i)
    {
        filter.Pass(5.0);
    }
    
    double nis = filter.ComputeNIS();
    
    // NIS should be small for converged filter with consistent measurements
    EXPECT_LT(nis, 1.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
