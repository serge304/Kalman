# Makefile for Kalman Filter Unit Tests

CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -I.
LDFLAGS = -pthread -lgtest -lgtest_main

# Source files
KALMAN_SRC = Kalman.cpp
TEST_SRC = KalmanTest.cpp

# Output
TEST_EXEC = KalmanTest

.PHONY: all clean test run

all: $(TEST_EXEC)

$(TEST_EXEC): $(TEST_SRC) $(KALMAN_SRC)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) $(KALMAN_SRC) $(LDFLAGS) -o $(TEST_EXEC)

test: $(TEST_EXEC)
	./$(TEST_EXEC)

run: $(TEST_EXEC)
	./$(TEST_EXEC) --gtest_color=yes

clean:
	rm -f $(TEST_EXEC)

# Debug build with more verbose output
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TEST_EXEC)
	./$(TEST_EXEC) --gtest_break_on_failure

# Release build with optimizations
release: CXXFLAGS += -O3 -DNDEBUG
release: $(TEST_EXEC)
