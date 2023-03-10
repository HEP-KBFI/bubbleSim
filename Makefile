
TARGET_EXEC := bubbleSim.exe

BUILD_DIR := ./build
SRC_DIR := ./bubbleSim

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

ifndef OPTFLAGS
  OPTFLAGS := -O3
endif

#link against common OCL headers
CXXFLAGS=-std=c++2a -I./dependencies/json/ $(OPTFLAGS)
LDFLAGS=-lOpenCL -lm -lstdc++ $(OPTFLAGS)

# The final build step of the executable
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

format:
	clang-format -i --style=Google $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h

clean:
	rm -Rf build


.PHONY: format clean
