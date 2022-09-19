
TARGET_EXEC := final_program

BUILD_DIR := ./build
SRC_DIRS := ./bubbleSim

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

CXXFLAGS=-I/usr/local/cuda/include/ -fpermissive -std=c++2a
LDFLAGS=-L/usr/local/cuda/lib64/ -lOpenCL

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


