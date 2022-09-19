
TARGET_EXEC := bubbleSim.exe

BUILD_DIR := ./build
SRC_DIR := ./bubbleSim

SRCS := $(SRC_DIR)/opencl.cpp $(SRC_DIR)/simulation.cpp $(SRC_DIR)/source.cpp $(SRC_DIR)/stream_data.cpp
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)


#link against POCL (CPU)
CXXFLAGS=-I/usr/include/ -fpermissive -std=c++2a
LDFLAGS=-lpocl

#link against Nvidia OpenCL (GPU)
#LDFLAGS=-L/usr/local/cuda/lib64/ -lOpenCL

# The final build step of the executable
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


