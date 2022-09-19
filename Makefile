
TARGET_EXEC := bubbleSim.exe

BUILD_DIR := ./build
SRC_DIR := ./bubbleSim

SRCS := $(SRC_DIR)/opencl.cpp $(SRC_DIR)/simulation.cpp $(SRC_DIR)/source.cpp $(SRC_DIR)/stream_data.cpp
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)


#link against common OCL implementation
CXXFLAGS=-fpermissive -std=c++2a
LDFLAGS=-L/usr/lib/x86_64-linux-gnu/ -lOpenCL

# The final build step of the executable
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -Rf build
