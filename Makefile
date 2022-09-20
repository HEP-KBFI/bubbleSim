
TARGET_EXEC := bubbleSim.exe

BUILD_DIR := ./build
SRC_DIR := ./bubbleSim

SRCS := $(SRC_DIR)/bubble.cpp $(SRC_DIR)/openclwrapper.cpp $(SRC_DIR)/simulation.cpp $(SRC_DIR)/source.cpp $(SRC_DIR)/datastreamer.cpp
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

CXX=g++

#link against common OCL headers
CXXFLAGS=-fpermissive -std=c++2a -I./dependencies/json/
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
