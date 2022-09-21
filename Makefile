
TARGET_EXEC := bubbleSim.exe

BUILD_DIR := ./build
SRC_DIR := ./bubbleSim

SRCS := $(SRC_DIR)/bubble.cpp $(SRC_DIR)/openclwrapper.cpp $(SRC_DIR)/simulation.cpp $(SRC_DIR)/source.cpp $(SRC_DIR)/datastreamer.cpp
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

#CXX=clang++
CXX=g++

#link against common OCL headers
CXXFLAGS=-std=c++2a -I./dependencies/json/
LDFLAGS=-L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lm -lstdc++

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
