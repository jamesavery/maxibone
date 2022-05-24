
PYBIND_FLAGS += $(shell python3 -m pybind11 --include) -O3 -march=native -mtune=native -Wall -shared -fPIC -fopenmp -g -std=c++17
PYBIND_SUFFIX = $(shell python3-config --extension-suffix)

OPENCV_INCLUDE=$(shell pkg-config opencv4 --cflags)
OPENCV_LIB=$(shell pkg-config opencv4 --libs)


# Detect if OpenACC can be used
ifneq (, $(shell which nvc++))
	CXX = nvc++
	CXXFLAGS += -acc=gpu -Minfo=accel -tp=native
else
	$(info OpenACC compiler nvc++ not found. Compiling without)
endif

all: histograms$(PYBIND_SUFFIX) geometry$(PYBIND_SUFFIX) label$(PYBIND_SUFFIX)

histograms$(PYBIND_SUFFIX): histograms.cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $< -o histograms$(PYBIND_SUFFIX)

geometry$(PYBIND_SUFFIX): geometry-pybind.cc geometry.cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $< -o geometry$(PYBIND_SUFFIX)

label$(PYBIND_SUFFIX): label.cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $< -o label$(PYBIND_SUFFIX)

opencv_pybind$(PYBIND_SUFFIX): opencv_pybind.cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) $< -o opencv_pybind$(PYBIND_SUFFIX)

opencv_tester: opencv_tester.cc
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) $< -o opencv_tester

clean:
	rm -f histograms.o histograms$(PYBIND_SUFFIX) opencv_pybind$(PYBIND_SUFFIX) opencv_tester



