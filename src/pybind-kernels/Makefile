
PYBIND_FLAGS += $(shell python3 -m pybind11 --include) -O3 -Wall -shared -fPIC -fopenmp -g
PYBIND_SUFFIX = $(shell python3-config --extension-suffix)

OPENCV_INCLUDE=$(shell pkg-config opencv4 --cflags)
OPENCV_LIB=$(shell pkg-config opencv4 --libs)


# Detect if OpenACC can be used
ifneq (, $(shell which nvc++))
CXX = nvc++
CXXFLAGS += -acc=gpu -Minfo=accel
else
$(info OpenACC compiler nvc++ not found. Compiling without)
endif


histograms$(PYBIND_SUFFIX): histograms.cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $< -o histograms$(PYBIND_SUFFIX)

geometry$(PYBIND_SUFFIX): opencv_pybind.cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) $< -o opencv_pybind$(PYBIND_SUFFIX)


opencv_pybind$(PYBIND_SUFFIX): opencv_pybind.cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) $< -o opencv_pybind$(PYBIND_SUFFIX)

opencv_tester: opencv_tester.cc
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) $< -o opencv_tester

all: histograms$(PYBIND_SUFFIX) geometry$(PYBIND_SUFFIX) opencv_pybind$(PYBIND_SUFFIX) opencv_tester

clean:
	rm -f histograms.o histograms$(PYBIND_SUFFIX) opencv_pybind$(PYBIND_SUFFIX) opencv_tester



