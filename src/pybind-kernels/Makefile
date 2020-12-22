CXXFLAGS += $(shell python3 -m pybind11 --include) -O3 -Wall -shared -fPIC -fopenmp
SUFFIX = $(shell python3-config --extension-suffix)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

all: histograms.o
	$(CXX) $(CXXFLAGS) $< -o histograms$(SUFFIX)

