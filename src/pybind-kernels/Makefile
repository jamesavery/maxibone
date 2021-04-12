CXXFLAGS += $(shell /usr/bin/python3 -m pybind11 --include) -O3 -Wall -shared -fPIC -fopenmp -g
SUFFIX = $(shell /usr/bin/python3-config --extension-suffix)

# Detect if OpenACC can be used
ifneq (, $(shell which nvc++))
CXX = nvc++
CXXFLAGS += -acc=gpu -Minfo=accel
else
$(info OpenACC compiler nvc++ not found. Compiling without)
endif

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

histograms$(SUFFIX): histograms.o
	$(CXX) $(CXXFLAGS) $< -o histograms$(SUFFIX)

all: histograms$(SUFFIX)

clean:
	rm histograms.o histograms$(SUFFIX)