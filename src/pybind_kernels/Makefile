PYBIND_FLAGS += $(shell python3 -m pybind11 --include) -march=native -Wall -shared -fPIC -fopenmp -g -std=c++17
PYBIND_SUFFIX = $(shell python3-config --extension-suffix)

OPENCV_INCLUDE=$(shell pkg-config opencv4 --cflags)
OPENCV_LIB=$(shell pkg-config opencv4 --libs)

# Detect if OpenACC can be used
#ifneq (, $(shell which nvc++))
#CXX = nvc++
#CXXFLAGS += -acc=gpu -Minfo=accel -tp=native
#else
#$(info OpenACC compiler nvc++ not found. Compiling without)
#endif

CXXFLAGS += -I../contrib/cpptqdm/ -Iinclude

PLATFORMS=cpu_seq cpu gpu
LIBS=io morphology
TARGETS = $(foreach PLATFORM, $(PLATFORMS), $(foreach LIB, $(LIBS), $(PLATFORM)/$(LIB)$(PYBIND_SUFFIX)))

all: $(TARGETS)

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

define GEN_RULE
$(PLATFORM)/$(LIB)$(PYBIND_SUFFIX): pybind/$(LIB)-pybind.cc $(PLATFORM)/$(LIB).cc
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) -I$(PLATFORM) $$< -o $(PLATFORM)/$(LIB)$(PYBIND_SUFFIX)
endef

$(foreach PLATFORM, $(PLATFORMS), \
	$(foreach LIB, $(LIBS), \
		$(eval $(GEN_RULE)) \
	) \
)

test: all
	python3 -m pytest test

clean:
	rm -f $(TARGETS)
