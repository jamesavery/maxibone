# Define constants and collections
PYTHON = python3.10
PYBIND_FLAGS += $(shell $(PYTHON) -m pybind11 --include) -march=native -Wall -shared -fPIC -fopenmp -g -std=c++17 -O3
PYBIND_SUFFIX = $(shell $(PYTHON)-config --extension-suffix)
CPP_FOLDER=lib/cpp
#CXXFLAGS += -I../contrib/cpptqdm/ -Iinclude
CXXFLAGS += -I$(CPP_FOLDER)/include
PLATFORMS=cpu_seq cpu gpu
LIBS=io geometry morphology
TARGETS = $(foreach PLATFORM, $(PLATFORMS), $(foreach LIB, $(LIBS), $(CPP_FOLDER)/$(PLATFORM)/$(LIB)$(PYBIND_SUFFIX)))
CLEANUP = $(TARGETS) $(foreach PLATFORM, $(PLATFORMS), $(CPP_FOLDER)/$(PLATFORM)/__pycache__)

# Detect if OpenACC can be used
ifneq (, $(shell which nvc++))
CXX = nvc++
CXXFLAGS += -acc=gpu -Minfo=accel -tp=native
else
$(info OpenACC compiler nvc++ not found. Compiling without.)
endif

# Detect OS for OS specific changes
ifeq ($(shell uname -s), Darwin) # Mac OSX
CXX = g++-12 # Use homebrew gcc, as system gcc is an alias for clang
CXXFLAGS += -undefined dynamic_lookup # https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually
CLEANUP += $(TARGETS) $(foreach TARGET, $(TARGETS), $(TARGET).dSYM) # These are also generated on Mac
endif

all: $(TARGETS)

define GEN_RULE
$(CPP_FOLDER)/$(PLATFORM)/$(LIB)$(PYBIND_SUFFIX): pybind/$(LIB)-pybind.cc $(CPP_FOLDER)/$(PLATFORM)/$(LIB).cc $(CPP_FOLDER)/include/*.hh
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) -I$(CPP_FOLDER)/$(PLATFORM) $$< -o $(CPP_FOLDER)/$(PLATFORM)/$(LIB)$(PYBIND_SUFFIX)
endef

$(foreach PLATFORM, $(PLATFORMS), \
	$(foreach LIB, $(LIBS), \
		$(eval $(GEN_RULE)) \
	) \
)

test: all
	$(PYTHON) -m pytest -n auto test

test_%: test/test_%.py all
	$(PYTHON) -m pytest -n auto $<

clean:
	rm -rf $(CLEANUP) __pycache__ test/__pycache__ .pytest_cache lib/cpp/**/*.so