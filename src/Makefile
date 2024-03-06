# Define constants and collections
PYTHON = python3.10
PYBIND_FLAGS = $(shell $(PYTHON) -m pybind11 --include)
CXXFLAGS += $(subst -I,-isystem ,$(PYBIND_FLAGS)) # We don't care about warnings from the python headers
PYBIND_SUFFIX = $(shell $(PYTHON)-config --extension-suffix)

# Detect OS for OS specific changes
ifeq ($(shell uname -s), Darwin) # Mac OSX
CXX = g++-13 # Use homebrew gcc, as system gcc is an alias for clang
CXXFLAGS += -undefined dynamic_lookup # https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually
CLEANUP += $(TARGETS) $(foreach TARGET, $(TARGETS), $(TARGET).dSYM) # These are also generated on Mac
endif

CPP_FOLDER=lib/cpp
#CXXFLAGS += -I../contrib/cpptqdm/ -Iinclude
CXXFLAGS += -I$(CPP_FOLDER)/include -march=native -Wall -Wextra -Wfloat-equal -Wundef -Wshadow -Wuninitialized -Winit-self -shared -fPIC -g -std=c++17 -O3
PLATFORMS=cpu_seq cpu
cpu_seq_CXX=$(CXX)
cpu_seq_FLAGS=-Wno-unknown-pragmas -Wno-comment -Wconversion #-Weffc++
cpu_CXX=$(cpu_seq_CXX)
cpu_FLAGS=$(cpu_seq_FLAGS) -fopenmp

LIBS=io geometry morphology diffusion
TARGETS = $(foreach PLATFORM, $(PLATFORMS), $(foreach LIB, $(LIBS), $(CPP_FOLDER)/$(PLATFORM)/$(LIB)$(PYBIND_SUFFIX)))
CLEANUP = $(TARGETS) $(foreach PLATFORM, $(PLATFORMS), $(CPP_FOLDER)/$(PLATFORM)/__pycache__)

# Detect if OpenACC can be used
ifneq (, $(shell which nvc++))
PLATFORMS += gpu
gpu_CXX = nvc++
gpu_FLAGS = -acc=gpu -tp=native #-Minfo=accel
gpu_FLAGS += -Xcudafe --display_error_number # Getting the warning codes for later suppression
gpu_FLAGS += --diag_suppress 1626 # Remove the annoying pybind warning that routine is both inline and noinline
gpu_FLAGS += --diag_suppress 9 # Remove the warning about nested comments
gpu_FLAGS += -Wnvlink,-w # Disable nvlink warnings
else
$(info OpenACC compiler nvc++ not found. Compiling without.)
endif

all: $(TARGETS) pip_install

define GEN_RULE
$(CPP_FOLDER)/$(PLATFORM)/$(LIB)$(PYBIND_SUFFIX): pybind/$(LIB)-pybind.cc $(CPP_FOLDER)/$(PLATFORM)/$(LIB).cc $(CPP_FOLDER)/include/*.hh
	$($(PLATFORM)_CXX) $($(PLATFORM)_FLAGS) $(CXXFLAGS) -I$(CPP_FOLDER)/$(PLATFORM) $$< -o $(CPP_FOLDER)/$(PLATFORM)/$(LIB)$(PYBIND_SUFFIX)
endef

$(foreach PLATFORM, $(PLATFORMS), \
	$(foreach LIB, $(LIBS), \
		$(eval $(GEN_RULE)) \
	) \
)

# TODO lightweight openacc profiling can be done with the environment variable NV_ACC_TIME=1 !!!

pip_install:
	$(PYTHON) -m pip install -r requirements.txt

test: all
	$(PYTHON) -m pytest -n auto test

test_%: test/test_%.py all
	$(PYTHON) -m pytest -n auto $<

clean:
	rm -rf $(CLEANUP) __pycache__ test/__pycache__ .pytest_cache lib/cpp/**/*.so
