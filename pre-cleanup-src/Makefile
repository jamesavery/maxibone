CXX=icpc
CXXFLAGS += -std=c++17
CXXFLAGS += $(shell pkg-config --cflags fmt)
CXXFLAGS += $(shell pkg-config --cflags hdf5)
LIBS += $(shell pkg-config --libs fmt)
LIBS += $(shell pkg-config --libs hdf5)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< 

h5-blockmap: h5-blockmap.o
	echo $(LIBS)
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

