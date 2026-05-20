CXX      := g++
CXXFLAGS := -std=c++17 -O1 -Wall -Wextra
LDFLAGS  := -Wl,--stack,268435456

SRCS := $(wildcard src/*.cpp)
OBJS := $(SRCS:.cpp=.o)
BIN  := ctg

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(BIN) $(BIN).exe

.PHONY: clean
