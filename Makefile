BIN_NAME = main
CXX = $(PREP) mpic++
CXXFLAGS += -O3 -std=c++11 -Wno-unused-parameter -framework OpenCL
ifdef DEBUG
CXXFLAGS += -ggdb3 
endif
ifdef PROFILING
CXXFLAGS += -DPROFILING
endif
ifdef VERIFICATION
CXXFLAGS += -DVERIFICATION
endif


SRC_DIR = ./src
BIN_DIR = ./bin
BUILD_DIR = ./build
LOG_DIR = ./logs
INC_DIR = ./include
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(patsubst src/%.cpp,build/%.o,$(SRC))
DEPS = $(patsubst build/%.o,build/%.d,$(OBJ))
BIN = $(BIN_NAME)

vpath %.cpp $(SRC_DIR)




.DEFAULT_GOAL = all


$(BUILD_DIR)/%.o: %.cpp
	$(info "Compiling objects")
	@$(CXX) -c $< -o $@ -lm $(CXXFLAGS) -I$(INC_DIR)

$(BIN_DIR)/$(BIN_NAME): $(DEPS) $(OBJ)
	$(info Linking objects)
	@$(CXX) -o $@ $(OBJ) -lm $(CXXFLAGS) -I$(INC_DIR)
	@echo "Compiling with $(CXXFLAGS)"
	@echo "Done!"

checkdirs:
	@echo "Checking directories"
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(SRC_DIR)
	@mkdir -p $(LOG_DIR)

all: checkdirs $(BIN_DIR)/$(BIN_NAME)

clean:
	@rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*
	$(info Cleaning build and bin directories)