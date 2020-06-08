# Comp 4510 - Sample CUDA Makefile

###############################################
# Change these files names to match your own: #
###############################################

# name of the executable to be generated
PROG = reduce
INCLUDE_DIR = include
SRC_DIR = src
# space-delimited list of header files
HDR_FILES = kernels.h reduce.h timer.h utils.h perf.h
# space-delimited list of source files
SRC_FILES = reduce.cu kernel.cu timer.cu utils.cu perf.cu

# Tack on the appropriate dir name (cut down on the amount of typing required above)
HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(HDR_FILES))
SRCS = $(patsubst %.cu, $(SRC_DIR)/%.cu, $(SRC_FILES))

#######################
# Don't change these: #
#######################

# directory to store object files
OBJDIR = object
# names of object files
OBJS = $(patsubst %.cu, $(OBJDIR)/%.o, $(SRCS))

# name of the compiler
CC = nvcc
# additional compiler flags to pass in
CFLAGS = -L. -I$(INCLUDE_DIR)
# libraries for the linker
LIBS = -lm

####################
# Compiling rules: #
####################
# WARNING: *must* have a tab before each definition

# invoked when "make" is run
all : $(OBJDIR) $(PROG)

# links object files into executable
$(PROG) : $(OBJS)
	$(CC) $(CFLAGS) $(patsubst %.o, $(OBJDIR)/%.o, $(notdir $^)) -o $(PROG) $(LIBS)

# compiles source files into object files
object/%.o : %.cu $(HDRS)
	$(CC) -c $(CFLAGS) $< -o $(OBJDIR)/$(notdir $@) $(LIBS)

# creates directory to store object files
$(OBJDIR) :
	mkdir -p $@/

# cleans up object files and executable
# type "make clean" to use
# Note: you can add your own commands to remove other things (e.g. output files)
clean:
	rm -rf object/
	rm -f $(PROG)
