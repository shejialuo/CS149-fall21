CXX=g++ -m64 -fPIE
CXXFLAGS=-I../common -Iobjs/ -O3 -Wall
LDLIBS=-lm $(TASKSYS_LIB)
ISPC=ispc
# note: requires AVX2
ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64 --pic

APP_NAME=gemm
OBJDIR=objs
COMMONDIR=../common

TASKSYS_CXX=$(COMMONDIR)/tasksys.cpp
TASKSYS_LIB=-lpthread
TASKSYS_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(TASKSYS_CXX:.cpp=.o)))

MKL_CXX=-DMKL_ILP64 -I$(MKLROOT)/include
MKL_LIB=-Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a \
	$(MKLROOT)/lib/intel64/libmkl_gnu_thread.a \
	$(MKLROOT)/lib/intel64/libmkl_core.a \
	-Wl,--end-group -lgomp -lpthread -lm -ldl

# If MKLROOT is defined, we use MKL. Otherwise, the student hasn't
# installed MKL yet so we try to compile a version just using their
# solution and the reference ISPC solution.
ifdef MKLROOT
	CXXFLAGS += -DMKL_INSTALLED=1
	CXXFLAGS += $(MKL_CXX)
	LDLIBS += -lm $(MKL_LIB)
else
	CXXFLAGS += -DMKL_INSTALLED=0
endif

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/gemm.o $(OBJDIR)/gemm_ispc.o $(TASKSYS_OBJ) 

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) ref_gemm_ispc.a $(LDLIBS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(OBJDIR)/$(APP_NAME)_ispc.h $(COMMONDIR)/CycleTimer.h

$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: %.ispc
		$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h
