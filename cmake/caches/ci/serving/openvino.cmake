set(BUILD_SHARED_LIBS YES CACHE BOOL "")
# uncomment only if you know what you are doing
# set(CMAKE_LINK_DEPENDS_NO_SHARED YES CACHE BOOL "")
set(BUILD_CUDA NO CACHE BOOL "")
set(WITH_OPENVINO ON CACHE BOOL "")
set(BUILD_CPP_API ON CACHE BOOL "")
set(BUILD_GIT_VERSION NO CACHE BOOL "")
set(TREAT_WARNINGS_AS_ERRORS YES CACHE BOOL "")
set(BUILD_HWLOC NO CACHE BOOL "")
set(BUILD_TESTING ON CACHE BOOL "")
set(WITH_MLIR YES CACHE BOOL "")
set(THIRD_PARTY_MIRROR aliyun CACHE STRING "")
set(PIP_INDEX_MIRROR "https://pypi.tuna.tsinghua.edu.cn/simple" CACHE STRING "")
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_GENERATOR Ninja CACHE STRING "")
set(CMAKE_C_COMPILER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "")
set(CPU_THREADING_RUNTIME SEQ CACHE STRING "")
set(BUILD_HWLOC OFF CACHE BOOL "")
set(WITH_ONEDNN OFF CACHE BOOL "")
set(OPENVINO_ROOT "/opt/intel/openvino" CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE STRING "")
