cmake_minimum_required(VERSION 3.30)
project(LRE_Axel)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 20)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -lineinfo -g -G --expt-relaxed-constexpr -rdc=true -Wno-deprecated-gpu-targets")

include_directories(include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(SOURCES
        src/max_tree.cu
        src/max_tree_c.cpp
        src/utils.cpp
        src/tilling.cu 
        src/tilling_c.cpp
        src/attribute_compute_c.cpp
        src/attribute_compute.cu
        src/area_filtering_c.cpp
        src/area_filtering.cu
        main.cpp
)

set(HEADERS
        include/max_tree.cuh
        include/max_tree_c.hpp
        include/tilling.cuh 
        include/tilling_c.hpp
        include/attribute_compute.cuh
        include/attribute_compute_c.hpp
        include/area_filtering.cuh
        include/area_filtering_c.hpp
        include/utils.hpp
        include/vector2D.cuh
)

add_executable(LRE_Axel ${SOURCES} ${HEADERS})

set_target_properties(LRE_Axel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(LRE_Axel PRIVATE CUDA::cudart ${OpenCV_LIBS})

target_compile_options(LRE_Axel PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -rdc=true -g -G -Wno-deprecated-gpu-targets>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
)
