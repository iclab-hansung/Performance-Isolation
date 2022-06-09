cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(thpool-test)

set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/perfisol/libtorch )
# set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/perfisol/libtorch)
find_package(Torch REQUIRED)

set(COMMON_DIR /home/kmsjames/very-big-storage/hojin/pytorch/perfisol/exe_time/include)

file(GLOB SOURCES "src/*")
#set(SOURCES src/multithreadtest.cpp) #manual add

include_directories(${COMMON_DIR})

add_executable(thpool-test ${SOURCES})
target_link_libraries(thpool-test "${TORCH_LIBRARIES}")
set_property(TARGET thpool-test PROPERTY CXX_STANDARD 14)
