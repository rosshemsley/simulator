CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT (DT)
FIND_PACKAGE(CGAL REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE(${CGAL_USE_FILE})	

SET( CMAKE_CXX_FLAGS "-Wall -O3 -std=c++11" )

ADD_EXECUTABLE(
    simulate 
    main.cpp 
    simulator.h 
)

TARGET_LINK_LIBRARIES( simulate ${CGAL_LIBRARIES})
