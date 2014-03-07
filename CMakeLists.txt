#---------------------------------------------------
# OpenMM Example Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")

INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(EXAMPLE_PLUGIN_SOURCE_SUBDIRS . openmmapi serialization)

SET(EXAMPLE_LIBRARY_NAME ExamplePlugin)
SET(SHARED_EXAMPLE_TARGET ${EXAMPLE_LIBRARY_NAME})

# But on Unix or Cygwin we have to add the suffix manually
IF (UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)
    SET(SHARED_EXAMPLE_TARGET ${SHARED_EXAMPLE_TARGET}_d)
ENDIF (UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)

# These are all the places to search for header files which are
# to be part of the API.
SET(API_EXAMPLE_INCLUDE_DIRS) # start empty
FOREACH(subdir ${EXAMPLE_PLUGIN_SOURCE_SUBDIRS})
    # append
    SET(API_EXAMPLE_INCLUDE_DIRS ${API_EXAMPLE_INCLUDE_DIRS}
                                     ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include
                                     ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include/internal)
ENDFOREACH(subdir)

# We'll need both *relative* path names, starting with their API_INCLUDE_DIRS,
# and absolute pathnames.
SET(API_EXAMPLE_REL_INCLUDE_FILES)   # start these out empty
SET(API_EXAMPLE_ABS_INCLUDE_FILES)

FOREACH(dir ${API_EXAMPLE_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)	# returns full pathnames
    SET(API_EXAMPLE_ABS_INCLUDE_FILES ${API_EXAMPLE_ABS_INCLUDE_FILES} ${fullpaths})

    FOREACH(pathname ${fullpaths})
        GET_FILENAME_COMPONENT(filename ${pathname} NAME)
        SET(API_EXAMPLE_REL_INCLUDE_FILES ${API_EXAMPLE_REL_INCLUDE_FILES} ${dir}/${filename})
    ENDFOREACH(pathname)
ENDFOREACH(dir)

# collect up source files
SET(SOURCE_EXAMPLE_FILES) # empty
SET(SOURCE_EXAMPLE_INCLUDE_FILES)

FOREACH(subdir ${EXAMPLE_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_EXAMPLE_FILES         ${SOURCE_EXAMPLE_FILES}         ${src_files})   #append
    SET(SOURCE_EXAMPLE_INCLUDE_FILES ${SOURCE_EXAMPLE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# ----------------------------------------------------------------------------

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

ADD_LIBRARY(${SHARED_EXAMPLE_TARGET} SHARED ${SOURCE_EXAMPLE_FILES} ${SOURCE_EXAMPLE_INCLUDE_FILES} ${API_EXAMPLE_ABS_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_EXAMPLE_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DEXAMPLE_BUILDING_SHARED_LIBRARY -DLEPTON_BUILDING_SHARED_LIBRARY")

FILE(GLOB serialization_files  ${CMAKE_CUURENT_SOURCE_DIR}/serialization/src/*.cpp)
SET_SOURCE_FILES_PROPERTIES(${serialization_files} PROPERTIES COMPILE_FLAGS "-DEXAMPLE_BUILDING_SHARED_LIBRARY -DTIXML_USE_STL")

# ----------------------------------------------------------------------------

# On Linux need to link to libdl
FIND_LIBRARY(DL_LIBRARY dl)
IF(DL_LIBRARY)
  TARGET_LINK_LIBRARIES(${SHARED_EXAMPLE_TARGET} ${DL_LIBRARY})
  IF( CREATE_SERIALIZABLE_EXAMPLE )
      TARGET_LINK_LIBRARIES(${SHARED_EXAMPLE_SERIALIZABLE_TARGET} ${DL_LIBRARY})
  ENDIF( CREATE_SERIALIZABLE_EXAMPLE )
ENDIF(DL_LIBRARY)

SET( OpenMMLib OpenMM )
# But on Unix or Cygwin we have to add the suffix manually
IF (UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)
    SET(OpenMMLib ${OpenMMLib}_d)
ENDIF (UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)

TARGET_LINK_LIBRARIES(${SHARED_EXAMPLE_TARGET} debug OpenMM_d optimized OpenMM)

# Enable testing

ENABLE_TESTING()
ADD_SUBDIRECTORY(serialization/tests)

# Which hardware platforms to build

ADD_SUBDIRECTORY(platforms/reference)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
FIND_PACKAGE(OpenCL QUIET)
IF(OPENCL_FOUND)
    SET(OPENMM_BUILD_EXAMPLE_OPENCL_LIB ON CACHE BOOL "Build Example implementation for OpenCL")
ELSE(OPENCL_FOUND)
    SET(OPENMM_BUILD_EXAMPLE_OPENCL_LIB OFF CACHE BOOL "Build Example implementation for OpenCL")
ENDIF(OPENCL_FOUND)
IF(OPENMM_BUILD_EXAMPLE_OPENCL_LIB)
    ADD_SUBDIRECTORY(platforms/opencl)
ENDIF(OPENMM_BUILD_EXAMPLE_OPENCL_LIB)

FIND_PACKAGE(CUDA QUIET)
IF(CUDA_FOUND)
    SET(OPENMM_BUILD_EXAMPLE_CUDA_LIB ON CACHE BOOL "Build Example implementation for CUDA")
ELSE(CUDA_FOUND)
    SET(OPENMM_BUILD_EXAMPLE_CUDA_LIB OFF CACHE BOOL "Build Example implementation for CUDA")
ENDIF(CUDA_FOUND)
IF(OPENMM_BUILD_EXAMPLE_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(OPENMM_BUILD_EXAMPLE_CUDA_LIB)

INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_EXAMPLE_TARGET})
FILE(GLOB CORE_HEADERS     include/*.h          */include/*.h)
FILE(GLOB TOP_HEADERS      include/openmm/*.h          */include/openmm/*.h)
FILE(GLOB INTERNAL_HEADERS include/openmm/internal/*.h */include/openmm/internal/*.h )
INSTALL_FILES(/include                 FILES ${CORE_HEADERS})
INSTALL_FILES(/include/openmm          FILES ${TOP_HEADERS})
INSTALL_FILES(/include/openmm/internal FILES ${INTERNAL_HEADERS})
