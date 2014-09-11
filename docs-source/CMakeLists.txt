#
# Build and install API documentation
#

find_package(Doxygen QUIET)
mark_as_advanced(CLEAR DOXYGEN_EXECUTABLE)
IF(DOXYGEN_EXECUTABLE)
    # Generate C++ API documentation

    SET(DOXY_CONFIG_C++ "${CMAKE_BINARY_DIR}/DoxyfileC++")
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/DoxyfileC++.in 
          ${DOXY_CONFIG_C++}
          @ONLY )
    FILE(GLOB_RECURSE OPENMM_INCLUDES "openmmapi/include/*.h")
    ADD_CUSTOM_COMMAND(
        OUTPUT "${CMAKE_BINARY_DIR}/meld-api-c++/index.html"
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXY_CONFIG_C++}
        DEPENDS ${OPENMM_INCLUDES}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating C++ API documentation using Doxygen")
    ADD_CUSTOM_TARGET(C++ApiDocs 
        DEPENDS "${CMAKE_BINARY_DIR}/meld-api-c++/index.html"
        COMMENT "Generating C++ API documentation using Doxygen"
        SOURCES 
            "${CMAKE_CURRENT_SOURCE_DIR}/DoxyfileC++.in"
            ${OPENMM_INCLUDES}
    ) 
    FILE(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/meld-api-c++/")
    INSTALL(DIRECTORY "${PROJECT_BINARY_DIR}/meld-api-c++/"
            DESTINATION "docs/meld-api-c++/")
    INSTALL(FILES "C++ API Reference.html"
            DESTINATION "docs/")
    ADD_CUSTOM_TARGET(DoxygenApiDocs
        DEPENDS "${CMAKE_BINARY_DIR}/meld-api-c++/index.html"
        COMMENT "Generating C++ API documentation using Doxygen"
        SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/DoxyfileC++.in"
        ${OPENMM_INCLUDES}
    )
    set(OPENMM_GENERATE_API_DOCS OFF CACHE BOOL "Whether to create API documentation using Doxygen")
    IF (OPENMM_GENERATE_API_DOCS)
        SET_TARGET_PROPERTIES(DoxygenApiDocs PROPERTIES EXCLUDE_FROM_ALL FALSE)
    ENDIF (OPENMM_GENERATE_API_DOCS)

ENDIF(DOXYGEN_EXECUTABLE)
