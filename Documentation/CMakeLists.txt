find_package(Doxygen)

if (DOXYGEN_FOUND)
    set(DOXYGEN_INPUT_FILE  Doxyfile.in)
    set(DOXYGEN_IN          ${CMAKE_CURRENT_SOURCE_DIR}/${DOXYGEN_INPUT_FILE})
    set(DOXYGEN_OUT         ${CMAKE_CURRENT_BINARY_DIR})

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(Documentation ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM 
        )
    
    # Unfortunately, it doesn't seem you can ALIAS a custom target.
    # Therefore, to make a shorthand, just make an empty target that
    # depends on the target we're trying to alias.
    add_custom_target(docs DEPENDS Documentation)
else (DOXYGEN_FOUND)
    message("Doxygen required to generate documentation.")
endif (DOXYGEN_FOUND)
