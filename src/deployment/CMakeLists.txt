project(Turi)

if(APPLE AND HAS_COREML_CUSTOM_MODEL)
  message(STATUS "Enabling Recommender and AudioPreprocessing targets.")

  make_library(Recommender
    SOURCES
      TuriCreateObjC.mm
      recommender_initialization.cpp
      ${_TC_COMMON_OBJECTS}
      $<TARGET_OBJECTS:capi>
    REQUIRES
      ${_TC_COMMON_REQUIREMENTS}
    SHARED
    SHARED_ALL_DEFINED
    EXPORT_OSX_MAP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/symbol_exports.ver"
    DEAD_STRIP
  )

  set_target_properties(AudioPreprocessing PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    MACOSX_FRAMEWORK_IDENTIFIER "AudioPreprocessing"
    MACOSX_FRAMEWORK_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/...
    MACOSX_RPATH TRUE
  )

  # Adding sources here allows us to unit test these custom models in python.
  make_library(objcapi OBJECT
    SOURCES
      AudioPreprocessing.mm
      TuriCreateObjC.mm
      ${_TC_COMMON_OBJECTS}
    REQUIRES
      ${_TC_COMMON_REQUIREMENTS}
      ${FOUNDATION}
      ${CORE_ML}
  )

  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET Recommender
      POST_BUILD
      COMMAND strip -Sx "$<TARGET_FILE:Recommender>"
      COMMENT "Stripping Objective-C Recommender library"
    )
    add_custom_command(TARGET AudioPreprocessing
      POST_BUILD
      COMMAND strip -Sx "$<TARGET_FILE:AudioPreprocessing>"
      COMMENT "Stripping Objective-C AudioPreprocessing library"
    )
  endif()
else()

  make_empty_library(Recommender)
  make_empty_library(AudioPreprocessing)

  message(STATUS "Skipping Recommender and AudioPreprocessing targets.")
endif()
