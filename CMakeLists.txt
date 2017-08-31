IF (WIN32)
  ADD_DEFINITIONS(-D__BUILD_GADGETRON_CPURADIAL__)
ENDIF (WIN32)

find_package(Ismrmrd REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/gadgets/mri_core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/solvers
  ${CMAKE_SOURCE_DIR}/toolboxes/solvers/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/operators
  ${CMAKE_SOURCE_DIR}/toolboxes/operators/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/gtplus
  ${CMAKE_SOURCE_DIR}/toolboxes/gtplus/workflow
  ${CMAKE_SOURCE_DIR}/toolboxes/gtplus/util
  ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
  ${CMAKE_SOURCE_DIR}/toolboxes/klt/cpu
  
  
        ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
        ${CMAKE_SOURCE_DIR}/toolboxes/pmri/gpu 
  	${CMAKE_SOURCE_DIR}/toolboxes/core/gpu 
        ${CMAKE_SOURCE_DIR}/toolboxes/fft/gpu
	${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu
	${CMAKE_SOURCE_DIR}/toolboxes/operators/gpu
	${CMAKE_SOURCE_DIR}/toolboxes/solvers/gpu
	${CUDA_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR} 
	${ISMRMRD_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  )
  

add_library(gadgetron_cpuradial SHARED 
  gadgetron_cpuradial_export.h
  
  
  cpuRadialNUFFTGadget.h 
  cpuRadialNUFFTGadget.cpp

  gpuRadialNUFFTGadget.h
  gpuRadialNUFFTGadget.cpp
 
  gpuRadialCGSENSEGadget.h 
  gpuRadialCGSENSEGadget.cpp 
  
  cpuRadialCGSENSEGadget.h
  cpuRadialCGSENSEGadget.cpp
  
  Trajectory2D.h
  ${PHASE_GADGET})

set_target_properties(gadgetron_cpuradial PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_cpuradial
  gadgetron_gadgetbase
  gadgetron_toolbox_log
   gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math
   gadgetron_toolbox_gtplus
  ${ISMRMRD_LIBRARIES} ${FFTW3_LIBRARIES}
  optimized ${ACE_LIBRARIES} debug ${ACE_DEBUG_LIBRARY}
  )

target_link_libraries(gadgetron_cpuradial libnfft3.so)

target_link_libraries(gadgetron_cpuradial gadgetron_toolbox_gpuoperators gadgetron_toolbox_gpucore gadgetron_toolbox_gpuparallelmri gadgetron_toolbox_gpunfft gadgetron_toolbox_hostutils gadgetron_toolbox_gpusolvers ${CUDA_LIBRARIES})

install (FILES  gadgetron_cpuradial_export.h

                cpuRadialNUFFTGadget.h 
                gpuRadialNUFFTGadget.h
                gpuRadialCGSENSEGadget.h
                cpuRadialCGSENSEGadget.h
                
                DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

install (TARGETS gadgetron_cpuradial DESTINATION lib COMPONENT main)

add_subdirectory(config)
