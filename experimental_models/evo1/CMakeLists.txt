# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License. Frames are resized with OpenCV's bilinear filter, as the official
# server does (JetPack ships OpenCV).
find_package(OpenCV QUIET COMPONENTS core imgproc)
if(NOT OpenCV_FOUND)
  message(STATUS "OpenCV not found: skipping Evo-1")
  return()
endif()

add_library(evo1Experimental STATIC cpp/evo1Policy.cpp)
target_include_directories(
  evo1Experimental
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
          ${ONNX_PARSER_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(evo1Experimental PUBLIC vlaExperimental opencv_core
                                              opencv_imgproc)
set_property(TARGET evo1Experimental PROPERTY POSITION_INDEPENDENT_CODE ON)

foreach(example evo1_policy_server evo1_tokenize)
  add_executable(${example} examples/${example}.cpp)
  target_include_directories(${example} PRIVATE ${CMAKE_SOURCE_DIR}/cpp
                                                ${COMMON_INCLUDE_DIRS})
  target_link_libraries(${example} PRIVATE evo1Experimental vlaExperimental
                                           edgellmCore commonLibraryExt)
endforeach()
