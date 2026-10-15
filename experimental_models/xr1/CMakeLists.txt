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
# the License.
add_library(xr1Experimental STATIC cpp/xr1Policy.cpp)
target_include_directories(
  xr1Experimental
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
          ${ONNX_PARSER_INCLUDE_DIR})
target_link_libraries(xr1Experimental PUBLIC vlaExperimental)
set_property(TARGET xr1Experimental PROPERTY POSITION_INDEPENDENT_CODE ON)

foreach(example xr1_policy_server xr1_inputs_dump)
  add_executable(${example} examples/${example}.cpp)
  target_include_directories(${example} PRIVATE ${CMAKE_SOURCE_DIR}/cpp
                                                ${COMMON_INCLUDE_DIRS})
  target_link_libraries(${example} PRIVATE xr1Experimental vlaExperimental
                                           edgellmCore commonLibraryExt)
endforeach()
