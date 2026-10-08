# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
add_library(xvlaExperimental STATIC cpp/xvlaPolicy.cpp)
target_include_directories(
  xvlaExperimental
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
          ${ONNX_PARSER_INCLUDE_DIR})
target_link_libraries(xvlaExperimental PUBLIC vlaExperimental)
set_property(TARGET xvlaExperimental PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(xvla_policy_inference examples/xvla_policy_inference.cpp)
target_include_directories(xvla_policy_inference PRIVATE ${CMAKE_SOURCE_DIR}/cpp
                                                         ${COMMON_INCLUDE_DIRS})
target_link_libraries(
  xvla_policy_inference PRIVATE xvlaExperimental vlaExperimental edgellmCore
                                commonLibraryExt)

add_executable(xvla_async_control examples/xvla_async_control.cpp)
target_include_directories(xvla_async_control PRIVATE ${CMAKE_SOURCE_DIR}/cpp
                                                      ${COMMON_INCLUDE_DIRS})
target_link_libraries(
  xvla_async_control PRIVATE xvlaExperimental vlaExperimental edgellmCore
                             commonLibraryExt)
