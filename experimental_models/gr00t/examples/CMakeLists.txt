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

add_executable(gr00t_backbone_dump gr00t_backbone_dump.cpp)
target_include_directories(
  gr00t_backbone_dump PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
                              ${ONNX_PARSER_INCLUDE_DIR})
target_link_libraries(gr00t_backbone_dump PRIVATE edgellmCore commonLibraryExt)

add_executable(gr00t_policy_inference gr00t_policy_inference.cpp)
target_include_directories(
  gr00t_policy_inference PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
                                 ${ONNX_PARSER_INCLUDE_DIR})
target_link_libraries(gr00t_policy_inference
                      PRIVATE gr00tExperimental edgellmCore commonLibraryExt)

add_executable(gr00t_policy_server gr00t_policy_server.cpp)
target_include_directories(
  gr00t_policy_server PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
                              ${ONNX_PARSER_INCLUDE_DIR})
target_link_libraries(gr00t_policy_server PRIVATE gr00tExperimental edgellmCore
                                                  commonLibraryExt)
