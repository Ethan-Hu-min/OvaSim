#
# SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

find_package( OpenGL REQUIRED )

if( UNIX )
  # On Unix-like systems, shared libraries can use the soname system.
  set(GLFW_LIB_NAME glfw)
else()
  set(GLFW_LIB_NAME glfw3)
endif()

# Filter out warnings that cause problems with GLFW.
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  string( REPLACE "-Wdeclaration-after-statement" "" filtered_c_flags ${CMAKE_C_FLAGS} )
  string( REPLACE "-Wsign-compare" "" filtered_c_flags ${filtered_c_flags} )
  push_variable( CMAKE_C_FLAGS "${filtered_c_flags} -Wno-format-truncation -Wno-deprecated" )
endif()
add_subdirectory( GLFW )
set_property( TARGET glfw PROPERTY C_STANDARD 99 )
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  pop_variable( CMAKE_C_FLAGS )
endif()

add_subdirectory(gdt) 

if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
    push_variable( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-memaccess" )
endif()
add_subdirectory( imgui )
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  pop_variable( CMAKE_CXX_FLAGS )
endif()

add_library( glad SHARED
  KHR/khrplatform.h
  glad/glad.c
  glad/glad.h
 "spdlog/async.h" "spdlog/async_logger-inl.h" "spdlog/async_logger.h" "spdlog/common-inl.h" "spdlog/common.h" "spdlog/formatter.h" "spdlog/fwd.h" "spdlog/logger-inl.h" "spdlog/logger.h" "spdlog/mdc.h" "spdlog/pattern_formatter-inl.h" "spdlog/pattern_formatter.h" "spdlog/spdlog-inl.h" "spdlog/spdlog.h" "spdlog/stopwatch.h" "spdlog/tweakme.h" "spdlog/version.h" "spdlog/cfg/argv.h" "spdlog/cfg/env.h" "spdlog/cfg/helpers-inl.h" "spdlog/cfg/helpers.h" "spdlog/details/backtracer-inl.h" "spdlog/details/backtracer.h" "spdlog/details/circular_q.h" "spdlog/details/console_globals.h" "spdlog/details/file_helper-inl.h" "spdlog/details/file_helper.h" "spdlog/details/fmt_helper.h" "spdlog/details/log_msg-inl.h" "spdlog/details/log_msg.h" "spdlog/details/log_msg_buffer-inl.h" "spdlog/details/log_msg_buffer.h" "spdlog/details/mpmc_blocking_q.h" "spdlog/details/null_mutex.h" "spdlog/details/os-inl.h" "spdlog/details/os.h" "spdlog/details/periodic_worker-inl.h" "spdlog/details/periodic_worker.h" "spdlog/details/registry-inl.h" "spdlog/details/registry.h" "spdlog/details/synchronous_factory.h" "spdlog/details/tcp_client-windows.h" "spdlog/details/tcp_client.h" "spdlog/details/thread_pool-inl.h" "spdlog/details/thread_pool.h" "spdlog/details/udp_client-windows.h" "spdlog/details/udp_client.h" "spdlog/details/windows_include.h" "spdlog/fmt/bin_to_hex.h" "spdlog/fmt/chrono.h" "spdlog/fmt/compile.h" "spdlog/fmt/fmt.h" "spdlog/fmt/ostr.h" "spdlog/fmt/ranges.h" "spdlog/fmt/std.h" "spdlog/fmt/xchar.h" "spdlog/fmt/bundled/args.h" "spdlog/fmt/bundled/base.h" "spdlog/fmt/bundled/chrono.h" "spdlog/fmt/bundled/color.h" "spdlog/fmt/bundled/compile.h" "spdlog/fmt/bundled/core.h" "spdlog/fmt/bundled/format-inl.h" "spdlog/fmt/bundled/format.h" "spdlog/fmt/bundled/locale.h" "spdlog/fmt/bundled/os.h" "spdlog/fmt/bundled/ostream.h" "spdlog/fmt/bundled/printf.h" "spdlog/fmt/bundled/ranges.h" "spdlog/fmt/bundled/std.h" "spdlog/fmt/bundled/xchar.h" "spdlog/sinks/android_sink.h" "spdlog/sinks/ansicolor_sink-inl.h" "spdlog/sinks/ansicolor_sink.h" "spdlog/sinks/base_sink-inl.h" "spdlog/sinks/base_sink.h" "spdlog/sinks/basic_file_sink-inl.h" "spdlog/sinks/basic_file_sink.h" "spdlog/sinks/callback_sink.h" "spdlog/sinks/daily_file_sink.h" "spdlog/sinks/dist_sink.h" "spdlog/sinks/dup_filter_sink.h" "spdlog/sinks/hourly_file_sink.h" "spdlog/sinks/kafka_sink.h" "spdlog/sinks/mongo_sink.h" "spdlog/sinks/msvc_sink.h" "spdlog/sinks/null_sink.h" "spdlog/sinks/ostream_sink.h" "spdlog/sinks/qt_sinks.h" "spdlog/sinks/ringbuffer_sink.h" "spdlog/sinks/rotating_file_sink-inl.h" "spdlog/sinks/rotating_file_sink.h" "spdlog/sinks/sink-inl.h" "spdlog/sinks/sink.h" "spdlog/sinks/stdout_color_sinks-inl.h" "spdlog/sinks/stdout_color_sinks.h" "spdlog/sinks/stdout_sinks-inl.h" "spdlog/sinks/stdout_sinks.h" "spdlog/sinks/syslog_sink.h" "spdlog/sinks/systemd_sink.h" "spdlog/sinks/tcp_sink.h" "spdlog/sinks/udp_sink.h" "spdlog/sinks/wincolor_sink-inl.h" "spdlog/sinks/wincolor_sink.h" "spdlog/sinks/win_eventlog_sink.h")
target_compile_definitions( glad
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC  GLAD_GLAPI_EXPORT )
target_include_directories( glad PUBLIC . )
target_link_libraries( glad PUBLIC ${OPENGL_LIBRARIES} )


# Set IDE folders for targets
set_property( TARGET glad PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
set_property( TARGET glfw PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
set_property( TARGET imgui PROPERTY FOLDER ${OPTIX_IDE_FOLDER} )
