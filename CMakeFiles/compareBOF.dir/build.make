# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/new/src/opencv_BOF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/new/src/opencv_BOF

# Include any dependencies generated for this target.
include CMakeFiles/compareBOF.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/compareBOF.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/compareBOF.dir/flags.make

CMakeFiles/compareBOF.dir/compareBOF.cpp.o: CMakeFiles/compareBOF.dir/flags.make
CMakeFiles/compareBOF.dir/compareBOF.cpp.o: compareBOF.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/new/src/opencv_BOF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/compareBOF.dir/compareBOF.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/compareBOF.dir/compareBOF.cpp.o -c /home/new/src/opencv_BOF/compareBOF.cpp

CMakeFiles/compareBOF.dir/compareBOF.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compareBOF.dir/compareBOF.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/new/src/opencv_BOF/compareBOF.cpp > CMakeFiles/compareBOF.dir/compareBOF.cpp.i

CMakeFiles/compareBOF.dir/compareBOF.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compareBOF.dir/compareBOF.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/new/src/opencv_BOF/compareBOF.cpp -o CMakeFiles/compareBOF.dir/compareBOF.cpp.s

CMakeFiles/compareBOF.dir/compareBOF.cpp.o.requires:

.PHONY : CMakeFiles/compareBOF.dir/compareBOF.cpp.o.requires

CMakeFiles/compareBOF.dir/compareBOF.cpp.o.provides: CMakeFiles/compareBOF.dir/compareBOF.cpp.o.requires
	$(MAKE) -f CMakeFiles/compareBOF.dir/build.make CMakeFiles/compareBOF.dir/compareBOF.cpp.o.provides.build
.PHONY : CMakeFiles/compareBOF.dir/compareBOF.cpp.o.provides

CMakeFiles/compareBOF.dir/compareBOF.cpp.o.provides.build: CMakeFiles/compareBOF.dir/compareBOF.cpp.o


# Object files for target compareBOF
compareBOF_OBJECTS = \
"CMakeFiles/compareBOF.dir/compareBOF.cpp.o"

# External object files for target compareBOF
compareBOF_EXTERNAL_OBJECTS =

compareBOF: CMakeFiles/compareBOF.dir/compareBOF.cpp.o
compareBOF: CMakeFiles/compareBOF.dir/build.make
compareBOF: /usr/local/lib/libopencv_stitching.so.3.2.0
compareBOF: /usr/local/lib/libopencv_superres.so.3.2.0
compareBOF: /usr/local/lib/libopencv_videostab.so.3.2.0
compareBOF: /usr/local/lib/libopencv_aruco.so.3.2.0
compareBOF: /usr/local/lib/libopencv_bgsegm.so.3.2.0
compareBOF: /usr/local/lib/libopencv_bioinspired.so.3.2.0
compareBOF: /usr/local/lib/libopencv_ccalib.so.3.2.0
compareBOF: /usr/local/lib/libopencv_datasets.so.3.2.0
compareBOF: /usr/local/lib/libopencv_dpm.so.3.2.0
compareBOF: /usr/local/lib/libopencv_face.so.3.2.0
compareBOF: /usr/local/lib/libopencv_freetype.so.3.2.0
compareBOF: /usr/local/lib/libopencv_fuzzy.so.3.2.0
compareBOF: /usr/local/lib/libopencv_line_descriptor.so.3.2.0
compareBOF: /usr/local/lib/libopencv_optflow.so.3.2.0
compareBOF: /usr/local/lib/libopencv_plot.so.3.2.0
compareBOF: /usr/local/lib/libopencv_reg.so.3.2.0
compareBOF: /usr/local/lib/libopencv_saliency.so.3.2.0
compareBOF: /usr/local/lib/libopencv_stereo.so.3.2.0
compareBOF: /usr/local/lib/libopencv_structured_light.so.3.2.0
compareBOF: /usr/local/lib/libopencv_surface_matching.so.3.2.0
compareBOF: /usr/local/lib/libopencv_text.so.3.2.0
compareBOF: /usr/local/lib/libopencv_xfeatures2d.so.3.2.0
compareBOF: /usr/local/lib/libopencv_ximgproc.so.3.2.0
compareBOF: /usr/local/lib/libopencv_xobjdetect.so.3.2.0
compareBOF: /usr/local/lib/libopencv_xphoto.so.3.2.0
compareBOF: /usr/local/lib/libopencv_shape.so.3.2.0
compareBOF: /usr/local/lib/libopencv_video.so.3.2.0
compareBOF: /usr/local/lib/libopencv_phase_unwrapping.so.3.2.0
compareBOF: /usr/local/lib/libopencv_rgbd.so.3.2.0
compareBOF: /usr/local/lib/libopencv_calib3d.so.3.2.0
compareBOF: /usr/local/lib/libopencv_features2d.so.3.2.0
compareBOF: /usr/local/lib/libopencv_flann.so.3.2.0
compareBOF: /usr/local/lib/libopencv_objdetect.so.3.2.0
compareBOF: /usr/local/lib/libopencv_ml.so.3.2.0
compareBOF: /usr/local/lib/libopencv_highgui.so.3.2.0
compareBOF: /usr/local/lib/libopencv_photo.so.3.2.0
compareBOF: /usr/local/lib/libopencv_videoio.so.3.2.0
compareBOF: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
compareBOF: /usr/local/lib/libopencv_imgproc.so.3.2.0
compareBOF: /usr/local/lib/libopencv_core.so.3.2.0
compareBOF: CMakeFiles/compareBOF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/new/src/opencv_BOF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compareBOF"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compareBOF.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/compareBOF.dir/build: compareBOF

.PHONY : CMakeFiles/compareBOF.dir/build

CMakeFiles/compareBOF.dir/requires: CMakeFiles/compareBOF.dir/compareBOF.cpp.o.requires

.PHONY : CMakeFiles/compareBOF.dir/requires

CMakeFiles/compareBOF.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/compareBOF.dir/cmake_clean.cmake
.PHONY : CMakeFiles/compareBOF.dir/clean

CMakeFiles/compareBOF.dir/depend:
	cd /home/new/src/opencv_BOF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/new/src/opencv_BOF /home/new/src/opencv_BOF /home/new/src/opencv_BOF /home/new/src/opencv_BOF /home/new/src/opencv_BOF/CMakeFiles/compareBOF.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/compareBOF.dir/depend
