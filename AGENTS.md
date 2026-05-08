# Agent Instructions

## CMake Configuration

When performing a clean build, reconfiguring the project, or resolving dependency errors, configure CMake with the vcpkg toolchain file and disable the broken manual HDF5 installation:

```powershell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="H:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

## Build

Always build with:

```powershell
cmake --build build --config Release
```

If the build directory does not exist, run the CMake Configuration command above first.

## Important Build Notes

- **NEVER make multiple tool calls in the same message** - Send ONE command, wait for response, then send the next
- **NEVER run build and tests in parallel** - Build must complete successfully before running tests
- **NEVER run build and executables in parallel** - Wait for build to complete before running any .exe files
- Always verify build succeeds (check for "error" in output) before running test executables or examples

## Sequential Execution Example

1. Send message with: `cmake --build build --config Release`
2. Wait for response

