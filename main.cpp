#include "gtest/gtest.h"

#include <filesystem>
#include <print>

// Walk up from the current working directory looking for a ".root" marker
// file. When found, make that directory the process CWD so relative paths
// (e.g. bench_data/) resolve against the project root regardless of where
// the binary was launched from. Returns true if the root was found.
static bool SetCwdToProjectRoot()
{
    namespace fs = std::filesystem;

    for (fs::path Dir = fs::current_path(); ; Dir = Dir.parent_path())
    {
        if (fs::exists(Dir / ".root"))
        {
            fs::current_path(Dir);
            std::println("Project root: {}", Dir.string());
            return true;
        }

        if (Dir == Dir.parent_path())   // reached the filesystem root
            break;
    }

    std::println(stderr, "Warning: '.root' marker not found; leaving CWD unchanged.");
    return false;
}

int main(int argc, char** argv)
{
    SetCwdToProjectRoot();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}