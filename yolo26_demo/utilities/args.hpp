#pragma once

#include <array>
#include <vector>
#include <type_traits>

#include "utilities/split.hpp"


namespace utilities
{
    template <typename T, size_t N>
    bool parse_string(const std::string& argument_string, std::array<T, N>& arguments, const std::string& delimiter = ",")
    {
        std::vector<std::string> result = split_string(argument_string, delimiter);

        if (N != result.size())
        {
            return false;
        }

        for (size_t i = 0; i < N; i++)
        {
            if (std::is_integral<T>::value)
            {
                arguments[i] = std::stoi(result[i]);
            }

            if (std::is_floating_point<T>::value)
            {
                arguments[i] = std::stof(result[i]);
            }
        }

        return true;
    }
}