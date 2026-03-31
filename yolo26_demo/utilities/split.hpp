#pragma once

#include <string>
#include <vector>


namespace utilities
{
    std::vector<std::string> split_string(const std::string& content, const std::string& delimiter)
    {
        std::vector<std::string> result;

        std::string::size_type pos1 = 0;
        std::string::size_type pos2 = content.find(delimiter);

        while (std::string::npos != pos2)
        {
            result.push_back(content.substr(pos1, pos2 - pos1));

            pos1 = pos2 + delimiter.size();
            pos2 = content.find(delimiter, pos1);
        }

        if (pos1 != content.length())
        {
            result.push_back(content.substr(pos1));
        }

        return result;
    }
}