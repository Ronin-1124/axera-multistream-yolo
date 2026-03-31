#pragma once

#include <chrono>

class timer
{
private:
    std::chrono::system_clock::time_point start_time, end_time;

public:
    timer()
    {
        start();
    }

    void start()
    {
        stop();
        this->start_time = this->end_time;
    }

    void stop()
    {
#ifdef _MSC_VER
        this->end_time = std::chrono::system_clock::now();
#else
        this->end_time = std::chrono::high_resolution_clock::now();
#endif
    }

    float cost()
    {
        if (this->end_time <= this->start_time)
        {
            this->stop();
        }

        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        return static_cast<float>(ms) / 1000.f;
    }
};