#pragma once

#include <chrono>
#include <iostream>
#include <string>

class DurationLogger {
public:
    explicit DurationLogger(const std::string& message = "")
        : message(message + ": ")
        , start(std::chrono::steady_clock::now())
    {
    }

    ~DurationLogger()
    {
        auto finish = std::chrono::steady_clock::now();
        auto duration = finish - this->start;
        std::cerr << this->message
                  << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
                  << " ms" << std::endl;
    }

private:
    std::string message;
    std::chrono::steady_clock::time_point start;
};

#define UNIQ_ID(line_number) local_var##line_number

#define LOG_DURATION(message) \
    DurationLogger UNIQ_ID(__LINE__) { message };

#define PRINT_KV(value) #value << ": " << value
