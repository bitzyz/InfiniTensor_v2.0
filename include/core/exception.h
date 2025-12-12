#pragma once
#ifndef EXCEPTION_H
#define EXCEPTION_H
#include <sstream>
#include <stdexcept>
#include <string>

namespace infini {

class Exception : public std::runtime_error {
  private:
    std::string info;

  public:
    explicit Exception(const std::string &msg)
        : std::runtime_error(msg), info(msg) {}

    template <typename T> Exception &operator<<(const T &value) {
        std::ostringstream oss;
        oss << value;
        info += oss.str();
        return *this;
    }

    const char *what() const noexcept override { return info.c_str(); }
};

#define CHECK_INFINI_ERROR(call)                                               \
    if (auto err = call; err != INFINI_STATUS_SUCCESS)                         \
    throw ::infini::Exception(                                                 \
        std::string("[") + __FILE__ + ":" + std::to_string(__LINE__) +         \
        "] operators error (" + #call + "): " + std::to_string(err))

} // namespace infini
#endif // EXCEPTION_H
