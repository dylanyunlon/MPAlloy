#pragma once
#ifndef DRESS_CONFIG_H
#define DRESS_CONFIG_H

#include <string>
#include <unordered_map>

namespace Dress {

class Config {
private:
    std::unordered_map<std::string, std::string> cfg_str_;
    std::unordered_map<std::string, bool> cfg_bool_;
    std::unordered_map<std::string, uint64_t> cfg_int_; 
public:
    std::string getString(std::string key);
    uint64_t getInt(std::string key, uint64_t def=-1ull);
    bool getBool(std::string key);
};

Config* cfgi(); // Get global config instance

};  // namespace Dress

#endif  // DRESS_CONFIG_H
