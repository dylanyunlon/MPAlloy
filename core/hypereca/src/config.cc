#include <dress/config.h>
#include <dress/logging.h>
#include <cstdlib>
#include <cctype>

namespace Dress {

std::unordered_map<std::string, std::string> defaults_;

std::string key2env(std::string key) {
    for (auto& c: key) {
        if (islower(c)) {
            c -= 32;
        }
    }
    key = "DRESS_" + key;
    return key;
}

std::string Config::getString(std::string key) {
    key = key2env(key);
    auto cfg_it =  this->cfg_str_.find(key);
    if (cfg_it == this->cfg_str_.end()) {
        const char* e = getenv(key.c_str());
        if (e) {
            std::string v(e);
            this->cfg_str_[key] = v;
            return v;
        } else if (defaults_.find(key) != defaults_.end()) {
            auto v = defaults_[key];
            this->cfg_str_[key] = v;
            PWARNING << "Looked for configuration " << key
                << " but not found. Using default value: " << v << ".\n";
            return v;
        } else {
            PWARNING << "Looked for configuration " << key
                << " but not found. Using empty string.\n";
            this->cfg_str_[key] = "";
            return "";
        }
    }
    return cfg_it->second;
}

uint64_t Config::getInt(std::string key, uint64_t def) {
    auto cfg_it = this->cfg_int_.find(key);
    if (cfg_it  != this->cfg_int_.end()) {
        return cfg_it->second;
    }
    auto s(getString(key));
    if (s == "") {
        return cfg_int_[key] = def;
    }
    return cfg_int_[key] = (uint64_t)atoll(s.c_str());
}

bool Config::getBool(std::string key) {
    auto cfg_it = cfg_bool_.find(key);
    if (cfg_it != cfg_bool_.end()) {
        return cfg_it->second;
    }
    auto s(getString(key));
    bool on = false;
    if (s == "ON" || s == "on" || s == "1") {
        on = true;
    }
    return cfg_bool_[key] = on;
}

Config* cfg_instance_ = 0;

void setDefault(std::string key, std::string val) {
    defaults_[key2env(key)] = val;
}
void setDefault(std::string key, int n) {
    defaults_[key2env(key)] = std::to_string(n);
}
void initDefaultConfig() {
    setDefault("dedup_algo", "map");

    setDefault("nth_init", 8);
    setDefault("nth_prefetch", 1);
    setDefault("nth_embedding", 1);
    setDefault("nth_optimizer", 8);
    setDefault("nth_worker", 8);
    setDefault("nth_loader", 48);

    setDefault("profile_step", 100);

    setDefault("host_emb_pre_create", 512);
    setDefault("host_emb_map_bits", 6);
    setDefault("host_emb_chunk_sz", 0x4000);

    setDefault("ucwo_emb_worker", 8);
}

Config* cfgi() {
    if (!cfg_instance_) {
        initDefaultConfig();
        cfg_instance_ = new Config();
    }
    return cfg_instance_;
}

};  // namespace Dress

