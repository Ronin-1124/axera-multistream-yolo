#include "app/config.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <map>

namespace app {

namespace {

// Minimal JSON parser for our simple config format.
std::string parse_string(const std::string& json, size_t& pos) {
    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos;
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            result += json[pos++];
        } else {
            result += json[pos++];
        }
    }
    if (pos < json.size()) ++pos; // skip closing quote
    return result;
}

double parse_number(const std::string& json, size_t& pos) {
    size_t start = pos;
    if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) ++pos;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos < json.size() && json[pos] == '.') {
        ++pos;
        while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) ++pos;
    }
    if (pos < json.size() && (json[pos] == 'e' || json[pos] == 'E')) {
        ++pos;
        if (pos < json.size() && (json[pos] == '+' || json[pos] == '-')) ++pos;
        while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) ++pos;
    }
    return std::stod(json.substr(start, pos - start));
}

bool parse_bool(const std::string& json, size_t& pos) {
    if (json.compare(pos, 4, "true") == 0) {
        pos += 4;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        pos += 5;
        return false;
    }
    return false;
}

void skip_ws(const std::string& json, size_t& pos) {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
}

std::string parse_key(const std::string& json, size_t& pos) {
    skip_ws(json, pos);
    return parse_string(json, pos);
}

// Parse a simple { "key": value, ... } object into key->value string map.
std::map<std::string, std::string> parse_object(const std::string& json, size_t& pos) {
    std::map<std::string, std::string> result;
    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == '{') ++pos;
    while (true) {
        skip_ws(json, pos);
        if (pos >= json.size()) break;
        if (json[pos] == '}') { ++pos; break; }
        if (json[pos] == ',') { ++pos; continue; }

        std::string key = parse_key(json, pos);
        skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ':') ++pos;
        skip_ws(json, pos);

        if (pos < json.size() && json[pos] == '"') {
            result[key] = parse_string(json, pos);
        } else if (pos < json.size() && (json[pos] == '-' || std::isdigit(static_cast<unsigned char>(json[pos])))) {
            result[key] = std::to_string(parse_number(json, pos));
        } else if (pos < json.size() && (json[pos] == 't' || json[pos] == 'f')) {
            result[key] = parse_bool(json, pos) ? "true" : "false";
        } else if (pos < json.size() && json[pos] == '[') {
            // Skip arrays for now (not used in our config)
            int depth = 0;
            while (pos < json.size()) {
                if (json[pos] == '[') { ++depth; ++pos; }
                else if (json[pos] == ']') { if (--depth == 0) { ++pos; break; } else ++pos; }
                else ++pos;
            }
        } else {
            ++pos; // skip unknown
        }
    }
    return result;
}

// Parse a simple array of { ... } objects.
std::vector<std::map<std::string, std::string>> parse_array_of_objects(const std::string& json, size_t& pos) {
    std::vector<std::map<std::string, std::string>> result;
    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == '[') ++pos;
    while (true) {
        skip_ws(json, pos);
        if (pos >= json.size()) break;
        if (json[pos] == ']') { ++pos; break; }
        if (json[pos] == ',') { ++pos; continue; }
        // Must be start of an object
        if (json[pos] != '{') { ++pos; continue; }
        std::map<std::string, std::string> obj = parse_object(json, pos);
        result.push_back(obj);
        skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ',') ++pos;
    }
    return result;
}

} // anonymous namespace

AppConfig load_config(const std::string& json_path) {
    AppConfig cfg;

    std::ifstream file(json_path);
    if (!file.is_open()) {
        fprintf(stderr, "[Config] failed to open %s\n", json_path.c_str());
        return cfg;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    std::string json = ss.str();
    file.close();

    size_t pos = 0;

    // Parse streams array — find "streams" section and parse from there
    size_t streams_pos = json.find("\"streams\"");
    if (streams_pos != std::string::npos) {
        pos = streams_pos;
        while (pos < json.size() && json[pos] != '[') ++pos;
        auto stream_objs = parse_array_of_objects(json, pos);
        for (auto& obj : stream_objs) {
            StreamConfig sc;
            sc.id = std::stoi(obj.count("id") ? obj["id"] : "0");
            sc.url = obj.count("url") ? obj["url"] : "";
            sc.enabled = obj.count("enabled") ? obj["enabled"] == "true" : true;
            cfg.streams.push_back(sc);
        }
    }

    // Parse model
    size_t model_pos = json.find("\"model\"");
    if (model_pos != std::string::npos) {
        pos = model_pos;
        while (pos < json.size() && json[pos] != '{') ++pos;
        auto model_obj = parse_object(json, pos);
        cfg.model.path = model_obj.count("path") ? model_obj["path"] : "";
        cfg.model.input_h = model_obj.count("input_h") ? std::stoi(model_obj["input_h"]) : 640;
        cfg.model.input_w = model_obj.count("input_w") ? std::stoi(model_obj["input_w"]) : 640;
        cfg.model.type = model_obj.count("type") ? model_obj["type"] : "yolo26";
    }

    // Parse inference
    size_t inf_pos = json.find("\"inference\"");
    if (inf_pos != std::string::npos) {
        pos = inf_pos;
        while (pos < json.size() && json[pos] != '{') ++pos;
        auto inf_obj = parse_object(json, pos);
        cfg.inference.num_threads = inf_obj.count("num_threads") ? std::stoi(inf_obj["num_threads"]) : 0;
        cfg.inference.conf_thresh = inf_obj.count("conf_thresh") ? std::stod(inf_obj["conf_thresh"]) : 0.45f;
        cfg.inference.iou_thresh = inf_obj.count("iou_thresh") ? std::stod(inf_obj["iou_thresh"]) : 0.45f;
    }

    // Parse output
    size_t out_pos = json.find("\"output\"");
    if (out_pos != std::string::npos) {
        pos = out_pos;
        while (pos < json.size() && json[pos] != '{') ++pos;
        auto out_obj = parse_object(json, pos);
        cfg.output.display = out_obj.count("display") ? out_obj["display"] == "true" : true;
        cfg.output.mosaic_cols = out_obj.count("mosaic_cols") ? std::stoi(out_obj["mosaic_cols"]) : 2;
        cfg.output.queue_depth = out_obj.count("queue_depth") ? std::stoi(out_obj["queue_depth"]) : 2;
        cfg.output.display_w = out_obj.count("display_w") ? std::stoi(out_obj["display_w"]) : 1920;
        cfg.output.display_h = out_obj.count("display_h") ? std::stoi(out_obj["display_h"]) : 720;
    }

    return cfg;
}

} // namespace app
