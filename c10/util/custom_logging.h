#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>
#include <unordered_map>

namespace Logger {

// .conf Config 파일 읽기
inline std::unordered_map<std::string, std::string> readConfigFile(const std::string& filename) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open config file: " << filename << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            config[key] = value;
        }
    }

    file.close();
    return config;
}

// 타임스탬프 형식: 2024-08-05 14:22:57.747.397661
inline std::string currentDateTime() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    auto now_micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000;
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << now_ms.count();
    oss << '.' << std::setw(3) << std::setfill('0') << now_micro.count();
    oss << std::setw(3) << std::setfill('0') << now_ns.count();

    return oss.str();
}

inline std::string removeRootPath(const std::string& filePath, const std::string& rootPath) {
    std::size_t pos = filePath.find(rootPath);
    if (pos != std::string::npos) {
        return filePath.substr(pos + rootPath.length());
    }
    // 만약 rootPath가 발견되지 않으면 원래 문자열을 반환합니다.
    return filePath;
}

// 파일 이름, 라인 번호, 함수 이름을 포함한 로그 메시지 출력
inline void custom_log(const std::string& file, int line, const std::string& func, const std::string& message) {
    static auto config = readConfigFile("/home/tjryu/work/asolab/project/torch-dynamic/custom_logging.conf");

    if (config["CPP_LOG"] == "1") {
        std::string relativePath = removeRootPath(file, "/home/tjryu/work/asolab/project/torch-dynamic");
        std::cout << "[" << currentDateTime() << "] | [" << relativePath << ":" << line << "] | [" << func << "] | " << message << std::endl;
    }
}

// 매크로를 통해 파일 이름, 라인 번호, 함수 이름을 자동으로 전달
#if defined(__GNUC__) || defined(__clang__)
#define CUSTOM_FUNC __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define CUSTOM_FUNC __FUNCSIG__
#else
#define CUSTOM_FUNC __func__
#endif

#define CustomLOG(message) Logger::custom_log(__FILE__, __LINE__, CUSTOM_FUNC, message)

} // namespace Logger

#endif // LOGGING_H
