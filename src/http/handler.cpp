#include "cllm/http/handler.h"
#include <algorithm>

namespace cllm {

HttpHandler::HttpHandler() {
}

HttpHandler::~HttpHandler() {
}

void HttpHandler::get(const std::string& path, HandlerFunc handler) {
    std::string normalizedPath = normalizePath(path);
    getHandlers_[normalizedPath] = handler;
}

void HttpHandler::post(const std::string& path, HandlerFunc handler) {
    std::string normalizedPath = normalizePath(path);
    postHandlers_[normalizedPath] = handler;
}

void HttpHandler::put(const std::string& path, HandlerFunc handler) {
    std::string normalizedPath = normalizePath(path);
    putHandlers_[normalizedPath] = handler;
}

void HttpHandler::del(const std::string& path, HandlerFunc handler) {
    std::string normalizedPath = normalizePath(path);
    deleteHandlers_[normalizedPath] = handler;
}

HttpResponse HttpHandler::handleRequest(const HttpRequest& request) {
    std::string method = request.getMethod();
    std::string path = normalizePath(request.getPath());
    
    std::map<std::string, HandlerFunc>* handlers = nullptr;
    
    if (method == "GET") {
        handlers = &getHandlers_;
    } else if (method == "POST") {
        handlers = &postHandlers_;
    } else if (method == "PUT") {
        handlers = &putHandlers_;
    } else if (method == "DELETE") {
        handlers = &deleteHandlers_;
    } else {
        return HttpResponse::badRequest("Unsupported HTTP method: " + method);
    }
    
    for (const auto& entry : *handlers) {
        if (matchPath(entry.first, path)) {
            return entry.second(request);
        }
    }
    
    return HttpResponse::notFound();
}

bool HttpHandler::hasHandler(const std::string& method, const std::string& path) const {
    std::string normalizedPath = normalizePath(path);
    const std::map<std::string, HandlerFunc>* handlers = nullptr;
    
    if (method == "GET") {
        handlers = &getHandlers_;
    } else if (method == "POST") {
        handlers = &postHandlers_;
    } else if (method == "PUT") {
        handlers = &putHandlers_;
    } else if (method == "DELETE") {
        handlers = &deleteHandlers_;
    } else {
        return false;
    }
    
    for (const auto& entry : *handlers) {
        if (matchPath(entry.first, normalizedPath)) {
            return true;
        }
    }
    
    return false;
}

std::string HttpHandler::normalizePath(const std::string& path) const {
    std::string normalized = path;
    
    while (!normalized.empty() && normalized.back() == '/') {
        normalized.pop_back();
    }
    
    return normalized;
}

bool HttpHandler::matchPath(const std::string& pattern, const std::string& path) const {
    if (pattern == path) {
        return true;
    }
    
    if (pattern.find('*') != std::string::npos) {
        size_t starPos = pattern.find('*');
        std::string prefix = pattern.substr(0, starPos);
        std::string suffix = pattern.substr(starPos + 1);
        
        if (path.length() >= prefix.length() + suffix.length()) {
            if (path.substr(0, prefix.length()) == prefix &&
                path.substr(path.length() - suffix.length()) == suffix) {
                return true;
            }
        }
    }
    
    return false;
}

}
