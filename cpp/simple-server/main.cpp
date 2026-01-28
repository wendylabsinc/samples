#include "httplib.h"
#include "json.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

using json = nlohmann::json;

int main() {
    httplib::Server svr;

    const char* env_hostname = std::getenv("WENDY_HOSTNAME");
    std::string hostname = env_hostname ? env_hostname : "0.0.0.0";

    // GET / - Returns "Hello, World!"
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("Hello, World!", "text/plain");
    });

    // GET /hello/:name - Returns personalized greeting
    svr.Get(R"(/hello/(\w+))", [](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        res.set_content("Hello, " + name + "!", "text/plain");
    });

    // POST /users - Creates a new user from JSON body
    svr.Post("/users", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string username = body.value("username", "");

            json user;
            user["id"] = 1;
            user["username"] = username;

            res.status = 201;
            res.set_content(user.dump(), "application/json");
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(R"({"error": "Invalid JSON"})", "application/json");
        }
    });

    std::cout << "Server running on http://" << hostname << ":3000" << std::endl;
    svr.listen("0.0.0.0", 3000);

    return 0;
}
