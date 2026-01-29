import Foundation
import Hummingbird

struct Car: Encodable {
    let name: String
    let make: String
    let year: Int
    let color: String
    let createdAt: String
}

let carNames = ["Honda", "Toyota", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Tesla", "Nissan", "Mazda"]
let carMakes = ["Civic", "Camry", "Mustang", "Corvette", "M3", "C-Class", "A4", "Model 3", "Altima", "MX-5"]

func randomCar() -> Car {
    let name = carNames.randomElement()!
    let make = carMakes.randomElement()!
    let year = Int.random(in: 1990...2024)
    let color = String(format: "#%06X", Int.random(in: 0...0xFFFFFF))

    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    let createdAt = formatter.string(from: Date())

    return Car(name: name, make: make, year: year, color: color, createdAt: createdAt)
}

@main
struct WebAppServer {
    static func main() async throws {
        let hostname = ProcessInfo.processInfo.environment["WENDY_HOSTNAME"] ?? "0.0.0.0"

        // Determine frontend dist path
        let envPath = ProcessInfo.processInfo.environment["FRONTEND_DIST"]
        let containerPath = "/app/frontend/dist"
        let cwdPath = FileManager.default.currentDirectoryPath + "/../frontend/dist"

        let frontendDist: String
        if let env = envPath {
            frontendDist = env
        } else if FileManager.default.fileExists(atPath: containerPath + "/index.html") {
            frontendDist = containerPath
        } else {
            frontendDist = cwdPath
        }

        print("Serving frontend from: \(frontendDist)")

        let router = Router()

        // API route for random car
        router.get("/api/random-car") { _, _ in
            return randomCar()
        }

        // Serve static files from frontend dist
        router.get("/*") { request, context in
            let path = request.uri.path.removingPercentEncoding ?? request.uri.path
            let cleanPath = path == "/" ? "/index.html" : path

            let filePath = frontendDist + cleanPath
            let fileURL = URL(fileURLWithPath: filePath)

            // Check if file exists
            if FileManager.default.fileExists(atPath: filePath) {
                let data = try Data(contentsOf: fileURL)
                let contentType = getContentType(for: filePath)
                return Response(
                    status: .ok,
                    headers: [.contentType: contentType],
                    body: .init(byteBuffer: .init(data: data))
                )
            }

            // Fallback to index.html for SPA routing
            let indexPath = frontendDist + "/index.html"
            let indexURL = URL(fileURLWithPath: indexPath)
            let data = try Data(contentsOf: indexURL)
            return Response(
                status: .ok,
                headers: [.contentType: "text/html"],
                body: .init(byteBuffer: .init(data: data))
            )
        }

        let app = Application(
            router: router,
            configuration: .init(address: .hostname("0.0.0.0", port: 6002))
        )

        print("Server running on http://\(hostname):6002")
        try await app.runService()
    }

    static func getContentType(for path: String) -> String {
        if path.hasSuffix(".html") { return "text/html" }
        if path.hasSuffix(".css") { return "text/css" }
        if path.hasSuffix(".js") { return "application/javascript" }
        if path.hasSuffix(".json") { return "application/json" }
        if path.hasSuffix(".png") { return "image/png" }
        if path.hasSuffix(".jpg") || path.hasSuffix(".jpeg") { return "image/jpeg" }
        if path.hasSuffix(".svg") { return "image/svg+xml" }
        if path.hasSuffix(".ico") { return "image/x-icon" }
        if path.hasSuffix(".woff") { return "font/woff" }
        if path.hasSuffix(".woff2") { return "font/woff2" }
        return "application/octet-stream"
    }
}
