import Foundation
import Hummingbird

struct Car: ResponseEncodable {
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
            print("Random car request received")
            return randomCar()
        }

        // Serve static files from frontend dist using FileMiddleware
        router.add(middleware: FileMiddleware(frontendDist, searchForIndexHtml: true))

        // Bind to 0.0.0.0 to accept connections from all interfaces (required for container networking)
        let app = Application(
            router: router,
            configuration: .init(address: .hostname("0.0.0.0", port: 6002))
        )

        print("Server running on http://\(hostname):6002")
        try await app.runService()
    }
}
