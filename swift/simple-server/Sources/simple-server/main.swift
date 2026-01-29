import Foundation
import Hummingbird

struct Car: ResponseEncodable {
    let make: String
    let year: Int
}

@main
struct SimpleServer {
    static func main() async throws {
        let hostname = ProcessInfo.processInfo.environment["WENDY_HOSTNAME"] ?? "0.0.0.0"

        let router = Router()

        // GET / - Returns "hello-world"
        router.get("/") { _, _ in
            print("Received request: GET /")
            return "hello-world"
        }

        // GET /json - Returns a Car JSON object
        router.get("/json") { _, _ in
            print("Received request: GET /json")
            return Car(make: "Tesla", year: 2024)
        }

        let app = Application(
            router: router,
            configuration: .init(address: .hostname("0.0.0.0", port: 6001))
        )

        print("Server running on http://\(hostname):6001")
        try await app.runService()
    }
}
