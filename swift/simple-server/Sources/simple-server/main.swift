import Hummingbird

struct Car: Codable {
    let make: String
    let year: Int
}

@main
struct SimpleServer {
    static func main() async throws {
        let router = Router()

        // GET / - Returns "hello-world"
        router.get("/") { _, _ in
            "hello-world"
        }

        // GET /json - Returns a Car JSON object
        router.get("/json") { _, _ in
            Car(make: "Tesla", year: 2024)
        }

        let app = Application(
            router: router,
            configuration: .init(address: .hostname("0.0.0.0", port: 8000))
        )

        print("Server running on http://0.0.0.0:8000")
        try await app.runService()
    }
}
