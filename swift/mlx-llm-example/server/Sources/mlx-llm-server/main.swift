import Foundation
import Hummingbird
@preconcurrency import MLX
import MLXLLM
import MLXLMCommon

struct ChatRequest: Decodable, Sendable {
  let prompt: String
  let maxTokens: Int?
}

struct ChatResponse: ResponseEncodable {
  let text: String
  let tokensGenerated: Int
  let generationTimeMs: Int64
}

struct HealthResponse: ResponseEncodable {
  let status: String
  let model: String
}

@MainActor final class ChatManager {
  private let modelId: String
  private let model: ModelContext

  init(modelId: String) async throws {
    self.modelId = modelId
    self.model = try await MLXLMCommon.loadModel(id: modelId)
  }

  func respond(to prompt: String) async throws -> String {
    nonisolated(unsafe) let session = ChatSession(model)
    return try await session.respond(to: prompt)
  }

  enum ChatError: Error {
    case notLoaded
  }
}

@main
struct MLXLLMServer {
  nonisolated static func main() async throws {
    // Load the model
    try await Stream.withNewDefaultStream(device: .cpu) {
      try await MLXLLMServer.run()
    }
  }

  @Sendable nonisolated static func run() async throws {
    let hostname = ProcessInfo.processInfo.environment["WENDY_HOSTNAME"] ?? "0.0.0.0"
      let modelId =
        ProcessInfo.processInfo.environment["MODEL_ID"]
        ?? "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"

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
      print("Using model: \(modelId)")

      
      let chatManager = try await ChatManager(modelId: modelId)

      let router = Router()

      // Health check endpoint
      router.get("/health") { _, _ -> HealthResponse in
        HealthResponse(status: "ok", model: modelId)
      }

      // Chat endpoint
      router.post("/v1/chat") { request, context in
        let body = try await request.decode(as: ChatRequest.self, context: context)

        let text = try await chatManager.respond(to: body.prompt)

        return ChatResponse(
          text: text,
          tokensGenerated: 0,
          generationTimeMs: 0
        )
      }

      // Serve static files from frontend dist
      router.add(middleware: FileMiddleware(frontendDist, searchForIndexHtml: true))

      let app = Application(
        router: router,
        configuration: .init(address: .hostname("0.0.0.0", port: 8080))
      )

      print("Server running on http://\(hostname):8080")
      try await app.runService()
  }
}
