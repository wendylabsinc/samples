import SwiftCrossUI
import GtkBackend

@main
struct KioskApp: App {
    var body: some Scene {
        WindowGroup("Kiosk App") {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var count = 0

    var body: some View {
        VStack(spacing: 20) {
            Text("Swift GTK Kiosk App")
                .font(.largeTitle)

            Text("Running in Cage Wayland Compositor")
                .font(.body)

            Text("Count: \(count)")
                .font(.title)

            HStack(spacing: 10) {
                Button("Decrement") {
                    count -= 1
                }

                Button("Increment") {
                    count += 1
                }
            }
        }
        .padding(40)
    }
}
