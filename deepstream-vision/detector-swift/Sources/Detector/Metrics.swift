import Synchronization

// MARK: - MetricKey

/// Identifies a unique time series by metric name and a sorted set of label pairs.
struct MetricKey: Hashable, Sendable {
    let name: String
    /// Label pairs stored as sorted (key, value) tuples for stable rendering.
    let labels: [(key: String, value: String)]

    init(name: String, labels: [String: String]) {
        self.name = name
        self.labels = labels.sorted { $0.key < $1.key }
    }

    // Manual Hashable so the tuple array hashes correctly.
    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
        for pair in labels {
            hasher.combine(pair.key)
            hasher.combine(pair.value)
        }
    }

    static func == (lhs: MetricKey, rhs: MetricKey) -> Bool {
        guard lhs.name == rhs.name, lhs.labels.count == rhs.labels.count else { return false }
        for (l, r) in zip(lhs.labels, rhs.labels) {
            if l.key != r.key || l.value != r.value { return false }
        }
        return true
    }

    /// Renders `{key="value",...}` or empty string when there are no labels.
    func labelString() -> String {
        guard !labels.isEmpty else { return "" }
        let pairs = labels.map { "\($0.key)=\"\($0.value)\"" }.joined(separator: ",")
        return "{\(pairs)}"
    }
}

// MARK: - Storage types

private struct GaugeStorage: Sendable {
    var value: Double = 0.0
}

private struct CounterStorage: Sendable {
    var value: Double = 0.0
}

private struct HistogramStorage: Sendable {
    /// Upper bounds for each bucket, NOT including +Inf (added at render time).
    let bounds: [Double]
    /// counts[i] is the number of observations <= bounds[i]. Length == bounds.count.
    var counts: [UInt64]
    var sum: Double = 0.0
    var count: UInt64 = 0

    init(bounds: [Double]) {
        self.bounds = bounds.sorted()
        self.counts = [UInt64](repeating: 0, count: bounds.count)
    }

    mutating func observe(_ value: Double) {
        sum += value
        count += 1
        // Increment only the first (smallest) bucket the value fits into.
        // render() accumulates these into cumulative Prometheus buckets.
        for (i, bound) in bounds.enumerated() {
            if value <= bound {
                counts[i] += 1
                break
            }
        }
    }
}

// MARK: - Metric descriptors (registered once, used for # HELP / # TYPE output)

private struct MetricDescriptor: Sendable {
    enum Kind: String, Sendable {
        case gauge
        case counter
        case histogram
    }

    let name: String
    let help: String
    let kind: Kind
    /// Bucket bounds for histograms; empty for gauges/counters.
    let buckets: [Double]
}

// MARK: - Public metric handles

/// A thread-safe handle to a single gauge time series.
struct GaugeMetric: Sendable {
    fileprivate let key: MetricKey
    fileprivate let registry: MetricsRegistry

    func set(_ value: Double) {
        registry.setGauge(key: key, value: value)
    }

    func get() -> Double {
        registry.getGauge(key: key)
    }
}

/// A thread-safe handle to a single counter time series.
struct CounterMetric: Sendable {
    fileprivate let key: MetricKey
    fileprivate let registry: MetricsRegistry

    func inc(by amount: Double = 1.0) {
        registry.incCounter(key: key, by: amount)
    }

    func value() -> Double {
        registry.getCounter(key: key)
    }
}

/// A thread-safe handle to a single histogram time series.
struct HistogramMetric: Sendable {
    fileprivate let key: MetricKey
    fileprivate let registry: MetricsRegistry

    func observe(_ value: Double) {
        registry.observeHistogram(key: key, value: value)
    }
}

// MARK: - Registry state (all mutable state behind a Mutex)

private struct RegistryState: Sendable {
    /// Ordered list of descriptors so render output is deterministic.
    var descriptorOrder: [String] = []
    var descriptors: [String: MetricDescriptor] = [:]

    var gauges: [MetricKey: GaugeStorage] = [:]
    var counters: [MetricKey: CounterStorage] = [:]
    var histograms: [MetricKey: HistogramStorage] = [:]
}

// MARK: - MetricsRegistry

/// A self-contained, Prometheus-compatible metrics registry.
///
/// All metric handles returned by this registry are safe to use from any
/// concurrency domain. The registry uses a `Mutex` so that reads and writes
/// are atomic without requiring actor hops.
final class MetricsRegistry: Sendable {
    private let state: Mutex<RegistryState>

    init() {
        state = Mutex(RegistryState())
    }

    // MARK: Registration helpers

    private func registerDescriptor(
        name: String,
        help: String,
        kind: MetricDescriptor.Kind,
        buckets: [Double] = []
    ) {
        state.withLock { s in
            guard s.descriptors[name] == nil else { return }
            s.descriptors[name] = MetricDescriptor(
                name: name, help: help, kind: kind, buckets: buckets
            )
            s.descriptorOrder.append(name)
        }
    }

    // MARK: Public factory methods

    /// Returns a `GaugeMetric` for the given name and labels.
    /// Registering the same name + labels combination twice returns the same logical series.
    func gauge(
        _ name: String,
        help: String = "",
        labels: [String: String] = [:]
    ) -> GaugeMetric {
        let key = MetricKey(name: name, labels: labels)
        registerDescriptor(name: name, help: help, kind: .gauge)
        state.withLock { s in
            if s.gauges[key] == nil {
                s.gauges[key] = GaugeStorage()
            }
        }
        return GaugeMetric(key: key, registry: self)
    }

    /// Returns a `CounterMetric` for the given name and labels.
    func counter(
        _ name: String,
        help: String = "",
        labels: [String: String] = [:]
    ) -> CounterMetric {
        let key = MetricKey(name: name, labels: labels)
        registerDescriptor(name: name, help: help, kind: .counter)
        state.withLock { s in
            if s.counters[key] == nil {
                s.counters[key] = CounterStorage()
            }
        }
        return CounterMetric(key: key, registry: self)
    }

    /// Returns a `HistogramMetric` for the given name and labels.
    /// - Parameter buckets: Upper bounds (exclusive of +Inf, which is always added).
    func histogram(
        _ name: String,
        help: String = "",
        labels: [String: String] = [:],
        buckets: [Double]
    ) -> HistogramMetric {
        let key = MetricKey(name: name, labels: labels)
        registerDescriptor(name: name, help: help, kind: .histogram, buckets: buckets)
        state.withLock { s in
            if s.histograms[key] == nil {
                s.histograms[key] = HistogramStorage(bounds: buckets)
            }
        }
        return HistogramMetric(key: key, registry: self)
    }

    // MARK: Internal mutation (called by metric handles)

    fileprivate func setGauge(key: MetricKey, value: Double) {
        state.withLock { s in
            s.gauges[key, default: GaugeStorage()].value = value
        }
    }

    fileprivate func getGauge(key: MetricKey) -> Double {
        state.withLock { s in s.gauges[key]?.value ?? 0.0 }
    }

    fileprivate func incCounter(key: MetricKey, by amount: Double) {
        state.withLock { s in
            s.counters[key, default: CounterStorage()].value += amount
        }
    }

    fileprivate func getCounter(key: MetricKey) -> Double {
        state.withLock { s in s.counters[key]?.value ?? 0.0 }
    }

    fileprivate func observeHistogram(key: MetricKey, value: Double) {
        state.withLock { s in
            s.histograms[key]?.observe(value)
        }
    }

    // MARK: Prometheus text exposition

    /// Renders the full Prometheus text format.
    /// Designed to be cheap: one lock acquisition to snapshot state, then
    /// string building outside the lock.
    func render() -> String {
        // Snapshot everything under a single lock.
        let (order, descriptors, gauges, counters, histograms): (
            [String],
            [String: MetricDescriptor],
            [MetricKey: GaugeStorage],
            [MetricKey: CounterStorage],
            [MetricKey: HistogramStorage]
        ) = state.withLock { s in
            (s.descriptorOrder, s.descriptors, s.gauges, s.counters, s.histograms)
        }

        var output = ""
        // Reserve a reasonable capacity to avoid repeated reallocations.
        output.reserveCapacity(4096)

        for name in order {
            guard let desc = descriptors[name] else { continue }

            if !desc.help.isEmpty {
                output += "# HELP \(desc.name) \(desc.help)\n"
            }
            output += "# TYPE \(desc.name) \(desc.kind.rawValue)\n"

            switch desc.kind {
            case .gauge:
                let matching = gauges.filter { $0.key.name == name }
                    .sorted { $0.key.labelString() < $1.key.labelString() }
                for (key, storage) in matching {
                    output += "\(name)\(key.labelString()) \(formatDouble(storage.value))\n"
                }

            case .counter:
                let matching = counters.filter { $0.key.name == name }
                    .sorted { $0.key.labelString() < $1.key.labelString() }
                for (key, storage) in matching {
                    output += "\(name)\(key.labelString()) \(formatDouble(storage.value))\n"
                }

            case .histogram:
                let matching = histograms.filter { $0.key.name == name }
                    .sorted { $0.key.labelString() < $1.key.labelString() }
                for (key, storage) in matching {
                    let baseLabelStr = key.labelString()
                    // Compute cumulative bucket counts.
                    var cumulative: UInt64 = 0
                    for (i, bound) in storage.bounds.enumerated() {
                        cumulative += storage.counts[i]
                        let boundLabel = appendLabel(baseLabelStr, key: "le", value: formatBound(bound))
                        output += "\(name)_bucket\(boundLabel) \(cumulative)\n"
                    }
                    // +Inf bucket equals total count.
                    let infLabel = appendLabel(baseLabelStr, key: "le", value: "+Inf")
                    output += "\(name)_bucket\(infLabel) \(storage.count)\n"
                    output += "\(name)_sum\(baseLabelStr) \(formatDouble(storage.sum))\n"
                    output += "\(name)_count\(baseLabelStr) \(storage.count)\n"
                }
            }
        }

        return output
    }

    // MARK: Formatting helpers

    /// Formats a Double for Prometheus output, using integer notation when the
    /// value is a whole number to keep output clean.
    ///
    /// Prometheus text format requires: NaN → "NaN", +Inf → "+Inf", -Inf → "-Inf".
    private func formatDouble(_ value: Double) -> String {
        if value.isNaN { return "NaN" }
        if value.isInfinite { return value > 0 ? "+Inf" : "-Inf" }
        if value == value.rounded() {
            return String(Int64(value))
        }
        return String(value)
    }

    /// Formats a histogram bucket bound. Integer bounds render without decimals.
    private func formatBound(_ bound: Double) -> String {
        if bound == bound.rounded() && !bound.isInfinite {
            return String(Int64(bound))
        }
        return String(bound)
    }

    /// Inserts an extra label into an existing label string.
    /// `{a="1"}` + le="5"  -> `{a="1",le="5"}`
    /// `""`       + le="5"  -> `{le="5"}`
    private func appendLabel(_ existing: String, key: String, value: String) -> String {
        if existing.isEmpty {
            return "{\(key)=\"\(value)\"}"
        }
        // existing ends with '}', insert before the closing brace.
        return String(existing.dropLast()) + ",\(key)=\"\(value)\"}"
    }
}

// MARK: - Pre-defined detector metrics

/// Shared registry for the DeepStream detector process.
let metrics = MetricsRegistry()

// MARK: Latency histogram bucket definition

private let latencyBuckets: [Double] = [5, 10, 20, 50, 100, 200]

// MARK: Gauge metrics

/// Frames per second, per stream.
func fpsGauge(stream: String) -> GaugeMetric {
    metrics.gauge(
        "deepstream_fps",
        help: "Frames per second",
        labels: ["stream": stream]
    )
}

/// GPU memory usage in megabytes (no stream label).
let gpuMemoryMB: GaugeMetric = metrics.gauge(
    "deepstream_gpu_memory_mb",
    help: "GPU memory usage in megabytes"
)

// MARK: Counter metrics

/// Total detection events, per stream and object class.
func detectionsTotalCounter(stream: String, class_: String) -> CounterMetric {
    metrics.counter(
        "deepstream_detections_total",
        help: "Total detections by class",
        labels: ["stream": stream, "class_": class_]
    )
}

/// Total frames processed, per stream.
func framesProcessedCounter(stream: String) -> CounterMetric {
    metrics.counter(
        "deepstream_frames_processed_total",
        help: "Total frames processed",
        labels: ["stream": stream]
    )
}

// MARK: Histogram metrics

/// End-to-end inference latency in milliseconds, per stream.
func inferenceLatencyHistogram(stream: String) -> HistogramMetric {
    metrics.histogram(
        "deepstream_inference_latency_ms",
        help: "Inference latency in milliseconds",
        labels: ["stream": stream],
        buckets: latencyBuckets
    )
}

/// Frame decode latency in milliseconds, per stream.
func decodeLatencyHistogram(stream: String) -> HistogramMetric {
    metrics.histogram(
        "deepstream_decode_latency_ms",
        help: "Frame decode latency in milliseconds",
        labels: ["stream": stream],
        buckets: latencyBuckets
    )
}

/// Pre-processing latency in milliseconds, per stream.
func preprocessLatencyHistogram(stream: String) -> HistogramMetric {
    metrics.histogram(
        "deepstream_preprocess_latency_ms",
        help: "Pre-processing latency in milliseconds",
        labels: ["stream": stream],
        buckets: latencyBuckets
    )
}

/// Post-processing latency in milliseconds, per stream.
func postprocessLatencyHistogram(stream: String) -> HistogramMetric {
    metrics.histogram(
        "deepstream_postprocess_latency_ms",
        help: "Post-processing latency in milliseconds",
        labels: ["stream": stream],
        buckets: latencyBuckets
    )
}

/// Total pipeline latency in milliseconds, per stream.
func totalLatencyHistogram(stream: String) -> HistogramMetric {
    metrics.histogram(
        "deepstream_total_latency_ms",
        help: "Total pipeline latency in milliseconds",
        labels: ["stream": stream],
        buckets: latencyBuckets
    )
}
