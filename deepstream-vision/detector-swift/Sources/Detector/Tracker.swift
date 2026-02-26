// Tracker.swift
// SORT-style IoU multi-object tracker for DeepStream Vision.
//
// Mirrors the NvDCF tracker_config.yml settings:
//   associationMatcherType: 0  — GREEDY matching
//   checkClassMatch: 1         — same-class only association
//   stateEstimatorType: 1      — constant-velocity Kalman filter

// ---------------------------------------------------------------------------
// MARK: - TrackerConfig
// ---------------------------------------------------------------------------

/// Configuration that mirrors the tracker_config.yml parameters used by NvDCF.
struct TrackerConfig: Sendable {
    /// Maximum number of live tracks per stream.
    var maxTargetsPerStream: Int = 150

    /// Minimum IoU overlap required before an existing track can claim a
    /// detection.  Detections with lower IoU start a new tentative track.
    var minIouDiff4NewTarget: Float = 0.5

    /// Number of consecutive frames a track must receive detections before it
    /// transitions from `.tentative` to `.confirmed`.
    var probationAge: Int = 3

    /// Maximum number of consecutive missed frames before a confirmed track is
    /// removed (shadow-tracking period).
    var maxShadowTrackingAge: Int = 30

    /// Unconfirmed tracks are removed after this many missed frames.
    var earlyTerminationAge: Int = 1
}

// ---------------------------------------------------------------------------
// MARK: - TrackState
// ---------------------------------------------------------------------------

enum TrackState: Sendable {
    /// Newly created; not yet confirmed by enough detections.
    case tentative
    /// Has received detections for at least `probationAge` frames.
    case confirmed
    /// Lost detection; being predicted forward, will be pruned if too old.
    case lost
}

// ---------------------------------------------------------------------------
// MARK: - Internal BBox helper
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box using top-left origin, mirroring Detection layout.
/// Internal to this file only.
struct BBox: Sendable {
    var x: Float
    var y: Float
    var width: Float
    var height: Float

    /// Intersection-over-Union with another box.  Returns 0 when boxes do not
    /// overlap.
    func iou(with other: BBox) -> Float {
        let interX1 = max(x, other.x)
        let interY1 = max(y, other.y)
        let interX2 = min(x + width, other.x + other.width)
        let interY2 = min(y + height, other.y + other.height)

        let interW = interX2 - interX1
        let interH = interY2 - interY1
        guard interW > 0, interH > 0 else { return 0 }

        let intersection = interW * interH
        let areaA = width * height
        let areaB = other.width * other.height
        let union = areaA + areaB - intersection
        guard union > 0 else { return 0 }
        return intersection / union
    }
}

// ---------------------------------------------------------------------------
// MARK: - KalmanFilter2D
// ---------------------------------------------------------------------------

/// Constant-velocity Kalman filter operating on bounding-box state.
///
/// State vector (8 components):
///   `[cx, cy, w, h, vx, vy, vw, vh]`
///   where (cx, cy) is the box centre, (w, h) are width and height, and the
///   v-components are their respective first derivatives.
///
/// Measurement vector (4 components):
///   `[cx, cy, w, h]` — the observed bounding box from the detector.
///
/// Noise parameters come from tracker_config.yml:
///   processNoiseVar4Loc  = 2.0   -> processNoiseVarPosition
///   processNoiseVar4Vel  = 0.1   -> processNoiseVarVelocity
///   measurementNoiseVar4Detector = 4.0
struct KalmanFilter2D: Sendable {

    // MARK: Noise parameters

    /// Process noise variance for position/size dimensions (sigma^2 for cx, cy, w, h).
    var processNoiseVarPosition: Float = 2.0

    /// Process noise variance for velocity dimensions (sigma^2 for vx, vy, vw, vh).
    var processNoiseVarVelocity: Float = 0.1

    /// Measurement noise variance for detector observations (sigma^2 for cx, cy, w, h).
    var measurementNoiseVar: Float = 4.0

    // MARK: State

    /// State estimate: [cx, cy, w, h, vx, vy, vw, vh].
    private(set) var x: (Float, Float, Float, Float, Float, Float, Float, Float)

    /// Error covariance matrix P (8x8), stored row-major as a flat 64-element array.
    private var p: [Float]

    // MARK: Initialisation

    /// Initialise from the first detection.  Velocities start at zero and the
    /// initial covariance is inflated to reflect the uncertainty.
    init(measurement box: BBox) {
        let (cx, cy, w, h) = Self.bboxToCxCy(box)
        x = (cx, cy, w, h, 0, 0, 0, 0)

        // P = diag([2*qPos, 2*qPos, 2*qPos, 2*qPos, 100, 100, 100, 100])
        // A large initial velocity uncertainty prevents the filter from being
        // overconfident about velocity on the very first frame.
        let pPos: Float = 2.0 * 2.0   // 2 x default processNoiseVarPosition
        let pVel: Float = 100.0
        p = [Float](repeating: 0, count: 64)
        p[0 * 8 + 0] = pPos
        p[1 * 8 + 1] = pPos
        p[2 * 8 + 2] = pPos
        p[3 * 8 + 3] = pPos
        p[4 * 8 + 4] = pVel
        p[5 * 8 + 5] = pVel
        p[6 * 8 + 6] = pVel
        p[7 * 8 + 7] = pVel
    }

    // MARK: Predict

    /// Advance the state by one frame using the constant-velocity model.
    ///
    /// State transition:  x' = F * x   where F = [[I4, I4], [0, I4]].
    /// Covariance update: P' = F * P * F^T + Q.
    ///
    /// - Returns: The predicted bounding box (top-left origin).
    @discardableResult
    mutating func predict() -> BBox {
        let (cx, cy, w, h, vx, vy, vw, vh) = x
        x = (cx + vx, cy + vy, w + vw, h + vh, vx, vy, vw, vh)
        p = Self.predictCovariance(p,
                                   qPos: processNoiseVarPosition,
                                   qVel: processNoiseVarVelocity)
        return currentBBox
    }

    // MARK: Update

    /// Correct the state with a new detector measurement (observed bounding box).
    ///
    /// Standard Kalman update:
    ///   S = H * P * H^T + R          (innovation covariance, 4x4)
    ///   K = P * H^T * S^-1           (Kalman gain, 8x4)
    ///   x = x + K * (z - H * x)     (state correction)
    ///   P = (I - K * H) * P          (covariance correction)
    ///
    /// H is the 4x8 observation matrix that selects [cx, cy, w, h] from the state.
    mutating func update(measurement box: BBox) {
        let (mcx, mcy, mw, mh) = Self.bboxToCxCy(box)

        // Innovation: y = z - H*x  (H selects the first 4 state components).
        let y4: [Float] = [
            mcx - x.0,
            mcy - x.1,
            mw  - x.2,
            mh  - x.3,
        ]

        // S = H * P * H^T + R
        // H*P*H^T is the top-left 4x4 block of P; R = r * I4.
        var s = [Float](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                s[i * 4 + j] = p[i * 8 + j]
            }
            s[i * 4 + i] += measurementNoiseVar
        }

        // Invert S (4x4 symmetric positive-definite, Gauss-Jordan).
        guard let sInv = invert4x4(s) else {
            // Degenerate covariance — skip update and keep the prediction.
            return
        }

        // Kalman gain K (8x4) = P * H^T * S^-1
        // P * H^T is the left 8x4 block of P (columns 0..3).
        var k = [Float](repeating: 0, count: 32)
        for row in 0..<8 {
            for col in 0..<4 {
                var val: Float = 0
                for m in 0..<4 {
                    val += p[row * 8 + m] * sInv[m * 4 + col]
                }
                k[row * 4 + col] = val
            }
        }

        // State update: x = x + K * y
        var xArr: [Float] = [x.0, x.1, x.2, x.3, x.4, x.5, x.6, x.7]
        for row in 0..<8 {
            for col in 0..<4 {
                xArr[row] += k[row * 4 + col] * y4[col]
            }
        }
        x = (xArr[0], xArr[1], xArr[2], xArr[3],
             xArr[4], xArr[5], xArr[6], xArr[7])

        // Covariance update: P = (I - K * H) * P
        //
        // (K * H)[row, col] = K[row, col] when col < 4, else 0.
        // Therefore (I - K*H)[row, m] = delta(row, m) - (m < 4 ? K[row,m] : 0).
        //
        // new_P[row, col] = sum_m (I - K*H)[row, m] * P[m, col]
        var newP = [Float](repeating: 0, count: 64)
        for row in 0..<8 {
            for col in 0..<8 {
                var val: Float = 0
                for m in 0..<8 {
                    let ikH: Float = (row == m ? 1 : 0) - (m < 4 ? k[row * 4 + m] : 0)
                    val += ikH * p[m * 8 + col]
                }
                newP[row * 8 + col] = val
            }
        }

        // Symmetrise P = (P + P^T) / 2 to prevent numerical drift from
        // accumulating across many predict/update cycles.
        for row in 0..<8 {
            for col in (row + 1)..<8 {
                let avg = (newP[row * 8 + col] + newP[col * 8 + row]) * 0.5
                newP[row * 8 + col] = avg
                newP[col * 8 + row] = avg
            }
        }
        p = newP
    }

    // MARK: Helpers

    /// The current estimated state as a top-left-origin bounding box.
    var currentBBox: BBox {
        let cx = x.0, cy = x.1
        let w = max(x.2, 1), h = max(x.3, 1)   // guard against degenerate size
        return BBox(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
    }

    private static func bboxToCxCy(_ b: BBox) -> (Float, Float, Float, Float) {
        (b.x + b.width / 2, b.y + b.height / 2, b.width, b.height)
    }

    // MARK: Covariance prediction  F * P * F^T + Q
    //
    // For the constant-velocity model, the state transition matrix F is:
    //
    //   F = [ I4  I4 ]   (I4 = 4x4 identity)
    //       [ 0   I4 ]
    //
    // Partitioning P into 4x4 blocks A (top-left), B (top-right),
    // C (bottom-left), D (bottom-right):
    //
    //   F * P * F^T = [ A + B + C + D   B + D ]
    //                 [ C + D           D     ]
    //
    // (For a symmetric P, C = B^T, but we read both to tolerate rounding drift.)
    //
    // Process noise: Q = diag(qPos x4, qVel x4).

    private static func predictCovariance(
        _ p: [Float],
        qPos: Float,
        qVel: Float
    ) -> [Float] {
        var out = [Float](repeating: 0, count: 64)

        for i in 0..<4 {
            for j in 0..<4 {
                let a = p[(i    ) * 8 + (j    )]   // A block
                let b = p[(i    ) * 8 + (j + 4)]   // B block
                let c = p[(i + 4) * 8 + (j    )]   // C block
                let d = p[(i + 4) * 8 + (j + 4)]   // D block

                out[(i    ) * 8 + (j    )] = a + b + c + d
                out[(i    ) * 8 + (j + 4)] = b + d
                out[(i + 4) * 8 + (j    )] = c + d
                out[(i + 4) * 8 + (j + 4)] = d
            }
        }

        // Add process noise on the diagonal.
        for i in 0..<4 { out[i * 8 + i] += qPos }
        for i in 4..<8 { out[i * 8 + i] += qVel }

        return out
    }

    // MARK: 4x4 matrix inversion (Gauss-Jordan with partial pivoting)

    private func invert4x4(_ m: [Float]) -> [Float]? {
        var a   = m
        var inv = [Float](repeating: 0, count: 16)
        for i in 0..<4 { inv[i * 4 + i] = 1 }   // start as identity

        for col in 0..<4 {
            // Partial pivoting for numerical stability.
            var pivotRow = col
            for row in (col + 1)..<4 {
                if abs(a[row * 4 + col]) > abs(a[pivotRow * 4 + col]) {
                    pivotRow = row
                }
            }
            if pivotRow != col {
                for j in 0..<4 {
                    a.swapAt(col * 4 + j,   pivotRow * 4 + j)
                    inv.swapAt(col * 4 + j, pivotRow * 4 + j)
                }
            }

            let diag = a[col * 4 + col]
            guard abs(diag) > 1e-10 else { return nil }   // singular matrix

            let invDiag = 1.0 / diag
            for j in 0..<4 {
                a[col * 4 + j]   *= invDiag
                inv[col * 4 + j] *= invDiag
            }

            // Eliminate all other rows in this column.
            for row in 0..<4 where row != col {
                let factor = a[row * 4 + col]
                for j in 0..<4 {
                    a[row * 4 + j]   -= factor * a[col * 4 + j]
                    inv[row * 4 + j] -= factor * inv[col * 4 + j]
                }
            }
        }

        return inv
    }
}

// ---------------------------------------------------------------------------
// MARK: - Track
// ---------------------------------------------------------------------------

struct Track: Sendable {
    /// Unique identifier assigned at creation and never reused.
    let id: Int

    /// Kalman filter that maintains the state estimate for this track.
    var kalmanFilter: KalmanFilter2D

    /// Class index of the tracked object.  Only detections with a matching
    /// classId are associated to this track (mirrors `checkClassMatch: 1`).
    let classId: Int

    /// Most recent detection confidence score.
    var confidence: Float

    /// Total number of frames this track has existed (including missed frames).
    var age: Int

    /// Number of frames in which the track received a matched detection.
    var hitCount: Int

    /// Number of consecutive frames without a matching detection.
    var missCount: Int

    /// Lifecycle state of the track.
    var state: TrackState

    /// Current predicted bounding box (top-left origin) from the Kalman filter.
    fileprivate var bbox: BBox { kalmanFilter.currentBBox }
}

// ---------------------------------------------------------------------------
// MARK: - IOUTracker
// ---------------------------------------------------------------------------

/// SORT-style greedy IoU multi-object tracker.
///
/// Corresponds to the NvDCF configuration in tracker_config.yml:
///   - `associationMatcherType: 0`  (GREEDY — faster than GLOBAL/Hungarian)
///   - `checkClassMatch: 1`         (only associate tracks and detections of
///                                   the same class)
///   - `stateEstimatorType: 1`      (constant-velocity Kalman filter)
struct IOUTracker: Sendable {
    var tracks: [Track] = []
    var nextTrackId: Int = 1
    let config: TrackerConfig

    init(config: TrackerConfig = TrackerConfig()) {
        self.config = config
    }

    // MARK: Main update

    /// Process a new set of detections for one frame and return the detections
    /// with `trackId` assigned.
    ///
    /// Steps (following NvDCF logic):
    ///   1. Predict all existing tracks forward via constant-velocity Kalman.
    ///   2. Greedy IoU matching (same class only).
    ///   3. Kalman-correct matched tracks; update hit/miss counts.
    ///   4. Tentative -> confirmed after `probationAge` consecutive hits.
    ///   5. Create new tentative tracks for unmatched detections.
    ///   6. Increment miss count for unmatched tracks; mark confirmed tracks lost.
    ///   7. Prune dead tracks.
    ///   8. Hard cap at `maxTargetsPerStream`.
    ///
    /// - Parameter detections: Raw detections from the YOLO post-processor.
    /// - Returns: Same detections with `trackId` set for matched / new tracks.
    mutating func update(detections: [Detection]) -> [Detection] {

        // Step 1: Predict every existing track one frame forward.
        for i in tracks.indices {
            tracks[i].kalmanFilter.predict()
            tracks[i].age += 1
        }

        // Step 2: Greedy IoU matching.
        // assignment[detectionIndex] = trackIndex
        let assignment = greedyMatch(detections: detections)

        // Step 3 & 4: Update matched tracks, annotate output detections.
        var result = detections
        var matchedTrackIndices = Set<Int>()

        for (detIdx, trackIdx) in assignment {
            let det = detections[detIdx]
            let detBBox = BBox(x: det.x, y: det.y,
                               width: det.width, height: det.height)

            tracks[trackIdx].kalmanFilter.update(measurement: detBBox)
            tracks[trackIdx].confidence = det.confidence
            tracks[trackIdx].hitCount  += 1
            tracks[trackIdx].missCount  = 0

            switch tracks[trackIdx].state {
            case .tentative:
                if tracks[trackIdx].hitCount >= config.probationAge {
                    tracks[trackIdx].state = .confirmed
                }
            case .lost:
                // Re-detected after shadow-tracking period; reconfirm.
                tracks[trackIdx].state = .confirmed
            case .confirmed:
                break
            }

            result[detIdx].trackId = tracks[trackIdx].id
            matchedTrackIndices.insert(trackIdx)
        }

        // Step 5: New tentative tracks for unmatched detections.
        let matchedDetIndices = Set(assignment.keys)
        for detIdx in detections.indices where !matchedDetIndices.contains(detIdx) {
            guard tracks.count < config.maxTargetsPerStream else { break }
            let det = detections[detIdx]
            let bbox = BBox(x: det.x, y: det.y,
                            width: det.width, height: det.height)
            let newTrack = Track(
                id: nextTrackId,
                kalmanFilter: KalmanFilter2D(measurement: bbox),
                classId: det.classId,
                confidence: det.confidence,
                age: 1,
                hitCount: 1,
                missCount: 0,
                state: .tentative
            )
            nextTrackId += 1
            tracks.append(newTrack)
            result[detIdx].trackId = newTrack.id
        }

        // Step 6: Increment miss count for unmatched tracks.
        for trackIdx in tracks.indices where !matchedTrackIndices.contains(trackIdx) {
            tracks[trackIdx].missCount += 1
            if tracks[trackIdx].state == .confirmed {
                tracks[trackIdx].state = .lost
            }
        }

        // Step 7: Prune dead tracks.
        tracks.removeAll { track in
            switch track.state {
            case .tentative:
                // Kill unconfirmed tracks quickly (earlyTerminationAge misses).
                return track.missCount > config.earlyTerminationAge
            case .confirmed, .lost:
                // Confirmed / lost tracks get the full shadow-tracking window.
                return track.missCount > config.maxShadowTrackingAge
            }
        }

        // Step 8: Hard cap — keep the freshest tracks if over the limit.
        if tracks.count > config.maxTargetsPerStream {
            tracks.sort { $0.missCount < $1.missCount }
            tracks = Array(tracks.prefix(config.maxTargetsPerStream))
        }

        return result
    }

    // MARK: Greedy matching

    /// Match existing track predictions to new detections using greedy IoU
    /// assignment, mirroring `associationMatcherType: 0` (GREEDY).
    ///
    /// Algorithm:
    ///   1. Enumerate all (detection, track) pairs of the same class that meet
    ///      the IoU threshold.
    ///   2. Sort candidates by IoU descending.
    ///   3. Greedily assign — each detection and each track appears at most once.
    ///
    /// Complexity: O(D * T) pair enumeration + O(D*T * log(D*T)) sort,
    /// bounded in practice by maxTargetsPerStream = 150.
    ///
    /// - Returns: `[detectionIndex: trackIndex]` for every matched pair.
    private func greedyMatch(detections: [Detection]) -> [Int: Int] {
        guard !tracks.isEmpty, !detections.isEmpty else { return [:] }

        struct Candidate {
            let detIdx: Int
            let trackIdx: Int
            let iou: Float
        }

        // Collect all candidate pairs that pass the class and IoU gate.
        var candidates: [Candidate] = []
        candidates.reserveCapacity(detections.count)

        for (detIdx, det) in detections.enumerated() {
            let detBBox = BBox(x: det.x, y: det.y,
                               width: det.width, height: det.height)
            for (trackIdx, track) in tracks.enumerated() {
                guard track.classId == det.classId else { continue }
                let iou = track.bbox.iou(with: detBBox)
                guard iou >= config.minIouDiff4NewTarget else { continue }
                candidates.append(Candidate(detIdx: detIdx,
                                            trackIdx: trackIdx,
                                            iou: iou))
            }
        }

        // Sort strongest IoU matches first so greedy assignment is near-optimal.
        let sorted = candidates.sorted { $0.iou > $1.iou }

        var assignment: [Int: Int] = [:]
        var usedDetections = Set<Int>()
        var usedTracks     = Set<Int>()

        for candidate in sorted {
            guard !usedDetections.contains(candidate.detIdx),
                  !usedTracks.contains(candidate.trackIdx) else { continue }
            assignment[candidate.detIdx] = candidate.trackIdx
            usedDetections.insert(candidate.detIdx)
            usedTracks.insert(candidate.trackIdx)
        }

        return assignment
    }
}
