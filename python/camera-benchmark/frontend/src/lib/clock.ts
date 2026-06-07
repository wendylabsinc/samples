// NTP-style clock offset estimation (Cristian's algorithm).
//
// The browser timestamps frames with Date.now(), the server stamps send_ts_ms
// with its own wall clock. To turn (recv - send) into a real end-to-end latency
// we need the clock offset between the two machines. We estimate it from ping/pong
// round-trips and keep the offset from the LOWEST-RTT sample (least network noise).
//
// The same offset applies to both camera streams (same host), so even residual
// error cancels in the USB-vs-CSI comparison.

export class ClockSync {
  offsetMs = 0
  private bestRtt = Infinity

  /** Call with the t0 we sent and the server's t1 from the pong. */
  onPong(t0: number, t1: number): void {
    const t2 = Date.now()
    const rtt = t2 - t0
    if (rtt < this.bestRtt) {
      this.bestRtt = rtt
      this.offsetMs = (t1 - t0 + (t1 - t2)) / 2
    }
  }

  /** Allow the offset to re-adapt (e.g. after a reconnect or periodically). */
  relax(): void {
    this.bestRtt = Infinity
  }

  /** Convert a server send timestamp to an end-to-end latency in ms. */
  latencyMs(sendTsMs: number): number {
    return Date.now() - sendTsMs - this.offsetMs
  }
}
