// time-publisher: answers "what is the current time?" over HTTP and
// publishes a custom _wendytime._tcp mDNS service for it, using the
// hashicorp/mdns library (raw multicast under the hood; co-binds :5353
// with the agent bridge via SO_REUSEADDR on Linux).
//
// On a Wendy device (wendy.json: network mode mesh + published port) the
// agent observes this multicast on the app container network and signs the
// record into the mesh catalog, exporting it to every mesh peer.
package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/hashicorp/mdns"
)

const (
	port        = 8080
	serviceType = "_wendytime._tcp"
)

func hostname() string {
	if h := os.Getenv("TIME_SERVICE_HOSTNAME"); h != "" {
		return strings.TrimSuffix(h, ".local")
	}
	if h := os.Getenv("WENDY_DEVICE_HOSTNAME"); h != "" {
		return strings.TrimSuffix(h, ".local")
	}
	if h, err := os.Hostname(); err == nil {
		if i := strings.IndexByte(h, '.'); i >= 0 {
			return h[:i]
		}
		return h
	}
	return "unknown"
}

// containerInterface returns the network interface holding ip, so mDNS
// joins the link that carries our records.
func containerInterface(ip string) *net.Interface {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagMulticast == 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			var addr net.IP
			switch v := a.(type) {
			case *net.IPNet:
				addr = v.IP
			case *net.IPAddr:
				addr = v.IP
			}
			if addr != nil && addr.String() == ip {
				return &iface
			}
		}
	}
	return nil
}

// logInterfaces dumps every interface for startup diagnostics.
func logInterfaces() {
	ifaces, err := net.Interfaces()
	if err != nil {
		fmt.Printf("time-publisher: interfaces error: %v\n", err)
		return
	}
	for _, iface := range ifaces {
		addrs, _ := iface.Addrs()
		fmt.Printf("time-publisher: iface %s flags=%s addrs=%v\n", iface.Name, iface.Flags.String(), addrs)
	}
}

// firstIPv4 reads the address assigned before the entrypoint is released.
func firstIPv4() string {
	ifaces, err := net.Interfaces()
	if err == nil {
		best := ""
		bestRank := 9
		rank := func(name string) int {
			for _, p := range []string{"eth", "en", "wl", "wlan"} {
				if strings.HasPrefix(name, p) {
					return 0
				}
			}
			for _, p := range []string{"utun", "docker", "br-", "tailscale"} {
				if strings.HasPrefix(name, p) {
					return 2
				}
			}
			return 1
		}
		for _, iface := range ifaces {
			if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagMulticast == 0 {
				continue
			}
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}
			for _, a := range addrs {
				var addr net.IP
				switch v := a.(type) {
				case *net.IPNet:
					addr = v.IP
				case *net.IPAddr:
					addr = v.IP
				}
				if addr == nil || addr.IsLoopback() || addr.To4() == nil {
					continue
				}
				if r := rank(iface.Name); r < bestRank {
					bestRank, best = r, addr.String()
				}
			}
		}
		return best
	}
	return ""
}

func run() error {
	logInterfaces()
	ip := firstIPv4()
	if ip == "" {
		return fmt.Errorf("container started without a usable IPv4 address")
	}
	host := hostname()
	fmt.Printf("time-publisher: host=%s ip=%s port=%d\n", host, ip, port)
	listener, err := net.Listen("tcp4", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("HTTP listener: %w", err)
	}
	defer listener.Close()

	instance := "WendyTime-" + host
	svc, err := mdns.NewMDNSService(instance, serviceType, "", "", port,
		[]net.IP{net.ParseIP(ip)}, []string{"path=/", "format=json"})
	if err != nil {
		return fmt.Errorf("mdns service: %w", err)
	}
	server, err := mdns.NewServer(&mdns.Config{Zone: svc, Iface: containerInterface(ip)})
	if err != nil {
		return fmt.Errorf("mdns server: %w", err)
	}
	defer server.Shutdown()
	fmt.Printf("time-publisher: published %s.%s.local -> %s:%d\n", instance, serviceType, ip, port)

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		body := fmt.Sprintf("{\"time\":%q,\"host\":%q}\n", time.Now().UTC().Format(time.RFC3339), host)
		fmt.Printf("time-publisher: GET / -> %s\n", strings.TrimSpace(body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
	srv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}
	fmt.Printf("time-publisher: serving http://0.0.0.0:%d/\n", port)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)
	defer signal.Stop(stop)
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-stop:
			srv.Close()
		case <-done:
		}
	}()
	if err := srv.Serve(listener); err != http.ErrServerClosed {
		return fmt.Errorf("HTTP server: %w", err)
	}
	fmt.Println("time-publisher: exiting (records expire by TTL)")
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "time-publisher:", err)
		os.Exit(1)
	}
}
