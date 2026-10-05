// time-browser: browses _wendytime._tcp using DNS-SD; every 10
// seconds connects to each known instance and logs its answer and host.
// Prints records as they are discovered and lost (TTL expiry).
//
// Records arrive identically from mesh peers (agent catalog projection) and
// physical-LAN devices (agent LAN projection) with no app-side distinction.
package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/hashicorp/mdns"
)

const serviceType = "_wendytime._tcp"

// expireAfter bounds how long an unseen instance is tracked (record TTL 120s).
const expireAfter = 130 * time.Second

type tracked struct {
	name    string
	host    string
	address string
	port    int
	last    time.Time
}

type tracker struct {
	service string
	mu      sync.Mutex
	m       map[string]*tracked
}

func (t *tracker) seen(e *mdns.ServiceEntry) {
	if e.AddrV4 == nil || e.Port == 0 {
		return
	}
	// The agent projects LAN records alongside mesh records; only track
	// our own service type so lab LAN junk (printers, workstations)
	// doesn't fill the poll loop with doomed dials.
	service := t.service
	if service == "" {
		service = serviceType
	}
	name := strings.ToLower(e.Name)
	if !strings.HasSuffix(name, service+".local.") || service == "_http._tcp" && !strings.HasPrefix(name, "wendytimeweb-") {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	key := strings.ToLower(e.Name)
	cur, found := t.m[key]
	if !found {
		t.m[key] = &tracked{name: e.Name, host: e.Host, address: e.AddrV4.String(), port: e.Port, last: time.Now()}
		fmt.Printf("time-browser: discovered %s\n", e.Name)
		return
	}
	cur.last = time.Now()
	cur.host, cur.address, cur.port = e.Host, e.AddrV4.String(), e.Port
}

func (t *tracker) sweep() {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now()
	for k, cur := range t.m {
		if now.Sub(cur.last) > expireAfter {
			delete(t.m, k)
			fmt.Printf("time-browser: lost %s (expired)\n", cur.name)
		}
	}
}

func (t *tracker) snapshot() []*tracked {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]*tracked, 0, len(t.m))
	for _, cur := range t.m {
		if cur.address != "" {
			copy := *cur
			out = append(out, &copy)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out
}

// multicastInterface picks the container interface for mDNS (first
// up+multicast non-loopback IPv4). The library joins multicast per
// interface; an explicit choice avoids dead (e.g. IPv6-less) stacks.
func multicastInterface() *net.Interface {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagMulticast == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			var ip net.IP
			switch v := a.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip != nil && !ip.IsLoopback() && ip.To4() != nil {
				return &iface
			}
		}
	}
	return nil
}

func main() {
	iface := multicastInterface()
	if iface != nil {
		fmt.Printf("time-browser: multicast interface %s\n", iface.Name)
	} else {
		fmt.Fprintln(os.Stderr, "time-browser: platform networking is not ready: no multicast IPv4 interface")
		os.Exit(1)
	}
	service := os.Getenv("TIME_BROWSER_SERVICE_TYPE")
	if service == "" {
		service = serviceType
	}
	if service != serviceType && service != "_http._tcp" {
		fmt.Fprintln(os.Stderr, "time-browser: TIME_BROWSER_SERVICE_TYPE must be _wendytime._tcp or _http._tcp")
		os.Exit(1)
	}
	tr := &tracker{m: map[string]*tracked{}, service: service}
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	fmt.Printf("time-browser: browsing %s.local (poll every 10s)\n", service)
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}
			entries := make(chan *mdns.ServiceEntry, 16)
			go func() {
				if err := queryServices(ctx, iface, service, entries, 4*time.Second); err != nil && ctx.Err() == nil {
					fmt.Fprintf(os.Stderr, "time-browser: browse: %v\n", err)
					os.Exit(1)
				}
				close(entries)
			}()
			for e := range entries {
				tr.seen(e)
			}
			tr.sweep()
		}
	}()

	client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{Proxy: nil}}
	time.Sleep(3 * time.Second)
	for {
		select {
		case <-ctx.Done():
			fmt.Println("time-browser: exiting")
			return
		default:
		}
		peers := tr.snapshot()
		if len(peers) == 0 {
			fmt.Println("time-browser: poll: no resolved peers yet")
		}
		for _, peer := range peers {
			url := fmt.Sprintf("http://%s:%d/", peer.address, peer.port)
			resp, err := client.Get(url)
			if err != nil {
				fmt.Printf("time-browser: poll: host=%s addr=%s:%d FAILED %v\n", peer.name, peer.address, peer.port, err)
				continue
			}
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			resp.Body.Close()
			fmt.Printf("time-browser: poll: host=%s addr=%s:%d status=%d answer=%s\n",
				peer.name, peer.address, peer.port, resp.StatusCode, strings.TrimSpace(string(body)))
		}
		select {
		case <-ctx.Done():
			fmt.Println("time-browser: exiting")
			return
		case <-time.After(10 * time.Second):
		}
	}
}
