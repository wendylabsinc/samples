// time-web: "what is the current time?" as HTML on the conventional
// _http._tcp service, published through avahi-compat (dns_sd.h) against a
// container-local avahi-daemon (Docker test) or the agent-provided per-app
// daemon (B2 on device, see wendy.json). No raw-socket mDNS code here.
package main

/*
#cgo pkg-config: avahi-compat-libdns_sd
#include <dns_sd.h>
#include <stdlib.h>
#include <string.h>

extern void goServiceRegistered(int err);
DNSServiceRegisterReply timeWebRegReply(void);
*/
import "C"

import (
	"context"
	"encoding/binary"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

const port = 8081

var regConfirmed = make(chan int, 8)

//export goServiceRegistered
func goServiceRegistered(err C.int) {
	if err != 0 {
		fmt.Fprintf(os.Stderr, "time-web: avahi registration failed: %d\n", int(err))
	} else {
		fmt.Println("time-web: avahi-compat service registered")
	}
	select {
	case regConfirmed <- int(err):
	default:
	}
}

// processResults is the sole owner of DNSServiceProcessResult. Polling keeps
// shutdown bounded even when the daemon sends no further callbacks.
func processResults(ctx context.Context, ref C.DNSServiceRef) error {
	fd := int(C.DNSServiceRefSockFD(ref))
	for ctx.Err() == nil {
		pfds := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
		n, err := unix.Poll(pfds, 200)
		if err == unix.EINTR {
			continue
		}
		if err != nil {
			return fmt.Errorf("avahi poll: %w", err)
		}
		if n == 0 {
			continue
		}
		if pfds[0].Revents&(unix.POLLERR|unix.POLLHUP|unix.POLLNVAL) != 0 {
			return fmt.Errorf("avahi connection closed")
		}
		if result := C.DNSServiceProcessResult(ref); result != C.kDNSServiceErr_NoError {
			return fmt.Errorf("DNSServiceProcessResult: %d", int(result))
		}
	}
	return ctx.Err()
}

// DNS-SD registration is asynchronous: wait for the protocol callback after
// one registration against the daemon prepared before application entry.
func waitForRegistration(ctx context.Context, results <-chan error) error {
	select {
	case code := <-regConfirmed:
		if code != 0 {
			return fmt.Errorf("avahi registration callback: %d", code)
		}
		return nil
	case err := <-results:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

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

func registerAvahi(instance, regtype string, port int, txt map[string]string, cb C.DNSServiceRegisterReply) (C.DNSServiceRef, error) {
	var ref C.DNSServiceRef
	cInstance := C.CString(instance)
	defer C.free(unsafe.Pointer(cInstance))
	cRegtype := C.CString(regtype)
	defer C.free(unsafe.Pointer(cRegtype))
	var txtBuf []byte
	for k, v := range txt {
		entry := k
		if v != "" {
			entry = k + "=" + v
		}
		if len(entry) > 255 {
			entry = entry[:255]
		}
		txtBuf = append(txtBuf, byte(len(entry)))
		txtBuf = append(txtBuf, entry...)
	}
	var txtPtr unsafe.Pointer
	var txtLen C.uint16_t
	if len(txtBuf) > 0 {
		txtPtr = unsafe.Pointer(&txtBuf[0])
		txtLen = C.uint16_t(len(txtBuf))
	}
	var portBE [2]byte
	binary.BigEndian.PutUint16(portBE[:], uint16(port))
	err := C.DNSServiceRegister(&ref, 0, 0,
		cInstance, cRegtype, nil, nil,
		*(*C.uint16_t)(unsafe.Pointer(&portBE[0])),
		txtLen, txtPtr,
		cb,
		nil)
	if err != C.kDNSServiceErr_NoError {
		return nil, fmt.Errorf("DNSServiceRegister: %d", int(err))
	}
	return ref, nil
}

func run() error {
	host := hostname()
	addrs, _ := net.InterfaceAddrs()
	ip := ""
	for _, a := range addrs {
		if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
			ip = ipnet.IP.String()
			break
		}
	}
	if ip == "" {
		return fmt.Errorf("container started without a usable IPv4 address")
	}
	fmt.Printf("time-web: host=%s ip=%s port=%d (avahi-compat)\n", host, ip, port)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()
	startup, stopStartup := context.WithTimeout(ctx, 30*time.Second)
	defer stopStartup()
	instance := "WendyTimeWeb-" + host
	txt := map[string]string{"path": "/", "title": "Wendy Time"}
	fmt.Printf("time-web: registering %s._http._tcp\n", instance)
	ref, err := registerAvahi(instance, "_http._tcp", port, txt, C.timeWebRegReply())
	if err != nil {
		return fmt.Errorf("registration transport unavailable: %w", err)
	}
	results := make(chan error, 1)
	pumpCtx, stopPump := context.WithCancel(ctx)
	pumpDone := make(chan struct{})
	go func() { defer close(pumpDone); results <- processResults(pumpCtx, ref) }()
	defer func() { stopPump(); <-pumpDone; C.DNSServiceRefDeallocate(ref) }()
	if err := waitForRegistration(startup, results); err != nil {
		return fmt.Errorf("registration not confirmed: %w", err)
	}
	stopStartup()

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		now := time.Now().UTC().Format(time.RFC3339)
		fmt.Printf("time-web: GET %s from browser\n", r.URL.Path)
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprintf(w, `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Wendy Time — %s</title></head><body>
<h1>What is the current time?</h1>
<p><strong>%s</strong></p>
<p>served by %s (%s) via avahi-compat</p>
</body></html>
`, host, now, host, ip)
	})
	srv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}
	fmt.Printf("time-web: serving http://0.0.0.0:%d/\n", port)
	shutdown := make(chan error, 1)
	go func() {
		for {
			select {
			case <-ctx.Done():
				shutdown <- nil
				srv.Close()
				return
			case err := <-results:
				fmt.Fprintf(os.Stderr, "time-web: avahi connection lost: %v\n", err)
				shutdown <- fmt.Errorf("avahi connection lost: %w", err)
				srv.Close()
				return
			case code := <-regConfirmed:
				if code != 0 {
					shutdown <- fmt.Errorf("late registration failure: %d", code)
					srv.Close()
					return
				}
			}
		}
	}()
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		return fmt.Errorf("http failed: %w", err)
	}
	fmt.Println("time-web: exiting")
	return <-shutdown
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "time-web:", err)
		os.Exit(1)
	}
}
