package main

import (
	"context"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/hashicorp/mdns"
	"github.com/miekg/dns"
	"golang.org/x/net/ipv4"
)

// Every service instance keeps its own SRV/TXT. Host addresses are shared by
// all instances targeting that host, regardless of record order or datagram.
type browseRecords struct {
	instances map[string]string
	srv       map[string]*dns.SRV
	txt       map[string]*dns.TXT
	addresses map[string]net.IP
	sent      map[string]bool
}

func newBrowseRecords() *browseRecords {
	return &browseRecords{map[string]string{}, map[string]*dns.SRV{}, map[string]*dns.TXT{}, map[string]net.IP{}, map[string]bool{}}
}

func (r *browseRecords) observe(msg *dns.Msg, service string) []*mdns.ServiceEntry {
	if !msg.Response {
		return nil
	}
	service = strings.ToLower(service) + ".local."
	for _, section := range [][]dns.RR{msg.Answer, msg.Ns, msg.Extra} {
		for _, rr := range section {
			if rr.Header().Class&0x7fff != dns.ClassINET {
				continue
			}
			name := strings.ToLower(rr.Header().Name)
			switch v := rr.(type) {
			case *dns.PTR:
				if name == service && strings.HasSuffix(strings.ToLower(v.Ptr), "."+service) {
					key := strings.ToLower(v.Ptr)
					if v.Hdr.Ttl == 0 {
						delete(r.instances, key)
					} else {
						r.instances[key] = v.Ptr
					}
				}
			case *dns.SRV:
				if strings.HasSuffix(name, "."+service) {
					if v.Hdr.Ttl == 0 {
						delete(r.srv, name)
					} else {
						r.srv[name] = v
					}
				}
			case *dns.TXT:
				if strings.HasSuffix(name, "."+service) {
					if v.Hdr.Ttl == 0 {
						delete(r.txt, name)
					} else {
						r.txt[name] = v
					}
				}
			case *dns.A:
				if v.Hdr.Ttl == 0 {
					delete(r.addresses, name)
				} else {
					r.addresses[name] = append(net.IP(nil), v.A...)
				}
			}
		}
	}
	var entries []*mdns.ServiceEntry
	for key, name := range r.instances {
		srv, txt := r.srv[key], r.txt[key]
		if srv == nil || txt == nil || srv.Port == 0 || r.sent[key] {
			continue
		}
		ip := r.addresses[strings.ToLower(srv.Target)]
		if ip == nil {
			continue
		}
		r.sent[key] = true
		entries = append(entries, &mdns.ServiceEntry{Name: name, Host: srv.Target, Port: int(srv.Port), AddrV4: append(net.IP(nil), ip...), Info: strings.Join(txt.Txt, "|"), InfoFields: append([]string(nil), txt.Txt...)})
	}
	return entries
}

func queryServices(ctx context.Context, iface *net.Interface, service string, entries chan<- *mdns.ServiceEntry, duration time.Duration) error {
	if iface == nil {
		return fmt.Errorf("no multicast IPv4 interface")
	}
	addrs, err := iface.Addrs()
	if err != nil {
		return err
	}
	var address net.IP
	for _, addr := range addrs {
		if ip, ok := addr.(*net.IPNet); ok && ip.IP.To4() != nil {
			address = ip.IP.To4()
			break
		}
	}
	if address == nil {
		return fmt.Errorf("interface %s has no IPv4 address", iface.Name)
	}
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: address})
	if err != nil {
		return err
	}
	defer conn.Close()
	packet := ipv4.NewPacketConn(conn)
	if err = packet.SetMulticastInterface(iface); err != nil {
		return err
	}
	if err = packet.SetMulticastTTL(255); err != nil {
		return err
	}
	group := &net.UDPAddr{IP: net.IPv4(224, 0, 0, 251), Port: 5353}
	multicast, err := net.ListenMulticastUDP("udp4", iface, group)
	if err != nil {
		return err
	}
	defer multicast.Close()
	ctx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()
	messages := make(chan *dns.Msg, 32)
	read := func(socket *net.UDPConn) {
		buffer := make([]byte, 9000)
		for ctx.Err() == nil {
			_ = socket.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
			n, _, err := socket.ReadFromUDP(buffer)
			if err != nil {
				if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
					continue
				}
				return
			}
			msg := new(dns.Msg)
			if msg.Unpack(buffer[:n]) != nil {
				continue
			}
			select {
			case messages <- msg:
			case <-ctx.Done():
				return
			}
		}
	}
	done := make(chan struct{}, 2)
	for _, socket := range []*net.UDPConn{conn, multicast} {
		go func(socket *net.UDPConn) { defer func() { done <- struct{}{} }(); read(socket) }(socket)
	}
	defer func() { cancel(); conn.Close(); multicast.Close(); <-done; <-done }()
	send := func(name string, typ uint16) error {
		msg := new(dns.Msg)
		msg.SetQuestion(name, typ)
		msg.RecursionDesired = false
		data, err := msg.Pack()
		if err != nil {
			return err
		}
		_, err = packet.WriteTo(data, nil, group)
		return err
	}
	if err = send(service+".local.", dns.TypePTR); err != nil {
		return err
	}
	records := newBrowseRecords()
	queried := map[string]bool{}
	for {
		select {
		case <-ctx.Done():
			return nil
		case msg := <-messages:
			for _, entry := range records.observe(msg, service) {
				select {
				case entries <- entry:
				case <-ctx.Done():
					return nil
				}
			}
			for key := range records.instances {
				for _, typ := range []uint16{dns.TypeSRV, dns.TypeTXT} {
					if typ == dns.TypeSRV && records.srv[key] != nil || typ == dns.TypeTXT && records.txt[key] != nil {
						continue
					}
					q := fmt.Sprintf("%s|%d", key, typ)
					if !queried[q] {
						queried[q] = true
						if err = send(key, typ); err != nil {
							return err
						}
					}
				}
				if srv := records.srv[key]; srv != nil {
					host := strings.ToLower(srv.Target)
					if records.addresses[host] == nil && !queried[host] {
						queried[host] = true
						if err = send(host, dns.TypeA); err != nil {
							return err
						}
					}
				}
			}
		}
	}
}
