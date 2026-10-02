package main

import (
	"github.com/miekg/dns"
	"net"
	"testing"
)

func TestBrowseCompletesEveryInstanceAndSharedHost(t *testing.T) {
	msg := new(dns.Msg)
	msg.Response = true
	for _, name := range []string{"one", "two", "three"} {
		instance := name + "._http._tcp.local."
		for _, text := range []string{"_http._tcp.local. 120 IN PTR " + instance, instance + " 120 IN SRV 0 0 8081 shared.local.", instance + " 120 IN TXT \"path=/\""} {
			rr, err := dns.NewRR(text)
			if err != nil {
				t.Fatal(err)
			}
			msg.Answer = append(msg.Answer, rr)
		}
	}
	address, _ := dns.NewRR("shared.local. 120 IN A 10.99.2.28")
	for _, split := range []bool{false, true} {
		records := newBrowseRecords()
		response := msg.Copy()
		if !split {
			response.Answer = append([]dns.RR{address}, response.Answer...)
		}
		got := records.observe(response, "_http._tcp")
		if split {
			if len(got) != 0 {
				t.Fatal("emitted unresolved service")
			}
			response.Answer = []dns.RR{address}
			got = records.observe(response, "_http._tcp")
		}
		if len(got) != 3 {
			t.Fatalf("split=%t: got %d, want every service", split, len(got))
		}
		for _, entry := range got {
			if !entry.AddrV4.Equal(net.IPv4(10, 99, 2, 28)) || entry.Port != 8081 {
				t.Fatalf("wrong resolved endpoint: %+v", entry)
			}
		}
		if len(records.observe(response, "_http._tcp")) != 0 {
			t.Fatal("duplicate emission in one browse")
		}
	}
}
