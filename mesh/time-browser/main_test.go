package main

import (
	"net"
	"testing"

	"github.com/hashicorp/mdns"
)

func TestSnapshotSurvivesConcurrentRecordRefresh(t *testing.T) {
	tr := &tracker{m: map[string]*tracked{}}
	e := &mdns.ServiceEntry{Name: "WendyTime-test._wendytime._tcp.local.", AddrV4: net.IPv4(10, 99, 0, 1), Port: 8080}
	tr.seen(e)
	before := tr.snapshot()
	e.AddrV4 = net.IPv4(10, 99, 0, 2)
	tr.seen(e)
	if before[0].address != "10.99.0.1" || tr.snapshot()[0].address != "10.99.0.2" {
		t.Fatal("record refresh mutated an in-flight HTTP poll snapshot")
	}
}

func TestWebBrowseExcludesOtherLANHTTPRecords(t *testing.T) {
	tr := &tracker{m: map[string]*tracked{}, service: "_http._tcp"}
	for _, name := range []string{"printer._http._tcp.local.", "WendyTimeWeb-test._http._tcp.local.", "WendyTime-test._wendytime._tcp.local."} {
		tr.seen(&mdns.ServiceEntry{Name: name, AddrV4: net.IPv4(10, 99, 0, 1), Port: 8081})
	}
	if len(tr.snapshot()) != 1 || tr.snapshot()[0].name != "WendyTimeWeb-test._http._tcp.local." {
		t.Fatal("web observer polled unrelated LAN services")
	}
}
