package main

import (
	"net"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestPublisherRefusesOccupiedHTTPPort(t *testing.T) {
	if os.Getenv("TIME_PUBLISHER_TEST_MAIN") == "1" {
		main()
		return
	}
	if firstIPv4() == "" {
		t.Skip("needs a multicast IPv4 interface")
	}
	listener, err := net.Listen("tcp4", ":8080")
	if err != nil {
		t.Skipf("cannot reserve the sample port: %v", err)
	}
	defer listener.Close()
	cmd := exec.Command(os.Args[0], "-test.run=^TestPublisherRefusesOccupiedHTTPPort$")
	cmd.Env = append(os.Environ(), "TIME_PUBLISHER_TEST_MAIN=1")
	output, err := cmd.CombinedOutput()
	if exit, ok := err.(*exec.ExitError); !ok || exit.ExitCode() != 1 {
		t.Fatalf("occupied port must exit 1: err=%v output=%s", err, output)
	}
	if !strings.Contains(string(output), "HTTP listener:") || strings.Contains(string(output), "published ") {
		t.Fatalf("failed listener must never advertise a service: %s", output)
	}
}
