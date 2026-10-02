package main

import (
	"context"
	"testing"
	"time"
)

func TestFailedRegistrationCallbackIsNotConfirmation(t *testing.T) {
	regConfirmed <- -65537
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := waitForRegistration(ctx, make(chan error)); err == nil {
		t.Fatal("failed callback was accepted as registration")
	}
}
