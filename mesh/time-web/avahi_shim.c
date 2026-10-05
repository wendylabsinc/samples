// avahi_shim.c — tiny C trampoline: dns_sd.h callbacks cannot be Go
// functions directly, so C receives them and forwards to Go.
#include <dns_sd.h>
#include "_cgo_export.h"

static void regReplyTrampoline(DNSServiceRef sdRef, DNSServiceFlags flags,
                               DNSServiceErrorType errorCode, const char *name,
                               const char *regtype, const char *domain, void *context) {
    (void)sdRef; (void)flags; (void)name; (void)regtype; (void)domain; (void)context;
    goServiceRegistered(errorCode);
}

DNSServiceRegisterReply timeWebRegReply(void) {
    return regReplyTrampoline;
}
