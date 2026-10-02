// Run with AddressSanitizer+LeakSanitizer. A second SOI in place of EOI causes
// a fatal libjpeg error after scanline RGB allocation, not just header rejection.
#include "CameraC.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    assert(argc == 2);
    FILE *f = fopen(argv[1], "rb"); assert(f);
    assert(fseek(f, 0, SEEK_END) == 0); long n = ftell(f); assert(n > 4);
    rewind(f); unsigned char *jpeg = malloc(n); assert(jpeg);
    assert(fread(jpeg, 1, n, f) == (size_t)n); fclose(f);
    assert(jpeg[n-2] == 0xff && jpeg[n-1] == 0xd9);
    jpeg[n-1] = 0xd8;
    for (int i = 0; i < 8; ++i) {
        unsigned char *out = NULL; size_t length = 0; char error[256] = {0};
        assert(wendy_camera_thumbnail(jpeg, n, &out, &length, error, sizeof(error)) != 0);
        assert(!out && length == 0);
        assert(strstr(error, "decoding thumbnail source"));
    }
    free(jpeg);
    puts("post-allocation malformed JPEG rejected; sanitizer checks cleanup");
}
