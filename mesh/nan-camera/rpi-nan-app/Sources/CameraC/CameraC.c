#include "CameraC.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <jpeglib.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <setjmp.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define MAX_BUFFERS 4
#define MAX_JPEG (32U * 1024U * 1024U)

static volatile sig_atomic_t stop_requested;
static void request_stop(int signal_number) { (void)signal_number; stop_requested = 1; }
void wendy_install_stop_handlers(void) {
    struct sigaction action = {0};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);
}
int wendy_stop_requested(void) { return stop_requested != 0; }

struct WendyCamera {
    int fd;
    uint32_t width, height, stride, format;
    unsigned int count;
    struct { void *memory; size_t length; } buffers[MAX_BUFFERS];
    int streaming;
};

struct jpeg_error_guard {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
};

static void jpeg_failed(j_common_ptr cinfo) {
    struct jpeg_error_guard *guard = (struct jpeg_error_guard *)cinfo->err;
    longjmp(guard->jump, 1);
}

static int fail(char *error, size_t capacity, const char *message) {
    if (error && capacity) snprintf(error, capacity, "%s: %s", message, strerror(errno));
    return -1;
}

void wendy_camera_close(WendyCamera *camera) {
    if (!camera) return;
    if (camera->streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(camera->fd, VIDIOC_STREAMOFF, &type);
    }
    for (unsigned int i = 0; i < camera->count; i++)
        if (camera->buffers[i].memory && camera->buffers[i].memory != MAP_FAILED)
            munmap(camera->buffers[i].memory, camera->buffers[i].length);
    if (camera->fd >= 0) close(camera->fd);
    free(camera);
}

void wendy_camera_free_bytes(unsigned char *bytes) { free(bytes); }

int wendy_camera_open(const char *path, uint32_t width, uint32_t height,
                      WendyCamera **out, char *error, size_t error_capacity) {
    *out = NULL;
    WendyCamera *camera = calloc(1, sizeof(*camera));
    if (!camera) return fail(error, error_capacity, "camera allocation");
    camera->fd = -1;
    camera->fd = open(path, O_RDWR | O_CLOEXEC | O_NONBLOCK);
    if (camera->fd < 0) goto failed_open;
    struct v4l2_capability caps;
    if (ioctl(camera->fd, VIDIOC_QUERYCAP, &caps) < 0) goto failed_query;
    uint32_t flags = (caps.capabilities & V4L2_CAP_DEVICE_CAPS) ? caps.device_caps : caps.capabilities;
    if (!(flags & V4L2_CAP_VIDEO_CAPTURE) || !(flags & V4L2_CAP_STREAMING)) {
        errno = ENOTSUP;
        goto failed_query;
    }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(camera->fd, VIDIOC_S_FMT, &fmt) < 0) goto failed_format;
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        if (ioctl(camera->fd, VIDIOC_S_FMT, &fmt) < 0 || fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV)
            goto failed_format;
    }
    camera->width = fmt.fmt.pix.width;
    camera->height = fmt.fmt.pix.height;
    camera->stride = fmt.fmt.pix.bytesperline ? fmt.fmt.pix.bytesperline : camera->width * 2;
    camera->format = fmt.fmt.pix.pixelformat;
    if (!camera->width || !camera->height || camera->width > 4096 || camera->height > 4096) {
        errno = EINVAL;
        goto failed_format;
    }

    struct v4l2_requestbuffers request = {0};
    request.count = MAX_BUFFERS;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (ioctl(camera->fd, VIDIOC_REQBUFS, &request) < 0 || !request.count) goto failed_buffers;
    if (request.count > MAX_BUFFERS) { errno = EINVAL; goto failed_buffers; }
    for (unsigned int i = 0; i < request.count; i++) {
        struct v4l2_buffer buffer = {0};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;
        if (ioctl(camera->fd, VIDIOC_QUERYBUF, &buffer) < 0) goto failed_buffers;
        void *memory = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                            camera->fd, buffer.m.offset);
        if (memory == MAP_FAILED) goto failed_buffers;
        camera->buffers[i].memory = memory;
        camera->buffers[i].length = buffer.length;
        camera->count++;
        if (ioctl(camera->fd, VIDIOC_QBUF, &buffer) < 0) goto failed_buffers;
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->fd, VIDIOC_STREAMON, &type) < 0) goto failed_buffers;
    camera->streaming = 1;
    *out = camera;
    return 0;

failed_buffers: fail(error, error_capacity, "V4L2 buffers/stream"); goto cleanup;
failed_format: fail(error, error_capacity, "V4L2 format"); goto cleanup;
failed_query: fail(error, error_capacity, "V4L2 capture capability"); goto cleanup;
failed_open: fail(error, error_capacity, "opening camera");
cleanup: wendy_camera_close(camera); return -1;
}

static uint8_t clip(int value) {
    return (uint8_t)(value < 0 ? 0 : value > 255 ? 255 : value);
}

static int encode_yuyv(const WendyCamera *camera, const uint8_t *source, size_t bytes,
                       unsigned char **out, size_t *out_length) {
    if (camera->width & 1 || camera->stride < camera->width * 2 ||
        bytes < (size_t)camera->stride * camera->height) { errno = EINVAL; return -1; }
    struct jpeg_compress_struct cinfo = {0};
    struct jpeg_error_guard guard;
    cinfo.err = jpeg_std_error(&guard.pub);
    guard.pub.error_exit = jpeg_failed;
    if (setjmp(guard.jump)) { jpeg_destroy_compress(&cinfo); free(*out); *out = NULL; errno = EIO; return -1; }
    jpeg_create_compress(&cinfo);
    unsigned long length = 0;
    jpeg_mem_dest(&cinfo, out, &length);
    cinfo.image_width = camera->width;
    cinfo.image_height = camera->height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);
    JSAMPLE *row = malloc((size_t)camera->width * 3);
    if (!row) { jpeg_destroy_compress(&cinfo); free(*out); *out = NULL; return -1; }
    while (cinfo.next_scanline < cinfo.image_height) {
        const uint8_t *src = source + (size_t)cinfo.next_scanline * camera->stride;
        for (uint32_t x = 0; x < camera->width; x += 2) {
            int y0 = src[x * 2], u = src[x * 2 + 1] - 128;
            int y1 = src[x * 2 + 2], v = src[x * 2 + 3] - 128;
            row[x * 3] = clip(y0 + 359 * v / 256);
            row[x * 3 + 1] = clip(y0 - 88 * u / 256 - 183 * v / 256);
            row[x * 3 + 2] = clip(y0 + 454 * u / 256);
            row[x * 3 + 3] = clip(y1 + 359 * v / 256);
            row[x * 3 + 4] = clip(y1 - 88 * u / 256 - 183 * v / 256);
            row[x * 3 + 5] = clip(y1 + 454 * u / 256);
        }
        JSAMPROW scanline = row;
        jpeg_write_scanlines(&cinfo, &scanline, 1);
    }
    free(row);
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    *out_length = length;
    if (length > MAX_JPEG) { free(*out); *out = NULL; errno = EFBIG; return -1; }
    return 0;
}

int wendy_camera_grab_jpeg(WendyCamera *camera,
                           unsigned char **out, size_t *out_length,
                           char *error, size_t error_capacity) {
    *out = NULL; *out_length = 0;
    struct pollfd descriptor = {.fd = camera->fd, .events = POLLIN};
    struct v4l2_buffer frame = {0};
    frame.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    frame.memory = V4L2_MEMORY_MMAP;
    for (;;) {
        int ready = poll(&descriptor, 1, 5000);
        if (ready <= 0) { errno = ready == 0 ? ETIMEDOUT : errno; return fail(error, error_capacity, "waiting for webcam frame"); }
        if (ioctl(camera->fd, VIDIOC_DQBUF, &frame) == 0) break;
        if (errno != EAGAIN && errno != EINTR) return fail(error, error_capacity, "dequeueing webcam frame");
    }
    int result = -1;
    if (frame.index >= camera->count || frame.bytesused > camera->buffers[frame.index].length) {
        errno = EIO;
    } else if (camera->format == V4L2_PIX_FMT_MJPEG) {
        const uint8_t *data = camera->buffers[frame.index].memory;
        size_t length = frame.bytesused;
        if (length >= 4 && length <= MAX_JPEG && data[0] == 0xff && data[1] == 0xd8) {
            while (length >= 4 && !(data[length - 2] == 0xff && data[length - 1] == 0xd9)) length--;
            if (length >= 4) {
                *out = malloc(length);
                if (*out) { memcpy(*out, data, length); *out_length = length; result = 0; }
            }
        }
        if (result) errno = EIO;
    } else {
        result = encode_yuyv(camera, camera->buffers[frame.index].memory, frame.bytesused,
                             out, out_length);
    }
    int saved_errno = errno;
    if (ioctl(camera->fd, VIDIOC_QBUF, &frame) < 0) {
        free(*out); *out = NULL; *out_length = 0;
        return fail(error, error_capacity, "requeueing webcam frame");
    }
    errno = saved_errno;
    if (result) return fail(error, error_capacity, "encoding webcam JPEG");
    return 0;
}

int wendy_camera_thumbnail(const unsigned char *jpeg, size_t length,
                           unsigned char **out, size_t *out_length,
                           char *error, size_t error_capacity) {
    *out = NULL; *out_length = 0;
    if (length < 4 || length > MAX_JPEG) { errno = EINVAL; return fail(error, error_capacity, "thumbnail source"); }
    struct jpeg_decompress_struct input = {0};
    struct jpeg_error_guard in_guard;
    input.err = jpeg_std_error(&in_guard.pub);
    in_guard.pub.error_exit = jpeg_failed;
    // The pointer itself must survive longjmp after scanline/finish failures.
    // Automatic non-volatile locals changed after setjmp are indeterminate.
    uint8_t * volatile rgb = NULL;
    if (setjmp(in_guard.jump)) {
        jpeg_destroy_decompress(&input); free(rgb);
        errno = EIO; return fail(error, error_capacity, "decoding thumbnail source");
    }
    jpeg_create_decompress(&input);
    jpeg_mem_src(&input, jpeg, length);
    jpeg_read_header(&input, TRUE);
    input.scale_num = 1; input.scale_denom = 4;
    input.out_color_space = JCS_RGB;
    jpeg_start_decompress(&input);
    if (!input.output_width || !input.output_height || input.output_width > 4096 || input.output_height > 4096) {
        jpeg_destroy_decompress(&input); errno = EINVAL; return fail(error, error_capacity, "thumbnail dimensions");
    }
    size_t stride = (size_t)input.output_width * 3;
    rgb = malloc(stride * input.output_height);
    if (!rgb) { jpeg_destroy_decompress(&input); return fail(error, error_capacity, "thumbnail allocation"); }
    while (input.output_scanline < input.output_height) {
        JSAMPROW row = rgb + (size_t)input.output_scanline * stride;
        jpeg_read_scanlines(&input, &row, 1);
    }
    uint32_t source_width = input.output_width, source_height = input.output_height;
    jpeg_finish_decompress(&input);
    jpeg_destroy_decompress(&input);

    uint32_t target_width = 160, target_height = 120;
    struct jpeg_compress_struct output = {0};
    struct jpeg_error_guard out_guard;
    output.err = jpeg_std_error(&out_guard.pub);
    out_guard.pub.error_exit = jpeg_failed;
    if (setjmp(out_guard.jump)) {
        jpeg_destroy_compress(&output); free(rgb); free(*out); *out = NULL;
        errno = EIO; return fail(error, error_capacity, "encoding thumbnail");
    }
    jpeg_create_compress(&output);
    unsigned long compressed_length = 0;
    jpeg_mem_dest(&output, out, &compressed_length);
    output.image_width = target_width; output.image_height = target_height;
    output.input_components = 3; output.in_color_space = JCS_RGB;
    jpeg_set_defaults(&output);
    jpeg_start_compress(&output, TRUE);
    uint8_t row[target_width * 3];
    while (output.next_scanline < target_height) {
        uint32_t sy = (uint64_t)output.next_scanline * source_height / target_height;
        for (uint32_t x = 0; x < target_width; x++) {
            uint32_t sx = (uint64_t)x * source_width / target_width;
            memcpy(row + x * 3, rgb + (size_t)sy * stride + sx * 3, 3);
        }
        JSAMPROW line = row; jpeg_write_scanlines(&output, &line, 1);
    }
    jpeg_finish_compress(&output); jpeg_destroy_compress(&output); free(rgb);
    *out_length = compressed_length;
    return 0;
}
