#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct WendyCamera WendyCamera;

int wendy_camera_open(const char *path, uint32_t width, uint32_t height,
                      WendyCamera **out, char *error, size_t error_capacity);
int wendy_camera_grab_jpeg(WendyCamera *camera,
                           unsigned char **out, size_t *out_length,
                           char *error, size_t error_capacity);
int wendy_camera_thumbnail(const unsigned char *jpeg, size_t length,
                           unsigned char **out, size_t *out_length,
                           char *error, size_t error_capacity);
void wendy_camera_close(WendyCamera *camera);
void wendy_camera_free_bytes(unsigned char *bytes);
void wendy_install_stop_handlers(void);
int wendy_stop_requested(void);
