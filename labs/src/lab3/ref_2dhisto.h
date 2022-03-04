#define HISTO_WIDTH  1024
#define HISTO_HEIGHT 1
#define HISTO_LOG 10

int ref_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);
