#include <vector>
#include <fstream>
#include <iostream>

#define TRAIN 1
#define TEST 2

int32_t swap32(int32_t val)
{
  return (val << 24) |
         ((val <<  8) & 0x00ff0000) |
         ((val >>  8) & 0x0000ff00) |
         ((val >> 24) & 0x000000ff);
}

int read_image(int idx, uint8_t *image_out, int flags)
{
  std::ifstream img_file, lbl_file;

  if (flags & TRAIN) {
    img_file.open("data/train-images.idx3-ubyte", std::ios::binary);
    lbl_file.open("data/train-labels.idx1-ubyte", std::ios::binary);
  } else if (flags & TEST) {
    img_file.open("data/t10k-images.idx3-ubyte", std::ios::binary);
    lbl_file.open("data/t10k-labels.idx1-ubyte", std::ios::binary);
  } else {
    std::cerr << "No valid dataset selected!" << std::endl;
  }

  uint32_t meta_img[2], meta_lbl[2];

  img_file.read((char *) meta_img, 8);
  lbl_file.read((char *) meta_lbl, 8);
  meta_img[0] = swap32(meta_img[0]);
  meta_img[1] = swap32(meta_img[1]);
  meta_lbl[0] = swap32(meta_lbl[0]);
  meta_lbl[1] = swap32(meta_lbl[1]);
  std::cout << std::hex << meta_img[0] << "  " << meta_img[1] << std::endl;
  std::cout << meta_lbl[0] << "  " << meta_lbl[1] << std::dec << std::endl;

  if (meta_img[0] != 2051) {
    std::cerr << "Image file invalid!" << std::endl;
    return -1;
  }

  if (meta_lbl[0] != 2049) {
    std::cerr << "Label file invalid!" << std::endl;
    return -1;
  }

  if (meta_lbl[1] != meta_img[1]) {
    std::cerr << "Files different sizes!" << std::endl;
    return -1;
  }

  if (idx >= meta_lbl[1]) {
    std::cerr << "Index out of bounds! IDX = " << idx
      << ", File size = " << meta_lbl[1] << std::endl;
    return -1;
  }

  img_file.seekg(16 + idx*784, std::ios::beg);
  lbl_file.seekg(8 + idx, std::ios::beg);

  img_file.read((char *) image_out, 784);
  char num;
  lbl_file.read(&num, 1);
  lbl_file.close();
  img_file.close();
  return num;
}

int main(int argc, char **argv)
{
  int idx = 0;
  if (argc > 1)
    idx = argv[1][0];
  uint8_t image[784];
  int lbl;
  lbl = read_image(idx, image, TEST);
  for (int i = 0; i < 28; ++i)
  {
    for (int j = 0; j < 28; ++j)
    {
      printf("%3d ", image[i*28 + j]);
    }
    std::cout << std::endl;
  }

  std::cout << "number is: " << lbl << std::endl;
}
