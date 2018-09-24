#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "mnist.h"


static int32_t swap32(int32_t val)
{
  return (val << 24) |
         ((val <<  8) & 0x00ff0000) |
         ((val >>  8) & 0x0000ff00) |
         ((val >> 24) & 0x000000ff);
}


int read_images(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& image_out,
  Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& label_out, int flags)
{
  std::ifstream img_file, lbl_file;

  if (flags & TRAIN) {
    img_file.open("mnist/data/train-images.idx3-ubyte", std::ios::binary);
    lbl_file.open("mnist/data/train-labels.idx1-ubyte", std::ios::binary);
  } else if (flags & TEST) {
    img_file.open("mnist/data/t10k-images.idx3-ubyte", std::ios::binary);
    lbl_file.open("mnist/data/t10k-labels.idx1-ubyte", std::ios::binary);
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

  // Check file validity
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

  // // Resize outputs if necessary
  if (meta_lbl[1] != label_out.rows())
    label_out.resize(meta_lbl[1], 1);

  if (meta_img[1] != image_out.rows())
    image_out.resize(784, meta_img[1]);

  // Set stream location
  img_file.seekg(16);
  lbl_file.seekg(8);

  for (uint16_t i=0; i<meta_img[1]; ++i)
  {
    label_out(i) = lbl_file.get();
    for (int j=0; j<IMG_LEN; ++j)
    {
      image_out(j, i) = (uint8_t)img_file.get();
    }
  }
  return 0;
}
