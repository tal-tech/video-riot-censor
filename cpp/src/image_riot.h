#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "triton_infer.hpp"

using namespace facethink;

// 列出指定文件夹下的全部文件
std::vector<std::string> listFiles(const std::string& dir_name) {
  std::vector<std::string> filename_list;
  boost::filesystem::path root(dir_name);
  if (boost::filesystem::exists(root)) {
    std::string filename;
    boost::filesystem::directory_iterator iter_end;
    boost::filesystem::directory_iterator iter(root);
    for (; iter != iter_end; ++iter) {
      if (!boost::filesystem::is_directory(*iter)) {
        filename_list.push_back(iter->path().string());
      }
    }
  }
  return filename_list;
}


class Image_Riot
{
public:
  Image_Riot(std::string model_repository,std::string image_dir);
  void test();
private:
  std::string model_repository;
  std::string image_dir;
};