#include "image_riot.h"


Image_Riot::Image_Riot(std::string model_repository,std::string image_dir)
{
    this->model_repository = model_repository;
    this->image_dir = image_dir;
}

void Image_Riot::test()
{
  int run_loop = 1;

  // 创建log文件
  const std::string log_tag = "demo";
  typedef boost::log::sinks::synchronous_sink<boost::log::sinks::text_file_backend> sink_t;
  boost::shared_ptr<sink_t> file_sink = boost::log::add_file_log(
      boost::log::keywords::auto_flush = true,
      boost::log::keywords::file_name = log_tag + ".log",
      boost::log::keywords::format = "[%TimeStamp%]:%Message%");
  file_sink->set_filter(
      boost::log::trivial::severity >= 2 && boost::log::expressions::attr<std::string>("Channel") == log_tag);
      boost::log::sources::severity_channel_logger<boost::log::trivial::severity_level, std::string> logger =
      boost::log::sources::severity_channel_logger<boost::log::trivial::severity_level, std::string>(boost::log::keywords::channel = log_tag);

  // 遍历文件夹,获取图片列表
  std::vector<std::string> image_list = listFiles(image_dir);
  if (image_list.empty()) {
    std::cout << "Error: failed to find images in dir " << image_dir << std::endl;
    BOOST_LOG_SEV(logger, boost::log::trivial::error) << "Error: failed to find images in dir " << image_dir;
    return ;
  }

  // 设置模型名称, 模型输入名, 模型输出名
  std::string model_name = "cls_image_riot_resnet18_pipeline";
  std::vector<std::string> input_names = {"input"};
  std::vector<std::string> output_names = {"result", "prob"};
  TritonInfer* detector = TritonInfer::create(model_repository, model_name, input_names, output_names);
  if (detector == nullptr) {
    std::cerr << "Error: failed to create inference model" << std::endl;
    BOOST_LOG_SEV(logger, boost::log::trivial::error) << "Error: failed to create inference model";
    return ;
  }

  double total_cost_time = 0;
  int total_count = 0;
  for (int run_index = 0; run_index < run_loop || run_loop < 0; ++run_index) {
    for (auto& image_path : image_list) {
      // 读取图片
      cv::Mat image = cv::imread(image_path);
      if (image.empty()) {
        std::cerr << "Error: empty image " << image_path << std::endl;
        BOOST_LOG_SEV(logger, boost::log::trivial::error) << "Error: empty image " << image_path;
        continue;
      }
      std::vector<uchar> encoded_image;
      cv::imencode(".png", image, encoded_image);
      std::vector<std::vector<int64_t>> input_shapes;
      std::vector<int64_t> input_shape = {(int)encoded_image.size()};
      input_shapes.push_back(input_shape);
      std::vector<std::pair<void*, size_t>> input_data;
      input_data.push_back(std::make_pair(&encoded_image[0], encoded_image.size() * sizeof(uchar)));

      // 执行推理
      std::vector<std::pair<void*, size_t>> output_data;
      auto start_time = std::chrono::steady_clock::now();
      int ret = detector->inference(input_shapes, input_data, output_data);
      auto end_time = std::chrono::steady_clock::now();
      double cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      ++total_count;
      if (total_count > 1) {
        total_cost_time += cost_time;
      }
      // 获取模型输出
      std::vector<int> output_result(reinterpret_cast<const int*>(output_data[0].first), reinterpret_cast<const int*>(output_data[0].first) + output_data[0].second / sizeof(int));
      std::vector<float> output_prob(reinterpret_cast<const float*>(output_data[1].first), reinterpret_cast<const float*>(output_data[1].first) + output_data[1].second / sizeof(float));
      std::cout << image_path << "," << output_result[0] << "," << output_prob[0] << "," << output_prob[1] << "," << output_prob[2] << std::endl;
      BOOST_LOG_SEV(logger, boost::log::trivial::info) << image_path << "," << output_result[0] << "," << output_prob[0] << "," << output_prob[1] << "," << output_prob[2];
    }
  }
  if (total_count > 1) {
    double avg_cost_time = total_cost_time / (total_count - 1);
    std::cout << "avg_cost_time: " << avg_cost_time << " ms, total_count: " << total_count << std::endl;
    BOOST_LOG_SEV(logger, boost::log::trivial::info) << "avg_cost_time: " << avg_cost_time << " ms, total_count: " << total_count;
  }

  // 释放资源
  boost::log::core::get()->remove_sink(file_sink);
  file_sink->flush();
  file_sink.reset();
  delete detector;
}
