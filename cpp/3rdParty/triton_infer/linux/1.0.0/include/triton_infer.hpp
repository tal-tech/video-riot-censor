#ifndef __FACETHINK_API_TRITON_INFER_HPP__
#define __FACETHINK_API_TRITON_INFER_HPP__

#ifdef WIN32
#ifdef DLL_EXPORTS
#define EXPORT_CLASS __declspec(dllexport)
#define EXPORT_API extern "C" __declspec(dllexport)
#define EXPORT_CLASS_API
#else
#define EXPORT_CLASS __declspec(dllimport)
#define EXPORT_API extern "C" __declspec(dllimport)
#endif
#else
#define EXPORT_CLASS
#define EXPORT_API extern "C" __attribute__((visibility("default")))
#define EXPORT_CLASS_API __attribute__((visibility("default")))
#endif

#include <string>
#include <vector>

namespace facethink {

class EXPORT_CLASS TritonInfer {
 public:
  EXPORT_CLASS_API
  static TritonInfer* create(const std::string& model_repository, const std::string& model_name,
                             const std::vector<std::string>& input_names, const std::vector<std::string>& output_names, int log_level = 4);

  EXPORT_CLASS_API
  virtual int inference(const std::vector<std::vector<int64_t>>& input_shapes, const std::vector<std::pair<void*, size_t>>& input_data,
                        std::vector<std::pair<void*, size_t>>& output_data, int batch_size = 1, int timeout = 0) = 0;

  EXPORT_CLASS_API
  virtual ~TritonInfer(void);

 protected:
  TritonInfer();
  TritonInfer(const TritonInfer&) = delete;
  TritonInfer& operator=(const TritonInfer&) = delete;
};

}  // namespace facethink

#endif  //__FACETHINK_API_TRITON_INFER_HPP__