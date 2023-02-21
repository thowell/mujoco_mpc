#ifndef MJPC_LEARNING_ADAM_H_
#define MJPC_LEARNING_ADAM_H_

#include <vector>

namespace mjpc {

// gradient function interface
using GradientFunction =
    std::function<void(double* gradient, const double* parameters,
                       const int* minibatch, int dim_minibatch)>;

// Adam optimizer
class AdamOptimizer {
 public:
  // constructor
  AdamOptimizer() {
    // initialize default parameters
    beta1 = 0.9;
    beta2 = 0.999;
    alpha = 0.0003;
    eps = 1.0e-8;
    t = 0;

    // initialize default settings
    epochs = 1;
    dim_batch = 1;
    dim_minibatch = 1;

    // random seed
    seed = 0;
  };

  // destructor
  ~AdamOptimizer() = default;

  // initialize
  void Initialize(int dim);

  // step optimizer
  void Step(std::vector<double>& parameters);

  // optimize
  void Optimize(std::vector<double>& parameters, GradientFunction gradient);

  // memory
  std::vector<double> g;      // gradient
  std::vector<double> m;      // first moment
  std::vector<double> v;      // second momemnt
  std::vector<double> m_hat;  // corrected first moment
  std::vector<double> v_hat;  // corrected second moment
  std::vector<int> batch;     // batch indices

  // parameters
  double beta1;
  double beta2;
  double alpha;
  double eps;
  int t;

  // settings
  int epochs;
  int dim_batch;
  int dim_minibatch;
  double seed;
};

}  // namespace mjpc

#endif  // MJPC_LEARNING_ADAM_H_
