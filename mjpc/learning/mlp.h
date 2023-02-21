#ifndef MJPC_LEARNING_MLP_H_
#define MJPC_LEARNING_MLP_H_

#include <vector>

namespace mjpc {

// activation types
enum Activations : int {
  kPassThrough = 0,
  kReLU,
  kTanh,
};

// apply activation based on enum type
void Activation(double* output, const double* input, int dim, Activations type);

// multi-layer perceptron
class MLP {
 public:
  // constructor
  MLP() = default;

  // destructor
  ~MLP() = default;

  // initialize
  void Initialize(int dim_input, int dim_output, std::vector<int> dim_hidden,
                  std::vector<Activations> activations);

  // forward
  void Forward(const double* input);

  // backward
  void Backward(const double* loss_gradient);

  // output
  double* Output();

  // get weight 
  double* Weight(int layer);

  // get bias 
  double* Bias(int layer);

  // parameters
  std::vector<double> parameters;
  int num_parameters;

  // layer outputs
  std::vector<double> layer_output;
  std::vector<double> layer_output_activation_deriv;

  // activations
  std::vector<Activations> activations;
  std::vector<double> layer_activation;

  // delta (backpropagation)
  std::vector<double> delta;

  // loss gradient
  std::vector<double> gradient;

  // layer dimensions
  std::vector<int> dim_layer;

 private:
  // index shift for weights
  int WeightIndex(int layer);

  // index shift for biases
  int BiasIndex(int layer);

  // index shift for outputs
  int OutputIndex(int layer);

  // index shift for activations
  int ActivationIndex(int layer);
};

}  // namespace mjpc

#endif  // MJPC_LEARNING_MLP_H_
