#include "mjpc/learning/mlp.h"

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

namespace mjpc {

// apply activation based on enum type
void Activation(double* output, const double* input, int dim,
                Activations type) {
  for (int i = 0; i < dim; i++) {
    // pass through
    if (type == kPassThrough) {
      output[i] = input[i];
      // relu
    } else if (type == kReLU) {
      output[i] = mju_max(0.0, input[i]);
      // tanh
    } else if (type == kTanh) {
      output[i] = mju_tanh(input[i]);
    }
  }
}

// apply activation based on enum type
void ActivationDerivative(double* output, const double* input, int dim,
                          Activations type) {
  for (int i = 0; i < dim; i++) {
    // pass through
    if (type == kPassThrough) {
      output[i] = 1.0;
      // relu
    } else if (type == kReLU) {
      if (input[i] >= 0.0) {
        output[i] = 1.0;
      } else {
        output[i] = 0.0;
      }
      // tanh
    } else if (type == kTanh) {
      output[i] = 1.0 - mju_tanh(input[i]) * mju_tanh(input[i]);
    }
  }
}

// initialize
void MLP::Initialize(int dim_input, int dim_output, std::vector<int> dim_hidden,
                     std::vector<Activations> activations) {
  if (!(dim_hidden.size() == activations.size() - 1)) {
    mju_error(
        "Hidden layer dimensions and number of activations do not match\n");
  }

  // ----- set layer dimensions ----- //
  // reset
  dim_layer.resize(0);

  // input layer
  dim_layer.push_back(dim_input);

  // hidden layers
  for (const auto& dh : dim_hidden) {
    dim_layer.push_back(dh);
  }

  // output layer
  dim_layer.push_back(dim_output);

  // set activations
  this->activations = activations;

  // ----- number of parameters ----- //
  num_parameters = 0;
  for (int i = 1; i < (int)dim_layer.size(); i++) {
    num_parameters += dim_layer[i] * dim_layer[i - 1] + dim_layer[i];
  }

  // resize parameters
  parameters.resize(num_parameters);

  // add Gaussian noise
  absl::BitGen gen_;
  for (int i = 0; i < num_parameters; i++) {
    parameters[i] = 1.0e-1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // ----- layer outputs ----- //
  int dim_layers = 0;
  for (int i = 0; i < (int)dim_layer.size(); i++) {
    dim_layers += dim_layer[i];
  }

  // resize
  layer_output.resize(dim_layers - dim_layer[0]);
  layer_activation.resize(dim_layers);
  layer_output_activation_deriv.resize(dim_layers - dim_layer[0]);
  delta.resize(dim_layers - dim_layer[0]);
  gradient.resize(num_parameters);
}

// forward
void MLP::Forward(const double* input) {
  // initial activation is input
  mju_copy(layer_activation.data(), input, dim_layer[0]);

  // ----- layers ----- //
  for (int i = 1; i < dim_layer.size(); i++) {
    // layer output
    mju_mulMatVec(layer_output.data() + this->OutputIndex(i - 1),
                  parameters.data() + this->WeightIndex(i - 1),
                  layer_activation.data() + this->ActivationIndex(i - 1),
                  dim_layer[i], dim_layer[i - 1]);
    mju_addTo(layer_output.data() + this->OutputIndex(i - 1),
              parameters.data() + this->BiasIndex(i - 1), dim_layer[i]);

    // activation
    Activation(layer_activation.data() + this->ActivationIndex(i),
               layer_output.data() + this->OutputIndex(i - 1), dim_layer[i],
               activations[i - 1]);
  }
}

// backward
void MLP::Backward(const double* loss_gradient) {
  for (int i = dim_layer.size() - 1; i >= 1; i--) {
    // compute activation derivative
    int output_shift = this->OutputIndex(i - 1);
    double* deltal = delta.data() + output_shift;
    double* zl = layer_output.data() + output_shift;
    double* al = layer_activation.data() + this->ActivationIndex(i - 1);
    double* adl = layer_output_activation_deriv.data() + output_shift;
    ActivationDerivative(adl, zl, dim_layer[i], activations[i - 1]);

    // compute deltaL = [loss_gradient | (W^(l + 1))^T delta^(l + 1)] .*
    // sigma'(zl)
    if (i == dim_layer.size() - 1) {
      mju_copy(deltal, loss_gradient, dim_layer[i]);
    } else {
      double* deltap = delta.data() + this->OutputIndex(i);
      double* Wp = parameters.data() + this->WeightIndex(i);
      mju_mulMatTVec(deltal, Wp, deltap, dim_layer[i + 1], dim_layer[i]);
    }

    for (int j = 0; j < dim_layer[i]; j++) {
      deltal[j] *= adl[j];
    }

    // set d loss / d b
    mju_copy(gradient.data() + this->BiasIndex(i - 1), deltal, dim_layer[i]);

    // set d loss / d W
    for (int j = 0; j < dim_layer[i]; j++) {
      for (int k = 0; k < dim_layer[i - 1]; k++) {
        gradient[this->WeightIndex(i - 1) + j * dim_layer[i - 1] + k] =
            al[k] * deltal[j];
      }
    }
  }
}

// output
double* MLP::Output() {
  return layer_output.data() + this->OutputIndex(dim_layer.size() - 2);
}

// get weight 
double* MLP::Weight(int layer) {
  return parameters.data() + this->WeightIndex(layer);
}

// get bias 
double* MLP::Bias(int layer) {
  return parameters.data() + this->BiasIndex(layer);
}

// index shift for weights
int MLP::WeightIndex(int layer) {
  int index = 0;
  for (int i = 1; i < layer + 1; i++) {
    index += dim_layer[i] * dim_layer[i - 1];
    index += dim_layer[i];
  }
  return index;
}

// index shift for biases
int MLP::BiasIndex(int layer) {
  int index = 0;
  for (int i = 1; i < layer + 1; i++) {
    index += dim_layer[i] * dim_layer[i - 1];
    index += dim_layer[i];
  }
  index += dim_layer[layer + 1] * dim_layer[layer];
  return index;
}

// index shift for outputs
int MLP::OutputIndex(int layer) {
  int index = 0;
  for (int i = 1; i < layer + 1; i++) {
    index += dim_layer[i];
  }
  return index;
}

// index shift for activations
int MLP::ActivationIndex(int layer) {
  int index = 0;
  for (int i = 0; i < layer; i++) {
    index += dim_layer[i];
  }
  return index;
}

}  // namespace mjpc