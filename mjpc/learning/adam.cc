#include "mjpc/learning/adam.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <random>

#include "mjpc/utilities.h"

namespace mjpc {

// allocate
void AdamOptimizer::Initialize(int dim) {
  // resize memory
  g.resize(dim);
  m.resize(dim);
  v.resize(dim);
  m_hat.resize(dim);
  v_hat.resize(dim);

  // set batch
  batch.resize(dim_batch);
  for (int i = 0; i < dim_batch; i++) {
    batch[i] = i;
  }

  // reset
  mju_zero(g.data(), dim);
  mju_zero(m.data(), dim);
  mju_zero(v.data(), dim);
  mju_zero(m_hat.data(), dim);
  mju_zero(v_hat.data(), dim);
  t = 0;
}

// step optimizer
void AdamOptimizer::Step(std::vector<double>& x) {
  // increment iteration
  t += 1;

  // update biased first moment estimate
  // m = β1 * opt.m + (1.0 - β1) * g
  mju_scl(m.data(), m.data(), beta1, m.size());
  mju_addToScl(m.data(), g.data(), 1.0 - beta1, m.size());

  // update biased second raw moment estimate
  // v = β2 * v + (1.0 - β2) * g.^2.0
  mju_scl(v.data(), v.data(), beta2, v.size());
  for (int i = 0; i < v.size(); i++) {
    v[i] += (1.0 - beta2) * g[i] * g[i];
  }

  // compute bias-corrected first moment estimate
  // m̂ = m ./ (1.0 - β1^t)
  mju_scl(m_hat.data(), m.data(), 1.0 / (1.0 - mju_pow(beta1, t)),
          m_hat.size());

  // compute bias-corrected second raw moment estimate
  // v̂ .= v ./ (1.0 - β2^t)
  mju_scl(v_hat.data(), v.data(), 1.0 / (1.0 - mju_pow(beta2, t)),
          v_hat.size());

  // update parameters
  // θ .= θ - α .* m̂ ./ (sqrt.(v̂) .+ ϵ)
  for (int i = 0; i < x.size(); i++) {
    x[i] = x[i] - alpha * m_hat[i] / (mju_sqrt(v_hat[i]) + eps);
  }
  // mju_addToScl(x.data(), g.data(), -1.0 * alpha, g.size());
}

// optimize
void AdamOptimizer::Optimize(std::vector<double>& parameters,
                             GradientFunction gradient) {
  // reset
  mju_zero(g.data(), g.size());
  mju_zero(m.data(), m.size());
  mju_zero(v.data(), v.size());
  mju_zero(m_hat.data(), m_hat.size());
  mju_zero(v_hat.data(), v_hat.size());
  t = 0;

  // shuffle minibatches
  std::shuffle(batch.begin(), batch.end(), std::default_random_engine(seed));

  // loop over minibatches
  int num_minibatch = std::floor(dim_batch / dim_minibatch);

  // epochs
  for (int i = 0; i < epochs; i++) {
    for (int j = 0; j < num_minibatch; j++) {
      // minibatch
      const int* minibatch = batch.data() + dim_minibatch * j;

      // gradient
      mju_zero(g.data(), g.size());
      gradient(g.data(), parameters.data(), minibatch, dim_minibatch);

      // TODO(taylor): normalize / clip gradient

      // update
      this->Step(parameters);
    }
  }
}

// // -----train MLP ----- //
// int N = 16;
// std::vector<double> X(N);
// std::vector<double> Y(N);

// double X0 = -1.0;
// double XN = 1.0;

// // dataset
// for (int i = 0; i < N; i++) {
//   X[i] = (XN - X0) / (N - 1) * i + X0;
//   Y[i] = 2.0 * X[i] + 0.5;
//   printf("X[i] = %f\n", X[i]);
//   printf("Y[i] = %f\n", Y[i]);
// }

// // MLP
// mjpc::MLP mlp;
// int dim_input = 1;
// int dim_output = 1;
// std::vector<int> dim_hidden = {2, 2};
// std::vector<mjpc::Activations> activations = {mjpc::kTanh, mjpc::kTanh,
// mjpc::kPassThrough}; mlp.Initialize(dim_input, dim_output, dim_hidden,
// activations); absl::BitGen gen_; for (int i = 0; i < mlp.num_parameters; i++)
// {
//   mlp.parameters[i] = 1.0e-1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
//   printf("parameter[i] = %f\n", mlp.parameters[i]);
// }

// // loss
// auto loss = [](const double* y, const double* y0) {
//   double diff = y[0] - y0[0];
//   return 0.5 * diff * diff;
// };

// // loss gradient
// auto loss_y = [](double* gradient, const double* y, const double* y0) {
//   gradient[0] = y[0] - y0[0];
// };

// mlp.Forward(X.data());
// double* out = mlp.Output();
// printf("out[0]: %f\n", out[0]);

// printf("loss[0] = %f\n", loss(out, Y.data()));
// double ly[1] = {0.0};
// loss_y(ly, out, Y.data());
// printf("loss_y[0] = %f\n", ly[0]);

// // total loss (initial)
// double l0 = 0.0;
// for (int i = 0; i < N; i++) {
//   double* x = X.data() + i;
//   double* y = Y.data() + i;
//   mlp.Forward(x);
//   double* out = mlp.Output();
//   l0 += loss(out, y);
// }
// l0 /= N;
// printf("total_loss = %f\n", l0);

// printf("training set comparison:\n");
// for (int i = 0; i < N; i++) {
//   printf("Y0[%i] = %f\n", i, Y[i]);
//   mlp.Forward(X.data() + i);
//   out = mlp.Output();
//   printf("Y[%i] = %f\n", i, out[0]);
//   printf("\n");
// }

// // optimizer
// mjpc::AdamOptimizer opt;
// opt.dim_batch = N;
// opt.dim_minibatch = 4;
// opt.epochs = 10000;
// opt.alpha = 0.001;

// opt.Initialize(mlp.num_parameters);

// auto gradient = [&mlp, &loss_y, &X, &Y](double* gradient, const double*
// parameters, const int* minibatch, int dim_minibatch) {
//   for (int i = 0; i < dim_minibatch; i++) {
//     double* x = X.data() + minibatch[i];
//     double* y = Y.data() + minibatch[i];
//     mlp.Forward(x);
//     double* out = mlp.Output();
//     double ly[1] = {0.0};
//     loss_y(ly, out, y);
//     mlp.Backward(ly);
//     mju_addToScl(gradient, mlp.gradient.data(), 1.0 / dim_minibatch,
//     mlp.gradient.size());
//   }
// };

// // const int minibatch[4] = {0, 1, 2, 3};
// // gradient(opt.g.data(), mlp.parameters.data(), minibatch, 4);

// // printf("parameters gradient: \n");
// // mju_printMat(opt.g.data(), 1, opt.g.size());

// // mju_zero(mlp.gradient.data(), mlp.num_parameters);
// // gradient(opt.g.data(), mlp.parameters.data(), minibatch,
// opt.dim_minibatch);

// // printf("parameters gradient: \n");
// // mju_printMat(opt.g.data(), 1, opt.g.size());

// // printf("parameters:\n");
// // mju_printMat(mlp.parameters.data(), 1, mlp.num_parameters);

// // opt.Step(mlp.parameters);

// // printf("updated parameters:\n");
// // mju_printMat(mlp.parameters.data(), 1, mlp.num_parameters);

// opt.Optimize(mlp.parameters, gradient);

// // total loss (train)
// double lT = 0.0;
// for (int i = 0; i < N; i++) {
//   double* x = X.data() + i;
//   double* y = Y.data() + i;
//   mlp.Forward(x);
//   double* out = mlp.Output();
//   lT += loss(out, y);
// }
// lT /= N;
// printf("total_loss = %f\n", lT);

// printf("training set comparison:\n");
// for (int i = 0; i < N; i++) {
//   printf("Y0[%i] = %f\n", i, Y[i]);
//   mlp.Forward(X.data() + i);
//   out = mlp.Output();
//   printf("Y[%i] = %f\n", i, out[0]);
//   printf("\n");
// }

}  // namespace mjpc