#include "mjpc/learning/actor_critic.h"
#include "mjpc/learning/utilities.h"

#include <mujoco/mujoco.h>

// ----- Shared Actor Critic MLP ----- //
// evaluate policy
void SharedActorCritic::Policy(double* action, const double* observation) {
  // forward
  mlp.Forward(observation);

  // output
  // double* output = mlp.Output();
};

// evaluate value function
double SharedActorCritic::Value(const double* observation) {
  // forward
  mlp.Forward(observation);

  // output
  double* output = mlp.Output();

  // value
  return output[dim_actor];
};

// value loss
double SharedActorCritic::ValueLoss(const double* observation, double target) {
  double value = this->Value(observation);
  double diff = value - target;
  return 0.5 * diff * diff;
}

// value loss gradient
void SharedActorCritic::ValueLossGradient(double* gradient,
                                          const double* observation,
                                          double target) {
  // zero loss gradient
  Zero(mlp.loss_gradient.data(), mlp.loss_gradient.size());

  // set loss gradient for value element
  mlp.loss_gradient[dim_actor] = {this->Value(observation) - target};

  // backprop through MLP
  mlp.Backward(mlp.loss_gradient.data());

  // sum
  AddTo(gradient, mlp.gradient.data(), mlp.gradient.size());
}

// ----- Separate Actor and Critic MLP ----- //

// evaluate policy
void SeparateActorCritic::Policy(double* action, const double* observation) {
  // forward
  actor_mlp.Forward(observation);

  // output
  // double* output = actor_mlp.Output();
}

// evaluate value function
double SeparateActorCritic::Value(const double* observation) {
  // forward
  critic_mlp.Forward(observation);

  // output
  double* output = critic_mlp.Output();

  // value
  return output[0];
}

// value loss
double SeparateActorCritic::ValueLoss(const double* observation,
                                      double target) {
  double value = this->Value(observation);
  double diff = value - target;
  return 0.5 * diff * diff;
}

// value loss gradient
void SeparateActorCritic::ValueLossGradient(double* gradient,
                                            const double* observation,
                                            double target) {
  double loss_gradient[1] = {this->Value(observation) - target};
  critic_mlp.Backward(loss_gradient);
  AddTo(gradient, critic_mlp.gradient.data(), critic_mlp.gradient.size());
}
