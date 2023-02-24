#ifndef MJPC_LEARNING_ENV_H_
#define MJPC_LEARNING_ENV_H_

// virtual environment
class Environment {
 public:
  // destructor
  virtual ~Environment() = default;

  // step 
  virtual void Step(double* next_observation, const double* observation, const double* action) = 0;

  // reward 
  virtual double Reward(const double* observation, const double* action) = 0;

  // reset 
  virtual void Reset(double* observation) = 0;

  // observation dimension 
  virtual int ObservationDimension() = 0;

  // action dimension
  virtual int ActionDimension() = 0;
};

#endif  // MJPC_LEARNING_ENV_H_
