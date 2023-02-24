#ifndef MJPC_LEARNING_ACTOR_CRITIC_H_
#define MJPC_LEARNING_ACTOR_CRITIC_H_

#include "mjpc/learning/mlp.h"

class ActorCritic {
 public:
  // destructor
  virtual ~ActorCritic() = default;

  // evaluate policy
  virtual void Policy(double* action, const double* observation) = 0;

  // evaluate value function 
  virtual double Value(const double* observation) = 0;
};

class SharedActorCritic : public ActorCritic {
  public: 
    // constructor 
    SharedActorCritic() = default;

    // destructor
    ~SharedActorCritic() override = default;

    // evaluate policy
    void Policy(double* action, const double* observation) override;

    // evaluate value function 
    double Value(const double* observation) override;

    // value loss 
    double ValueLoss(const double* observation, double target);

    // value loss gradient 
    void ValueLossGradient(double* gradient, const double* observation, double target);

    // shared MLP network 
    mjpc::MLP mlp;

    // output dimension for actor
    int dim_actor;
};

class SeparateActorCritic : public ActorCritic {
  public: 
    // constructor 
    SeparateActorCritic() = default;

    // destructor
    ~SeparateActorCritic() override = default;

    // evaluate policy
    void Policy(double* action, const double* observation) override;

    // evaluate value function 
    double Value(const double* observation) override;

    // value loss 
    double ValueLoss(const double* observation, double target);

    // value loss gradient 
    void ValueLossGradient(double* gradient, const double* observation, double target);

    // actor MLP 
    mjpc::MLP actor_mlp;

    // critic MLP 
    mjpc::MLP critic_mlp; 
};

#endif  // MJPC_LEARNING_ACTOR_CRITIC_H_
