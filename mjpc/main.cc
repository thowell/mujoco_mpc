// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>
#include <absl/flags/parse.h>
#include <absl/random/random.h>

#include "mjpc/app.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

#include "mjpc/planners/linear_solve.h"

// machinery for replacing command line error by a macOS dialog box
// when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default")))
extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char** argv) {
  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  absl::ParseCommandLine(argc, argv);

  mjpc::StartApp(mjpc::GetTasks(), 3);  // start with humanoid stand
  return 0;
}

// // ----- problem setup ----- // 
// const double h = 0.1;
// const std::vector<double> A{1.0, h, 0.0, 1.0};          // dynamics state Jacobian
// const std::vector<double> B{1.0, 0.0, 0.0, 1.0};        // dynamics action Jacobian
// const std::vector<double> C{1.0, 0.0, 0.0, 1.0};        // sensor state Jacobian
// const std::vector<double> D{0.0, 0.0, 0.0, 0.0};        // sensor action Jacobian
// const std::vector<double> E{-1.0, -1.0 * h, 0.0, -1.0}; // inverse dynamics current state Jacobian
// const std::vector<double> F{1.0, 0.0, 0.0, 1.0};        // inverse dynamics next state Jacobian
// const std::vector<double> In{1.0, 0.0, 0.0, 1.0};       // identity (state dimension)
// const std::vector<double> Inq{1.0};                     // identity (configuration dimension)
// const std::vector<double> Inv{1.0};                     // identity (velocity dimension)
// const std::vector<double> Ina;                          // identity (acceleration dimension)
// const int n  = 2;                                       // state dimension
// const int nq = 1;                                       // configuration dimension
// const int nv = 1;                                       // velocity dimension 
// const int na = 0;                                       // acceleration dimension
// const int m  = 2;                                       // action dimension
// const int p  = 2;                                       // sensor dimension 

// // dynamics: z = A * x + B * u
// void f(double* z, const double* x, const double* u) {
//   double tmp[n];
//   mju_mulMatVec(z, A.data(), x, n, n);
//   mju_mulMatVec(tmp, B.data(), u, n, m);
//   mju_addTo(z, tmp, n);
// }

// // sensor: y = C * x + D * u
// void g(double* y, const double* x, const double* u) {
//   double tmp[p];
//   mju_mulMatVec(y, C.data(), x, p, n);
//   mju_mulMatVec(tmp, D.data(), u, p, m);
//   mju_addTo(y, tmp, p);
// }

// // inverse dynamics: u = E * x + F z = -B \ A * x + B \ z
// void d(double* u, const double* x, const double* z) {
//   double tmp[m];
//   mju_mulMatVec(u, E.data(), x, m, n);
//   mju_mulMatVec(tmp, F.data(), z, m, n);
//   mju_addTo(u, tmp, m);
// }

// // ----- objective ----- //

// // residual 1
// void r1(double* R, const double* Z, const double* U, int T) {
//   for (int t = 0; t < T - 1; t++) {
//     // rt = f(xt, ut) - xt+1
//     double* r = R + n * t;
//     const double* x = Z + n * t;
//     const double* u = U + m * t;
//     const double* z = Z + n * (t + 1);
//     f(r, x, u);
//     mju_subFrom(r, z, n);
//   }
// }

// // residual Jacobian 1
// void r1z(double* Rz, const double* Z, const double* U, int T) {
//   mju_zero(Rz, (n * (T - 1)) * (n * T));
//   for (int t = 0; t < T - 1; t++) {
//     mjpc::SetMatrixInMatrix(Rz, A.data(), 1.0, n * (T - 1), n * T, n, n, n * t, n * t);
//     mjpc::SetMatrixInMatrix(Rz, In.data(), -1.0, n * (T - 1), n * T, n, n, n * t, n * (t + 1));
//   }
// }

// // cost 1 
// double J1(const double* Z, const double* U, const double* P, int T) {
//   double r[n * (T - 1)];
//   double tmp[n * (T - 1)];
//   r1(r, Z, U, T);
//   mju_mulMatVec(tmp, P, r, n * (T - 1), n * (T - 1));
//   return 0.5 * mju_dot(r, tmp, n * (T - 1));
// }

// // cost gradient 1
// void J1z(double* Jz, const double* Z, const double* U, const double* P, int T) {
//   double r[n * (T - 1)];
//   double rz[(n * (T - 1)) * (n * T)];
//   double tmp[n * (T - 1)];
//   r1(r, Z, U, T);
//   r1z(rz, Z, U, T);
//   mju_mulMatVec(tmp, P, r, n * (T - 1), n * (T - 1));
//   mju_mulMatTVec(Jz, rz, tmp, n * (T - 1), n * T); 
// }

// // cost Hessian 1
// void J1zz(double* Jzz, const double* Z, const double* U, const double* P, int T) {
//   double rz[(n * (T - 1)) * (n * T)];
//   double tmp[(n * (T - 1)) * (n * T)];
//   r1z(rz, Z, U, T);
//   mju_mulMatMat(tmp, P, rz, n * (T - 1), n * (T - 1), n * T);
//   mju_mulMatTMat(Jzz, rz, tmp, n * (T - 1), n * T, n * T); 
// }

// // residual 2
// void r2(double* R, const double* Z, const double* U, const double* Y, int T) {
//   for (int t = 0; t < T - 1; t++) {
//     // rt = g(xt, ut) - yt
//     double* r = R + p * t;
//     const double* x = Z + n * t;
//     const double* u = U + m * t;
//     const double* y = Y + p * t;
//     g(r, x, u);
//     mju_subFrom(r, y, p);
//   }
// }

// // residual Jacobian 2
// void r2z(double* Rz, const double* Z, const double* U, const double* Y, int T) {
//   mju_zero(Rz, (p * (T - 1)) * (n * T));
//   for (int t = 0; t < T - 1; t++) {
//     mjpc::SetMatrixInMatrix(Rz, C.data(), 1.0, p * (T - 1), n * T, p, n, p * t, n * t);
//   }
// }

// // cost 2
// double J2(const double* Z, const double* U, const double* Y, const double* S, int T) {
//   double r[p * (T - 1)];
//   double tmp[p * (T - 1)];
//   r2(r, Z, U, Y, T);
//   mju_mulMatVec(tmp, S, r, p * (T - 1), p * (T - 1));
//   return 0.5 * mju_dot(r, tmp, p * (T - 1));
// }

// // cost gradient 2
// void J2z(double* Jz, const double* Z, const double* U, const double* Y, const double* S, int T) {
//   double r[p * (T - 1)];
//   double rz[(p * (T - 1)) * (n * T)];
//   double tmp[p * (T - 1)];
//   r2(r, Z, U, Y, T);
//   r2z(rz, Z, U, Y, T);
//   mju_mulMatVec(tmp, S, r, p * (T - 1), p * (T - 1));
//   mju_mulMatTVec(Jz, rz, tmp, p * (T - 1), n * T); 
// }

// // cost Hessian 2
// void J2zz(double* Jzz, const double* Z, const double* U, const double* Y, const double* S, int T) {
//   double rz[(p * (T - 1)) * (n * T)];
//   double tmp[(p * (T - 1)) * (n * T)];
//   r2z(rz, Z, U, Y, T);
//   mju_mulMatMat(tmp, S, rz, p * (T - 1), p * (T - 1), n * T);
//   mju_mulMatTMat(Jzz, rz, tmp, p * (T - 1), n * T, n * T); 
// }

// // residual 3
// void r3(double* R, const double* Z, const double* U, int T) {
//   for (int t = 0; t < T - 1; t++) {
//     // rt = d(xt, xt+1) - B * ut
//     double* r = R + m * t;
//     const double* x = Z + n * t;
//     const double* u = U + m * t;
//     const double* z = Z + n * (t + 1);
//     d(r, x, z);
//     double tmp[m];
//     mju_mulMatVec(tmp, B.data(), u, n, m);
//     mju_subFrom(r, tmp, m);
//   }
// }

// // residual Jacobian 3
// void r3z(double* Rz, const double* Z, const double* U, int T) {
//   mju_zero(Rz, (m * (T - 1)) * (n * T));
//   for (int t = 0; t < T - 1; t++) {
//     mjpc::SetMatrixInMatrix(Rz, E.data(), 1.0, m * (T - 1), n * T, m, n, m * t, n * t);
//     mjpc::SetMatrixInMatrix(Rz, F.data(), 1.0, m * (T - 1), n * T, m, n, m * t, n * (t + 1));
//   }
// }

// // cost 3
// double J3(const double* Z, const double* U, const double* R, int T) {
//   double r[m * (T - 1)];
//   double tmp[m * (T - 1)];
//   r3(r, Z, U, T);
//   mju_mulMatVec(tmp, R, r, m * (T - 1), m * (T - 1));
//   return 0.5 * mju_dot(r, tmp, m * (T - 1));
// }

// // cost gradient 3
// void J3z(double* Jz, const double* Z, const double* U, const double* R, int T) {
//   double r[m * (T - 1)];
//   double rz[(m * (T - 1)) * (n * T)];
//   double tmp[m * (T - 1)];
//   r3(r, Z, U, T);
//   r3z(rz, Z, U, T);
//   mju_mulMatVec(tmp, R, r, m * (T - 1), m * (T - 1));
//   mju_mulMatTVec(Jz, rz, tmp, m * (T - 1), n * T); 
// }

// // cost Hessian 3
// void J3zz(double* Jzz, const double* Z, const double* U, const double* R, int T) {
//   double rz[(m * (T - 1)) * (n * T)];
//   double tmp[(m * (T - 1)) * (n * T)];
//   r3z(rz, Z, U, T);
//   mju_mulMatMat(tmp, R, rz, m * (T - 1), m * (T - 1), n * T);
//   mju_mulMatTMat(Jzz, rz, tmp, m * (T - 1), n * T, n * T); 
// }

// // cumulative cost 
// double J(const double* Z, const double* U, const double* Y, const double* P, const double* S, const double* R, int T) {
//   return J1(Z, U, P, T) + J2(Z, U, Y, S, T) + J3(Z, U, R, T);
// }

// // cumulative cost gradient 
// void Jz(double* Jz_, const double* Z, const double* U, const double* Y, const double* P, const double* S, const double* R, int T) {
//   double tmp[n * T];
//   J1z(Jz_, Z, U, P, T);
//   J2z(tmp, Z, U, Y, S, T);
//   mju_addTo(Jz_, tmp, n * T);
//   J3z(tmp, Z, U, R, T);
//   mju_addTo(Jz_, tmp, n * T);
// }

// // cumulative cost Hessian
// void Jzz(double* Jzz_, const double* Z, const double* U, const double* Y, const double* P, const double* S, const double* R, int T) {
//   double tmp[(n * T) * (n * T)];
//   J1zz(Jzz_, Z, U, P, T);
//   J2zz(tmp, Z, U, Y, S, T);
//   mju_addTo(Jzz_, tmp, (n * T) * (n * T));
//   J3zz(tmp, Z, U, R, T);
//   mju_addTo(Jzz_, tmp, (n * T) * (n * T));
// }

// // mapping from configurations to state (via finite difference)
// // TODO(taylor): acceleration
// void ConfigurationToState(double* Z, const double* Q, int T) {
//   // time step for finite difference
//   double h_ = h * (T - 1) / (2 * T - 1);

//   for (int t = 0; t < T; t++) {
//     const double* q0 = Q + 2 * nq * t;
//     const double* q1 = Q + 2 * nq * t + nq;
//     const double* q2 = Q + 2 * nq * t + 2 * nq;

//     // configuration 
//     mju_copy(Z + n * t, q1, nq);

//     // velocity 
//     mju_scl(Z + n * t + nq, q1, 1.0 / h_, nv);
//     mju_addToScl(Z + n * t + nq, q0, -1.0 / h_, nv);

//     // acceleration 
//     mju_scl(Z + n * t + nq + nv, q2, 1.0 / (h_ * h_), na);
//     mju_addToScl(Z + n * t + nq + nv, q1, -2.0 / (h_ * h_), na);
//     mju_addToScl(Z + n * t + nq + nv, q0, 1.0 / (h_ * h_), na);
//   }
// }

// // TODO(taylor): acceleration
// void ConfigurationToStateMapping(double* M, int T) {
//   // set to zero 
//   mju_zero(M, (n * T) * (2 * nq * T));

//   // time step for finite difference
//   double h_ = h * (T - 1) / (2 * T - 1);

//   for (int t = 0; t < T; t++) {
//     // configuration 
//     mjpc::SetMatrixInMatrix(M, In.data(), 1.0, n * T, 2 * nq * T, nq, nq, n * t, 2 * nq * t + nq);

//     // velocity 
//     mjpc::SetMatrixInMatrix(M, Inq.data(), -1.0 / h_, n * T, 2 * nq * T, nv, nv, n * t + nq, 2 * nq * t);
//     mjpc::SetMatrixInMatrix(M, Inq.data(),  1.0 / h_, n * T, 2 * nq * T, nv, nv, n * t + nq, 2 * nq * t + nq);

//     // acceleration 
//     // TODO(taylor): check index overflow
//     if (na > 0) {
//       // mjpc::SetMatrixInMatrix(M, Ina.data(),  1.0 / (h_ * h_), n * T, 2 * nq * T, nv, nv, n * t + nq + nv, 2 * nq * t);
//       // mjpc::SetMatrixInMatrix(M, Ina.data(), -2.0 / (h_ * h_), n * T, 2 * nq * T, nv, nv, n * t + nq + nv, 2 * nq * t + nq);
//       // mjpc::SetMatrixInMatrix(M, Ina.data(),  1.0 / (h_ * h_), n * T, 2 * nq * T, nv, nv, n * t + nq + nv, 2 * nq * t + nq + nq);
//     }
//   }
// }

// // cumulative cost gradient wrt to configurations 
// void Jq(double* Jq_, const double* Z, const double* U, const double* Y, const double* P, const double* S, const double* R, int T) {
//   double Jz_[n * T];
//   double M[(n * T) * (2 * nq * T)];

//   // intermediate results
//   Jz(Jz_, Z, U, Y, P, S, R, T);
//   ConfigurationToStateMapping(M, T);

//   // total gradient 
//   mju_mulMatTVec(Jq_, M, Jz_, n * T, 2 * nq * T);
// }

// // cumulative cost Hessian wrt to configurations
// void Jqq(double* Jqq_, const double* Z, const double* U, const double* Y, const double* P, const double* S, const double* R, int T) {
//   double Jzz_[(n * T) * (n * T)];
//   double M[(n * T) * (2 * nq * T)];
//   double tmp[(n * T) * (2 * nq * T)];

//   // intermediate results
//   Jzz(Jzz_, Z, U, Y, P, S, R, T);
//   ConfigurationToStateMapping(M, T);

//   // total Hessian 
//   mju_mulMatMat(tmp, Jzz_, M, n * T, n * T, 2 * nq * T);
//   mju_mulMatTMat(Jqq_, M, tmp, n * T, 2 * nq * T, 2 * nq * T);
// }

// // run event loop
// int main(int argc, char** argv) {
//   // display an error if running on macOS under Rosetta 2
// #if defined(__APPLE__) && defined(__AVX__)
//   if (rosetta_error_msg) {
//     DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
//     std::exit(1);
//   }
// #endif

//   printf("State Estimation\n\n");

//   // ----- random problem ----- //
//   const int T = 3; // horizon
//   std::vector<double> x0{0.1, 0.2}; // initial state

//   // cost weights 
//   std::vector<double> P;
//   P.resize((n * (T - 1)) * (n * (T - 1)));
//   mju_eye(P.data(), n * (T - 1));
//   mju_scl(P.data(), P.data(), 2.0, (n * (T - 1)) * (n * (T - 1)));

//   std::vector<double> S;
//   S.resize((p * (T - 1)) * (p * (T - 1)));
//   mju_eye(S.data(), p * (T - 1));
//   mju_scl(S.data(), S.data(), 3.0, (p * (T - 1)) * (p * (T - 1)));

//   std::vector<double> R;
//   R.resize((m * (T - 1)) * (m * (T - 1)));
//   mju_eye(R.data(), m * (T - 1));
//   mju_scl(R.data(), R.data(), 4.0, (m * (T - 1)) * (m * (T - 1)));


//   // random actions
//   std::vector<double> U;
//   U.resize(m * (T - 1));

//   absl::BitGen gen_;
//   U.resize(m * (T - 1));
//   for (int i = 0; i < m * (T - 1); i++) {
//     U[i] = 0.1;//absl::Gaussian<double>(gen_, 0.0, 1.0e-3);
//   }

//   // state and sensor trajectories 
//   std::vector<double> X;
//   X.resize(n * T);
//   std::vector<double> Y;
//   Y.resize(p * (T - 1));

//   // set initial state
//   mju_copy(X.data(), x0.data(), n);

//   // rollout 
//   for (int t = 0; t < T - 1; t++) {
//     g(mjpc::DataAt(Y, p * t), mjpc::DataAt(X, n * t), mjpc::DataAt(U, m * t));
//     f(mjpc::DataAt(X, n * (t + 1)), mjpc::DataAt(X, n * t), mjpc::DataAt(U, m * t));
//   }

//   printf("actions:\n");
//   mju_printMat(U.data(), T - 1, m);

//   printf("rolled out states\n");
//   mju_printMat(X.data(), T, n);

//   printf("rolled out sensor values\n");
//   mju_printMat(Y.data(), T - 1, p);

//   // configuration trajectory decision variables 
//   std::vector<double> Z;
//   Z.resize(n * T);

//   // set matrix within matrix 
//   const int q = 8;
//   std::vector<double> W;
//   W.resize(q * q);
//   mju_zero(W.data(), q * q);

//   std::vector<double> W1{1.0, 2.0, 3.0, 4.0};
//   std::vector<double> W2{5.0, 6.0, 7.0, 8.0};
//   std::vector<double> W3{9.0, 10.0, 11.0, 12.0};
//   std::vector<double> W4{13.0, 14.0, 15.0, 16.0};

//   printf("W (initial):\n");
//   mju_printMat(W.data(), q, q);

//   printf("W1: \n");
//   mju_printMat(W1.data(), 2, 2);

//   printf("W2: \n");
//   mju_printMat(W2.data(), 2, 2);

//   printf("W3: \n");
//   mju_printMat(W3.data(), 2, 2);

//   printf("W4: \n");
//   mju_printMat(W4.data(), 2, 2);

//   // set W1 
//   mjpc::SetMatrixInMatrix(W.data(), W1.data(), 1.0, q, q, 2, 2, 0, 0);

//   // set W2
//   mjpc::SetMatrixInMatrix(W.data(), W2.data(), 1.0, q, q, 2, 2, 0, 4);

//   // set W3
//   mjpc::SetMatrixInMatrix(W.data(), W3.data(), 1.0, q, q, 2, 2, 4, 0);

//   // set W4
//   mjpc::SetMatrixInMatrix(W.data(), W4.data(), 1.0, q, q, 2, 2, 4, 4);

//   printf("W (filled):\n");
//   mju_printMat(W.data(), q, q);

//   // residuals 
//   std::vector<double> r1_;
//   r1_.resize(n * (T - 1));
//   std::vector<double> r2_;
//   r2_.resize(p * (T - 1));
//   std::vector<double> r3_;
//   r3_.resize(m * (T - 1));

//   r1(r1_.data(), X.data(), U.data(), T);
//   printf("r1:\n");
//   mju_printMat(r1_.data(), n * (T - 1), 1);

//   r2(r2_.data(), X.data(), U.data(), Y.data(), T);
//   printf("r2:\n");
//   mju_printMat(r2_.data(), p * (T - 1), 1);

//   r3(r3_.data(), X.data(), U.data(), T);
//   printf("r3:\n");
//   mju_printMat(r3_.data(), m * (T - 1), 1);

//   // residual Jacobians 
//   std::vector<double> r1z_;
//   r1z_.resize((n * (T - 1)) * (n * T));
//   std::vector<double> r2z_;
//   r2z_.resize((p * (T - 1)) * (n * T));
//   std::vector<double> r3z_;
//   r3z_.resize((m * (T - 1)) * (n * T));

//   r1z(r1z_.data(), X.data(), U.data(), T);
//   printf("r1z:\n");
//   mju_printMat(r1z_.data(), n * (T - 1), n * T);

//   r2z(r2z_.data(), X.data(), U.data(), Y.data(), T);
//   printf("r2z:\n");
//   mju_printMat(r2z_.data(), p * (T - 1), n * T);

//   r3z(r3z_.data(), X.data(), U.data(), T);
//   printf("r3z:\n");
//   mju_printMat(r3z_.data(), m * (T - 1), n * T);

//   // ---- costs ----- //

//   // cost 1 
//   double J1_ = J1(X.data(), U.data(), P.data(), T);
//   printf("J1: %f\n", J1_);

//   // cost gradient 1 
//   std::vector<double> J1z_;
//   J1z_.resize(n * T);
//   J1z(J1z_.data(), X.data(), U.data(), P.data(), T);
//   printf("J1z: \n");
//   mju_printMat(J1z_.data(), n * T, 1);

//   // cost Hessian 1 
//   std::vector<double> J1zz_;
//   J1zz_.resize((n * T) * (n * T));
//   J1zz(J1zz_.data(), X.data(), U.data(), P.data(), T);
//   printf("J1zz: \n");
//   mju_printMat(J1zz_.data(), n * T, n * T);

//   printf("P: \n");
//   mju_printMat(P.data(), n * (T - 1), n * (T - 1));

//   // cost 2 
//   double J2_ = J2(X.data(), U.data(), Y.data(), S.data(), T);
//   printf("J2: %f\n", J2_);

//   // cost gradient 2 
//   std::vector<double> J2z_;
//   J2z_.resize(n * T);
//   J2z(J2z_.data(), X.data(), U.data(), Y.data(), S.data(), T);
//   printf("J2z: \n");
//   mju_printMat(J2z_.data(), n * T, 1);

//   // cost Hessian 2
//   std::vector<double> J2zz_;
//   J2zz_.resize((n * T) * (n * T));
//   J2zz(J2zz_.data(), X.data(), U.data(), Y.data(), S.data(), T);
//   printf("J2zz: \n");
//   mju_printMat(J2zz_.data(), n * T, n * T);

//   // cost 3
//   double J3_ = J3(X.data(), U.data(), R.data(), T);
//   printf("J3: %f\n", J3_);

//   // cost gradient 3
//   std::vector<double> J3z_;
//   J3z_.resize(n * T);
//   J3z(J3z_.data(), X.data(), U.data(), R.data(), T);
//   printf("J3z: \n");
//   mju_printMat(J3z_.data(), n * T, 1);

//   // cost Hessian 3
//   std::vector<double> J3zz_;
//   J3zz_.resize((n * T) * (n * T));
//   J3zz(J3zz_.data(), X.data(), U.data(), R.data(), T);
//   printf("J3zz: \n");
//   mju_printMat(J3zz_.data(), n * T, n * T);

//   // cumulative cost 
//   double J_ = J(X.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);
//   printf("J: %f\n", J_);

//   // cumulative cost gradient 
//   std::vector<double> Jz_;
//   Jz_.resize(n * T);
//   Jz(Jz_.data(), X.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);
//   printf("Jz: \n");
//   mju_printMat(Jz_.data(), n * T, 1);

//   // cumulative cost Hessian 
//   std::vector<double> Jzz_;
//   Jzz_.resize((n * T) * (n * T));
//   Jzz(Jzz_.data(), X.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);
//   printf("Jzz: \n");
//   mju_printMat(Jzz_.data(), n * T, n * T);

//   // configuration mapping 
//   std::vector<double> M;
//   M.resize((n * T) * (2 * nq * T));
//   ConfigurationToStateMapping(M.data(), T);
//   printf("configuration to state mapping: \n");
//   mju_printMat(M.data(), n * T, 2 * nq * T);

//   // check mapping
//   mjpc::LinearSolve solver;
//   std::vector<double> Q(2 * nq * T);
//   solver.Initialize(n * T, 2 * nq * T);
//   solver.Solve(Q.data(), M.data(), X.data());

//   printf("X: \n");
//   mju_printMat(X.data(), n * T, 1);

//   printf("Q: \n");
//   mju_printMat(Q.data(), 2 * nq * T, 1);

//   // configuration to state conversion
//   std::vector<double> Z_(n * T);
//   ConfigurationToState(Z_.data(), Q.data(), T);

//   printf("recovered Z:\n");
//   mju_printMat(Z_.data(), n * T, 1);

//   // Jq 
//   std::vector<double> Jq_(2 * nq * T);
//   std::vector<double> X0(n * T);
//   mju_zero(X0.data(), n * T);
//   Jq(Jq_.data(), X0.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);

//   printf("Jq000: \n");
//   mju_printMat(Jq_.data(), 2 * nq * T, 1);

//   Jz(Jz_.data(), X0.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);
//   printf("Jz000: \n");
//   mju_printMat(Jz_.data(), n * T, 1);


//   // Jqq 
//   std::vector<double> Jqq_((2 * nq * T) * (2 * nq * T));
//   Jqq(Jqq_.data(), X.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);

//   printf("Jqq: \n");
//   mju_printMat(Jqq_.data(), 2 * nq * T, 2 * nq * T);

//   // ----- optimize ----- //

//   // allocate
//   std::vector<double> Qo(2 * nq * T);
//   std::vector<double> Zo(n * T);
//   std::vector<double> Qc(2 * nq * T);
//   std::vector<double> Zc(n * T);
//   std::vector<double> grad(2 * nq * T);
//   std::vector<double> hess((2 * nq * T) * (2 * nq * T));
//   std::vector<double> dir(2 * nq * T);

//   // initialize 
//   mju_zero(Qo.data(), 2 * nq * T);

//   // 
//   ConfigurationToState(Zo.data(), Qo.data(), T);
//   double obj = J(Zo.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);
//   printf("obj (0): %f\n", obj);

//   for (int i = 0; i < 10; i++) {
//     // gradient
//     Jq(grad.data(), Zo.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);

//     printf("grad:\n");
//     mju_printMat(grad.data(), 2 * nq * T, 1);

//     // check convergence
//     double residual = mju_norm(grad.data(), 2 * nq * T) / (2 * nq * T);
//     if (residual < 1.0e-3) {
//       printf("Solved!\n");
//       printf("Solution: \n");
//       mju_printMat(Qo.data(), 2 * nq * T, 1);
//       return 0;
//     }

//     // Hessian 
//     Jqq(hess.data(), Zo.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);
    
//     printf("hess:\n");
//     mju_printMat(hess.data(), 2 * nq * T, 2 * nq * T);

//     // compute search direction
//     mju_cholFactor(hess.data(), 2 * nq * T, 0.0);
//     mju_cholSolve(dir.data(), hess.data(), grad.data(), 2 * nq * T);

//     double step = 1.0;
//     int iter = 0;

//     mju_copy(Qc.data(), Qo.data(), 2 * nq * T);
//     mju_addToScl(Qc.data(), dir.data(), -1.0 * step, 2 * nq * T);
//     ConfigurationToState(Zc.data(), Qc.data(), T);
//     double obj_c = J(Zc.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);

//     while(obj_c >= obj) {
//       step *= 0.5;
//       mju_copy(Qc.data(), Qo.data(), 2 * nq * T);
//       mju_addToScl(Qc.data(), dir.data(), -1.0 * step, 2 * nq * T);
//       ConfigurationToState(Zc.data(), Qc.data(), T);
//       obj_c = J(Zc.data(), U.data(), Y.data(), P.data(), S.data(), R.data(), T);

//       iter += 1;

//       if (iter > 20) {
//         printf("line search failure\n");
//         mju_printMat(Qo.data(), 2 * nq * T, 1);
//         return 0;
//       }
//     }
//     mju_copy(Qo.data(), Qc.data(), 2 * nq * T);
//     mju_copy(Zo.data(), Zc.data(), n * T);
//     obj = obj_c;
//     printf("obj (%i): %f\n", i + 1, obj);
//   }

//   printf("optimization failure\n");


//   return 0;
// }
