#ifndef MOBILE_H
#define MOBILE_H

#include "test.h"
#include "balancecheck.h"

void get_submodule_mobilenet(torch::jit::script::Module module, Net &net);
void *predict_mobilenet(Net *input);
void forward_mobilenet(th_arg *th);
#endif

