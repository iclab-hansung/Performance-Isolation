#ifndef SHUFFLE_H
#define SHUFFLE_H

#include "net.h"
#include "test.h"
#include "thpool.h"

extern void cal_warp_enqueue(Net *net);
extern void cal_warp_dequeue(Net *net,int index);
at::Tensor channel_shuffle(at::Tensor x, int groups);
void get_submodule_shuffle(torch::jit::script::Module module, Net &net);
void *predict_shuffle(Net *input);
void forward_shuffle(th_arg *th);

#endif
