
// #include "mobile.h"

// namespace F = torch::nn::functional;

// void get_submodule_mobilenet(torch::jit::script::Module module, Net &net){
// 	Layer t_layer;
// 	for(auto children : module.named_children()){
// 		if(children.name == "features"){
// 			for(auto inverted : children.value.named_children()){
// 				int idx = net.layers.size();
// 				if(inverted.name == "0" || inverted.name == "18"){	//ConvBNReLU
// 					t_layer.layer = inverted.value;
// 					t_layer.name = "ConvBNReLU";
// 					net.layers.push_back(t_layer);
// 				}
// 				else{	//InvertedResidual
// 					for(auto block : inverted.value.named_children()){	//conv,in InvertedResidual
// 						for(auto conv : block.value.named_children()){//0,1,2 in conv
// 							t_layer.layer = conv.value;
// 							if(inverted.name == "1"){	//InvertedResidual_1
// 								if(conv.name == "0")	t_layer.name = "ConvBNReLU";
// 								else if(conv.name == "1")	t_layer.name = "conv";
// 								else if(conv.name == "2")	t_layer.name = "bn";	//relu6
// 							}else{
// 								if(conv.name == "0")	t_layer.name = "ConvBNReLU";
// 								else if(conv.name == "1")	t_layer.name = "ConvBNReLU";
// 								else if(conv.name == "2")	t_layer.name = "conv";
// 								else if(conv.name == "3")	t_layer.name = "bn";
// 							}
// 							net.layers.push_back(t_layer);
// 						}
// 					}
// 					if(inverted.name == "3" || inverted.name == "5" || inverted.name == "6" || inverted.name == "8" || inverted.name == "9" || \
// 					inverted.name == "10" || inverted.name == "12" || inverted.name == "13" || inverted.name == "15" || inverted.name == "16"){ // use_res_connect = self.stride == 1 and inp == oup
// 						net.layers.back().name = "last_use_res_connect";
// 						net.layers[idx].name = "first_use_res_connect";
// 					}
// 				}
// 			}
// 		}
// 		else if(children.name == "classifier"){
// 			for(auto child : children.value.named_children()){
// 				t_layer.layer = child.value;
// 				t_layer.name = "dropout+linear";
// 				net.layers.push_back(t_layer); 
// 			}
// 		}
// 	}
// }

// void *predict_mobilenet(Net *mobile){
// 	{
//         at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
// 		int i;
// 		float time;
// 		cudaEvent_t start, end;
// 		cudaEventCreate(&start);
// 		cudaEventCreate(&end);
// 		cudaEventRecord(start);
// 		for(i=0;i<mobile->layers.size();i++){
// 			pthread_mutex_lock(&mutex_t[mobile->index_n]);
// 			cond_i[mobile->index_n] = 1;
			
// 			netlayer nl;
// 			nl.net = mobile;
// 			nl.net->index = i;

// 			th_arg th;
// 			th.arg = &nl;

//             cal_kernels_enqueue(mobile);

// 			#if RECORD
// 			cudaEvent_t l_start, l_end;
// 			float l_time;
// 			cudaEventCreate(&l_start);
// 			cudaEventCreate(&l_end);
// 			cudaEventRecord(l_start);
// 			#endif

// 			thpool_add_work(thpool,(void(*)(void *))forward_mobilenet,&th);
			
// 			while (cond_i[mobile->index_n] == 1)
// 			{
// 				pthread_cond_wait(&cond_t[mobile->index_n], &mutex_t[mobile->index_n]);
// 			}
			
// 			#if L_SYNC
// 			cudaStreamSynchronize(streams[mobile->index_s%(n_streamPerPool)]); // 나중에 지워야함
// 			#endif

// 			cal_kernels_dequeue(mobile,i);

// 			#if RECORD
// 			cudaEventRecord(l_end);
// 			cudaEventSynchronize(l_end);
// 			cudaEventElapsedTime(&l_time, l_start, l_end);
// 			if(mobile->warming==true){
//             	fprintf((mobile->fp),"%d,%d,%d,%lf\n",i,mobile->layers[i].l_kernel,mobile->all_kernels,l_time);
//         	}
// 			#endif

// 			i = mobile->index;
// 			mobile->input.clear();
// 			mobile->input.push_back(mobile->layers[i].output);
// 			pthread_mutex_unlock(&mutex_t[mobile->index_n]);
// 		}
// 		cudaStreamSynchronize(streams[mobile->index_s%(n_streamPerPool)]);
// 		cudaEventRecord(end);
// 		cudaEventSynchronize(end);
// 		cudaEventElapsedTime(&time, start, end);
// 		std::cout << "\n*****"<<mobile->name<<" result " <<time/1000<<"s ***** \n";
// 		std::cout << (mobile->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
// 	}
// }

// void forward_mobilenet(th_arg *th){
// 	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
// 	{
// 		at::cuda::CUDAStreamGuard guard(streams[(th->arg->net->index_s)%(n_streamPerPool)]);

// 		#if NVTX	
// 		char str[30];
// 		sprintf(str, "Mobile layer - %d", th->arg->net->index);
// 		nvtxRangeId_t id1 = nvtxRangeStartA(str);
// 		#endif

// 		netlayer *nl = th->arg;
// 		std::vector<torch::jit::IValue> inputs = nl->net->input;
// 		int k = nl->net->index;
// 		at::Tensor out;

// 		std::cout<<k<<" , "<<nl->net->layers[k].name<<"\n";
		
// 		if(nl->net->layers[k].name == "first_use_res_connect"){
// 			nl->net->identity = inputs[0].toTensor();
// 		}
// 		// ***임시로 flatten 변수 사용***
// 		if(k == nl->net->flatten){	// nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
// 			out = torch::nn::functional::adaptive_avg_pool2d(inputs[0].toTensor(), \
// 			F::AdaptiveAvgPool2dFuncOptions(1)).reshape({inputs[0].toTensor().size(0), -1});
// 			inputs.clear();
// 			inputs.push_back(out);
// 		}
// 		/*
// 		InvertedResidual forward function

// 		def forward(self, x: Tensor) -> Tensor:
// 			if self.use_res_connect:
// 				return x + self.conv(x)
// 			else:
// 				return self.conv(x)
// 		*/
// 		if(nl->net->layers[k].name == "last_use_res_connect"){
// 			out = nl->net->layers[k].layer.forward(inputs).toTensor();
// 			out = nl->net->identity + out;
// 		}
// 		else{
// 			out = nl->net->layers[k].layer.forward(inputs).toTensor();

// 			// if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "bn"){
// 			// 	nl->net->layers[k].output = out;
// 			// 	k++;
// 			// 	inputs.clear();
// 			// 	inputs.push_back(out);
// 			// 	out = nl->net->layers[k].layer.forward(inputs).toTensor();
// 			// }
// 		}
		
// 		nl->net->layers[k].output = out;
// 		nl->net->index = k;

// 		#if NVTX
// 		nvtxRangeEnd(id1);
// 		#endif
// 	}
// 	cond_i[th->arg->net->index_n]=0;
// 	pthread_cond_signal(&cond_t[th->arg->net->index_n]);
// 	pthread_mutex_unlock(&mutex_t[th->arg->net->index_n]);		
// }

