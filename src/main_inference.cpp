#include "test.h"
#include <cuda_profiler_api.h>

// #define n_threads 1
#define WARMING 4 //4
// #define N_GPU 3

extern void *predict_densenet_warming(Net *dense);
extern void *predict_resnet_warming(Net *res);
extern void *predict_alexnet_warming(Net *alex);
extern void *predict_inception_warming(Net *inception);
extern void *predict_efficientnet_warming(Net *input);

extern void *predict_densenet(std::vector<Net*> *vec_dense);
extern void *predict_resnet(std::vector<Net*> *vec_res);
extern void *predict_alexnet(std::vector<Net*> *vec_alex);
extern void *predict_inception(std::vector<Net*> *vec_inception);
extern void *predict_efficientnet(std::vector<Net*> *vec_efficient);

namespace F = torch::nn::functional;
using namespace std;

FILE *fp_res = fopen("../0325-E27-p12-iter150-single-test.txt","w");
// FILE *fp_gpu = fopen("../test_g28.txt","w");

threadpool thpool[N_GPU];

pthread_cond_t* cond_t;
pthread_mutex_t* mutex_t;
int* cond_i;

pthread_mutex_t* mutex_g;

pthread_cond_t* cond_p;
int *cond_p_i;
// #if Q_OVERHEAD
//   pthread_cond_t* cond_q;
//   int * cond_q_i;
// #endif

std::vector<std::vector<at::cuda::CUDAStream>> streams;
// std::vector<at::cuda::CUDAStream> stream;

c10::DeviceIndex GPU_NUM=0;

#if CPU_PINNING
  cpu_set_t cpuset;
  int n_cpu = (27 - n_threads);
  vector<int> cpu_list = {21,22,23,24};
#endif

float CONST_P = 15; //ms  dense 60 res 60 alex 3 vgg 11
// int total_weight = 0;


vector<vector<float>> input_layer_values(const std::string& filePath){
  std::ifstream fs(filePath);

  if (true == fs.fail())
  {
      throw std::ifstream::failure("fail to open file");
  }
  vector<vector<float>> indexkernel;
  vector<float> data;

  while(!fs.eof()){

    string str_buf;
    float tmp;
    getline(fs,str_buf);
    size_t prev=0;
    size_t current;
    string substring;
    current = str_buf.find(',');

    while(current != string::npos){
      substring=str_buf.substr(prev,current-prev);
      prev = current + 1;
      current = str_buf.find(',',prev);
      tmp = std::stof(substring);
      data.push_back(tmp); 
    }
    substring=str_buf.substr(prev,current-prev);  //last
    tmp = std::stof(substring);
    data.push_back(tmp);
    indexkernel.push_back(data);
    data.clear();
  }
  fs.close();
  return indexkernel;
}


int nice2prio(int nice){
  int prio = nice+20;
  return prio;
}

float cal_timeslice(int total_weight,int my_weight){ 
  float timeslice = (float)(((float)my_weight/(float)total_weight)*CONST_P);
  return timeslice;
}


#if RECORD || Q_OVERHEAD || MEM_RECORD
string result_path;
#endif

vector<vector<torch::jit::IValue>> inputs(N_GPU);
vector<vector<torch::jit::IValue>> inputs2(N_GPU);
vector<vector<torch::jit::IValue>> inputs3(N_GPU);

vector<Gpu> gpu_list;
vector<int> gpu_idx={0};
vector<int> gpu_num={0, 0, 1, 2, 1, 2, 3, 1, 3, 2, 2, 3, 3, 0, 0, 0, 2, 3, 1, 1, 0, 0, 2, 2, 0, 0, 3, 1, 2, 2, 1, 3};

int gpu_n = gpu_idx.size();

int main(int argc, const char* argv[]) {
  GPU_NUM=atoi(argv[1]);
  c10::cuda::set_device(GPU_NUM);
  torch::Device device = {at::kCUDA,GPU_NUM};

  #if RECORD || Q_OVERHEAD || MEM_RECORD
    result_path = argv[2];
  #endif

  int n_dense=atoi(argv[3]);
  int n_res=atoi(argv[4]);
  int n_alex=atoi(argv[5]);
  int n_vgg=atoi(argv[6]);
  int n_wide=atoi(argv[7]);
  int n_squeeze=atoi(argv[8]);
  int n_mobile=atoi(argv[9]);
  int n_mnasnet=atoi(argv[10]);
  int n_inception=atoi(argv[11]);
  int n_shuffle=atoi(argv[12]);
  int n_resX=atoi(argv[13]);
  int n_efficient=atoi(argv[14]);
  
  #if RECORD || Q_OVERHEAD || MEM_RECORD
    std::string filename;
    if(n_dense) filename += "D"+to_string(n_dense);
    if(n_res) filename += "R"+to_string(n_res);
    if(n_alex) filename += "A"+to_string(n_alex);
    if(n_vgg) filename += "V"+to_string(n_vgg);
    if(n_wide) filename += "W"+to_string(n_wide);
    if(n_mobile) filename += "M"+to_string(n_mobile);
    if(n_mnasnet) filename += "N"+to_string(n_mnasnet);
    if(n_inception) filename += "I"+to_string(n_inception);
    if(n_resX) filename += "X"+to_string(n_resX);
    if(n_efficient) filename += "E"+to_string(n_efficient);
    //X5 - X1 : resX 5개 중 index 1 record
  #endif

  srand(time(NULL));

  int n_all = n_alex + n_vgg + n_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX + n_efficient;

  static int stream_index_H = 0;
  static int branch_index_H = 31;

  for(int g=0;g<N_GPU;g++){
    thpool[g] = thpool_init(n_threads,g);
  }

  vector<int> nice_list = {-3,0,5,10};

 // Gpu *gpu_list;
  // gpu_list = (Gpu *)malloc(sizeof(Gpu)*N_GPU);

  /*struct init*/
  streams.resize(N_GPU); // streams[][] 형식

  for(int i=0; i<gpu_idx.size(); i++){
    for(int j=0; j<n_streamPerPool;j++){
      streams[gpu_idx[i]].push_back(at::cuda::getStreamFromPool(true,gpu_idx[i]));//(c10::DeviceIndex)i));
    }
  }
  
  gpu_list.resize(N_GPU);
  
  for(int i=0;i<gpu_idx.size();i++){
    std::cout<<"***** GPU "<<gpu_idx[i]<<" INIT *****\n";
    Gpu g;
    g.g_index = gpu_idx[i];
    g.all_api = 0;
    g.total_weight = 0;
    g.load = 0;
    g.g_device = {at::kCUDA,(c10::DeviceIndex)g.g_index};
    g.g_stream = streams[gpu_idx[i]];
    gpu_list[gpu_idx[i]] = g;
    // g.n_net = 0;
    // g.last_cnt = 0; //필요?
  } 

  torch::jit::script::Module denseModule[N_GPU];
  torch::jit::script::Module resModule[N_GPU];
  torch::jit::script::Module alexModule[N_GPU];
  torch::jit::script::Module vggModule[N_GPU];
  torch::jit::script::Module inceptionModule[N_GPU];
  torch::jit::script::Module efficientModule[N_GPU];

  try {
    for(int i=0; i<gpu_n; i++){
      if(n_dense){
        denseModule[gpu_idx[i]] = torch::jit::load("../model/densenet201_model.pt",gpu_list[gpu_idx[i]].g_device);
      }
      if(n_res){
        resModule[gpu_idx[i]] = torch::jit::load("../model/resnet152_model.pt",gpu_list[gpu_idx[i]].g_device);
      }
      if(n_alex){
    	  alexModule[gpu_idx[i]] = torch::jit::load("../model/alexnet_model.pt",gpu_list[gpu_idx[i]].g_device);
      }
      if(n_inception){
        inceptionModule[gpu_idx[i]] = torch::jit::load("../model/inception_model.pt",gpu_list[gpu_idx[i]].g_device);
      }
      if(n_efficient){
        efficientModule[gpu_idx[i]] = torch::jit::load("../model/efficient_b3_model.pt",gpu_list[gpu_idx[i]].g_device);
      }
    }
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }
  cout<<"***** Model Load compelete *****"<<"\n";

  //Network Mutex
  cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
  mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
  cond_i = (int *)malloc(sizeof(int) * n_all);

  for (int i = 0; i < n_all; i++)
  {
      pthread_cond_init(&cond_t[i], NULL);
      pthread_mutex_init(&mutex_t[i], NULL);
      cond_i[i] = 0;
  }

//GPU Mutex
  mutex_g = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) *N_GPU);

  for(int i=0;i<N_GPU;i++)
  {
    pthread_mutex_init(&mutex_g[i], NULL);
  }

  for(int i=0;i<gpu_n;i++){
    torch::Tensor x = torch::ones({1,3,224,224}).to(gpu_list[gpu_idx[i]].g_device); 
    inputs[gpu_list[gpu_idx[i]].g_index].push_back(x);
    
    if(n_inception){
      torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(gpu_list[gpu_idx[i]].g_device);

      auto x_ch0 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
      auto x_ch1 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
      auto x_ch2 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
        
      x_ch0.to(gpu_list[gpu_idx[i]].g_device);
      x_ch1.to(gpu_list[gpu_idx[i]].g_device);
      x_ch2.to(gpu_list[gpu_idx[i]].g_device);

      auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(gpu_list[gpu_idx[i]].g_device);
      inputs2[gpu_list[gpu_idx[i]].g_index].push_back(x_cat);
    }

    if(n_efficient){
      torch::Tensor x3 = torch::ones({1, 3, 300, 300}).to(gpu_list[gpu_idx[i]].g_device);
      inputs3[gpu_list[gpu_idx[i]].g_index].push_back(x3);
    }
  }

  at::Tensor out;

  pthread_t networkArray_dense[n_dense];
  Net dense[n_dense][N_GPU];
  vector<Net*> multig_dense;
  vector< vector<Net*> > net_input_dense;
  multig_dense.resize(N_GPU);
  net_input_dense.resize(n_dense);


  vector<Net*> multig_res;
  vector< vector<Net*> > net_input_res;
  Net res[n_res][N_GPU];
  pthread_t networkArray_res[n_res];
  multig_res.resize(N_GPU);
  net_input_res.resize(n_res);

  vector<Net*> multig_alex;
  vector< vector<Net*> > net_input_alex;
  Net alex[n_alex][N_GPU];
  pthread_t networkArray_alex[n_alex];

  vector<Net*> multig_inception;
  vector< vector<Net*> > net_input_inception;
  Net inception[n_inception][N_GPU];
  pthread_t networkArray_inception[n_inception];
  multig_inception.resize(N_GPU);
  net_input_inception.resize(n_inception);

  vector<Net*> multig_efficient;
  vector< vector<Net*> > net_input_efficient;
  Net efficient[n_efficient][N_GPU];
  pthread_t networkArray_efficient[n_efficient];
  multig_efficient.resize(N_GPU);
  net_input_efficient.resize(n_efficient);

  for(int i=0;i<n_dense;i++){
    for(int g=0;g<N_GPU;g++){
      for(int j=0;j<gpu_n;j++){
        if(g==gpu_idx[j]){
          get_submodule_densenet(denseModule[g], dense[i][g]);
          dense[i][g].flatten = dense[i][g].layers.size()-1;
          dense[i][g].device = &gpu_list[g];
          // dense[i][g].g_index = g==gpu_idx[0] ? gpu_idx[rand()%gpu_n] : dense[i][gpu_idx[0]].g_index;
          dense[i][g].name = "DenseNet";
          dense[i][g].index_n = i;
          if(gpu_idx.size()>1){
            dense[i][g].g_index = gpu_num[dense[i][g].index_n];
          }else{
            dense[i][g].g_index = gpu_idx[0];
          }
          dense[i][g].index_s = stream_index_H;
          dense[i][g].nice = nice_list[dense[i][g].index_n%nice_list.size()];
          dense[i][g].weight = prio_to_weight[nice2prio(dense[i][g].nice)];
          dense[i][g].timeslice = CONST_P;
          dense[i][g].last = 801;
          dense[i][g].input = inputs[g];
          dense[i][g].change_gid = false;
          // dense[i][g].device->n_net += 1;
          dense[i][g].warming = false;
          dense[i][g].all_api = 0;
          #if RECORD || Q_OVERHEAD || MEM_RECORD
            /*=============FILE===============*/
            dense[i][g].fp = fopen((result_path+"dense/"+filename+"-"+"D"+to_string(i)+".txt").c_str(),"a"); 
          #endif

          vector<vector<float>> indexkernel;
          if(n_threads==1){
            indexkernel = input_layer_values("../layer_values/new_onlydense_th1.csv");  //detail file
          }else if(n_threads==3){
            indexkernel = input_layer_values("../layer_values/new_onlydense_th3.csv");  
          }else{
            indexkernel = input_layer_values("../layer_values/new_onlydense_th6.csv");  
          }
          int l_prev = 0;
          for(int l=0;l<dense[i][g].layers.size();l++){
            dense[i][g].layers[l].l_api = 0;
            dense[i][g].layers[l].l_mean = 0.0;
            dense[i][g].layers[l].l_mem = 0.0;

            for(int w=0;w<indexkernel.size();w++){
              if(l==(int)indexkernel[w][0]){
                dense[i][g].layers[l].l_mean = indexkernel[w][1];
                dense[i][g].layers[l].l_api = (int)indexkernel[w][2];
                dense[i][g].layers[l].l_mem = indexkernel[w][3];
                dense[i][g].layers[l].l_prev = l_prev;
                dense[i][g].layers[l_prev].l_next = l;
                if(l == dense[i][g].last){
                  dense[i][g].layers[l].l_next = l; //for last
                }
                l_prev = l;
                break;
              }
            }
          }
          for(int j=0;j<WARMING;j++){
            predict_densenet_warming(&dense[i][g]);
          }
          dense[i][g].input = inputs[dense[i][g].g_index];
          dense[i][g].warming = true;
          multig_dense[g]=(&(dense[i][g])); //하나의 net_index 0,1,2,3 
        }//if end
      }//gpu_n end 
    }//N_GPU end
    stream_index_H+=1;
    net_input_dense[i]=multig_dense;
  }

  for(int i=0;i<n_res;i++){
    for(int g=0;g<N_GPU;g++){
      for(int j=0;j<gpu_n;j++){
        if(g==gpu_idx[j]){
          get_submodule_resnet(resModule[g], res[i][g]);
          res[i][g].flatten = res[i][g].layers.size()-1;
          res[i][g].device = &gpu_list[g];
          // res[i][g].g_index = g==gpu_idx[0] ? gpu_idx[rand()%gpu_n] : res[i][gpu_idx[0]].g_index;
          res[i][g].name = "ResNet";
          res[i][g].index_n = i + n_dense;
          if(gpu_idx.size()>1){
            res[i][g].g_index = gpu_num[res[i][g].index_n];
          }else{
            res[i][g].g_index = gpu_idx[0];
          }
          res[i][g].index_s = stream_index_H;
          res[i][g].nice = nice_list[res[i][g].index_n%nice_list.size()];
          res[i][g].weight = prio_to_weight[nice2prio(res[i][g].nice)];
          res[i][g].timeslice = CONST_P;
          res[i][g].last = 308;
          res[i][g].input = inputs[g];
          res[i][g].change_gid = false;
          // res[i][g].device->n_net += 1;
          res[i][g].warming = false;
          res[i][g].all_api = 0;
          #if RECORD || Q_OVERHEAD || MEM_RECORD
            /*=============FILE===============*/
            res[i][g].fp = fopen((result_path+"res/"+filename+"-"+"R"+to_string(i)+".txt").c_str(),"a"); 
          #endif

          vector<vector<float>> indexkernel;
          if(n_threads==1){
            indexkernel = input_layer_values("../layer_values/new_onlyres_th1.csv");
          }else if(n_threads==3){
            indexkernel = input_layer_values("../layer_values/new_onlyres_th3.csv");  
          }else{
            indexkernel = input_layer_values("../layer_values/new_onlyres_th6.csv");  
          }
          int l_prev = 0;
          for(int l=0;l<res[i][g].layers.size();l++){
            res[i][g].layers[l].l_api = 0;
            res[i][g].layers[l].l_mean = 0.0;
            res[i][g].layers[l].l_mem = 0.0;

            for(int w=0;w<indexkernel.size();w++){
              if(l==(int)indexkernel[w][0]){
                res[i][g].layers[l].l_mean = indexkernel[w][1];
                res[i][g].layers[l].l_api = (int)indexkernel[w][2];
                res[i][g].layers[l].l_mem = indexkernel[w][3];
                res[i][g].layers[l].l_identity = indexkernel[w][4];
                res[i][g].layers[l].l_prev = l_prev;
                res[i][g].layers[l_prev].l_next = l;
                if(l == res[i][g].last){
                  res[i][g].layers[l].l_next = l; //for last
                }
                l_prev = l;
                break;
              }
            }
          }
          for(int j=0;j<WARMING;j++){
            predict_resnet_warming(&res[i][g]);
            
          }
          res[i][g].input = inputs[res[i][g].g_index];
          res[i][g].warming = true;
          multig_res[g]=(&(res[i][g])); //하나의 net_index 0,1,2,3 
        }
      }
    }
    stream_index_H+=1;
    net_input_res[i]=multig_res;
  }

  for(int i=0;i<n_alex;i++){
    for(int g=0;g<N_GPU;g++){
      get_submodule_alexnet(alexModule[g], alex[i][g]);
      alex[i][g].flatten = alex[i][g].layers.size()-5;
      alex[i][g].device = &gpu_list[g];
      alex[i][g].g_index = g==0? rand()%N_GPU : alex[i][0].g_index;
      alex[i][g].name = "AlexNet";
      alex[i][g].index_n = i + n_res + n_dense;
      alex[i][g].index_s = stream_index_H;
      alex[i][g].nice = nice_list[i%nice_list.size()];
      alex[i][g].weight = prio_to_weight[nice2prio(alex[i][g].nice)];
      alex[i][g].timeslice = CONST_P;
      alex[i][g].last = 18;
      alex[i][g].input = inputs[alex[i][0].g_index];
      alex[i][g].change_gid = false;
      // alex[i][g].device->n_net += 1;
      alex[i][g].warming = false;
      alex[i][g].all_api = 0;
      #if RECORD || Q_OVERHEAD || MEM_RECORD
        /*=============FILE===============*/
        alex[i][g].fp = fopen((result_path+"alex/"+filename+"-"+"A"+to_string(i)+".txt").c_str(),"a"); 
      #endif

      vector<vector<float>> indexkernel;
      if(n_threads==1){
        indexkernel = input_layer_values("../layer_values/new_onlyalex_th1.csv");
      }else if(n_threads==3){
        indexkernel = input_layer_values("../layer_values/new_onlyalex_th3.csv");
      }
      int l_prev = 0;
      for(int l=0;l<alex[i][g].layers.size();l++){
        alex[i][g].layers[l].l_api = 0;
        alex[i][g].layers[l].l_mean = 0.0;
        alex[i][g].layers[l].l_mem = 0.0;

        for(int w=0;w<indexkernel.size();w++){
          if(l==(int)indexkernel[w][0]){
            alex[i][g].layers[l].l_mean = indexkernel[w][1];
            alex[i][g].layers[l].l_api = (int)indexkernel[w][2];
            alex[i][g].layers[l].l_mem = indexkernel[w][3];
            alex[i][g].layers[l].l_prev = l_prev;
            alex[i][g].layers[l_prev].l_next = l;
            if(l == alex[i][g].last){
              alex[i][g].layers[l].l_next = l; //for last
            }
            l_prev = l;
            break;
          }
        }
      }
      for(int j=0;j<WARMING;j++){
        predict_alexnet_warming(&alex[i][g]);
        alex[i][g].device->all_api = 0;
      }
      alex[i][g].input = inputs[alex[i][g].g_index];
      alex[i][g].warming = true;
      multig_alex.push_back(&(alex[i][g])); //하나의 net_index 0,1,2,3 

    }
    stream_index_H+=1;
    net_input_alex.push_back(multig_alex);
    multig_alex.clear();
  }

  for(int i=0;i<n_inception;i++){
    for(int g=0;g<N_GPU;g++){
      for(int j=0;j<gpu_n;j++){
        if(g==gpu_idx[j]){
          get_submodule_inception(inceptionModule[g], inception[i][g]);
          inception[i][g].flatten = inception[i][g].layers.size()-1;
          inception[i][g].device = &gpu_list[g];
          // inception[i][g].g_index = g==gpu_idx[0] ? gpu_idx[rand()%gpu_n] : inception[i][gpu_idx[0]].g_index;
          inception[i][g].name = "Inception";
          inception[i][g].index_n = i + n_alex + n_res + n_dense;
          if(gpu_idx.size()>1){
            inception[i][g].g_index = gpu_num[inception[i][g].index_n];
          }else{
            inception[i][g].g_index = gpu_idx[0];
          }
          inception[i][g].index_s = stream_index_H;
          inception[i][g].nice = nice_list[inception[i][g].index_n%nice_list.size()];
          inception[i][g].weight = prio_to_weight[nice2prio(inception[i][g].nice)];
          inception[i][g].timeslice = CONST_P;
          inception[i][g].last = 123;
          inception[i][g].input = inputs2[g];
          inception[i][g].change_gid = false;
          // inception[i][g].device->n_net += 1;
          inception[i][g].warming = false;
          inception[i][g].all_api = 0;
          #if RECORD || Q_OVERHEAD || MEM_RECORD
            /*=============FILE===============*/
            inception[i][g].fp = fopen((result_path+"inception/"+filename+"-"+"I"+to_string(i)+".txt").c_str(),"a"); 
          #endif

          vector<vector<float>> indexkernel;
          if(n_threads==1){
            indexkernel = input_layer_values("../layer_values/new_onlyinception_th1.csv");
          }else if(n_threads==3){
            indexkernel = input_layer_values("../layer_values/new_onlyinception_th3.csv");  
          }else{
            indexkernel = input_layer_values("../layer_values/new_onlyinception_th6.csv");  
          }
          int l_prev = 0;
          for(int l=0;l<inception[i][g].layers.size();l++){
            inception[i][g].layers[l].l_api = 0;
            inception[i][g].layers[l].l_mean = 0.0;
            inception[i][g].layers[l].l_mem = 0.0;

            for(int w=0;w<indexkernel.size();w++){
              if(l==(int)indexkernel[w][0]){
                inception[i][g].layers[l].l_mean = indexkernel[w][1];
                inception[i][g].layers[l].l_api = (int)indexkernel[w][2];
                inception[i][g].layers[l].l_mem = indexkernel[w][3];
                inception[i][g].layers[l].l_prev = l_prev;
                inception[i][g].layers[l_prev].l_next = l;
                if(l == inception[i][g].last){
                  inception[i][g].layers[l].l_next = l; //for last
                }
                l_prev = l;
                break;
              }
            }
          }
          for(int j=0;j<WARMING;j++){
            predict_inception_warming(&inception[i][g]);
            inception[i][g].device->all_api = 0;
          }
          inception[i][g].input = inputs2[inception[i][g].g_index];
          inception[i][g].warming = true;
          multig_inception[g]=(&(inception[i][g])); //하나의 net_index 0,1,2,3 

        }
      }
    }
    stream_index_H+=1;
    net_input_inception[i]=multig_inception;
  }

  for(int i=0;i<n_efficient;i++){
    for(int g=0;g<N_GPU;g++){
      for(int j=0;j<gpu_n;j++){
        if(g==gpu_idx[j]){
          get_submodule_efficientnet(efficientModule[g], efficient[i][g]);
          efficient[i][g].flatten = efficient[i][g].layers.size()-1;
          efficient[i][g].device = &gpu_list[g];
          // efficient[i][g].g_index = g==gpu_idx[0] ? gpu_idx[rand()%gpu_n] : efficient[i][gpu_idx[0]].g_index;
          efficient[i][g].name = "EfficientNet";
          efficient[i][g].index_n = i + n_alex + n_res + n_dense + n_inception;
          if(gpu_idx.size()>1){
            efficient[i][g].g_index = gpu_num[efficient[i][g].index_n];
          }else{
            efficient[i][g].g_index = gpu_idx[0];
          }
          efficient[i][g].index_s = stream_index_H;
          efficient[i][g].nice = nice_list[efficient[i][g].index_n%nice_list.size()];
          efficient[i][g].weight = prio_to_weight[nice2prio(efficient[i][g].nice)];
          efficient[i][g].timeslice = CONST_P;
          efficient[i][g].last = 176;
          efficient[i][g].input = inputs3[g];
          efficient[i][g].change_gid = false;
          // efficient[i][g].device->n_net += 1;
          efficient[i][g].warming = false;
          efficient[i][g].all_api = 0;

          #if RECORD || Q_OVERHEAD || MEM_RECORD
            /*=============FILE===============*/
            efficient[i][g].fp = fopen((result_path+"efficient/"+filename+"-"+"E"+to_string(i)+".txt").c_str(),"a"); 
          #endif

          vector<vector<float>> indexkernel;
          if(n_threads==3){
            indexkernel = input_layer_values("../layer_values/new_onlyefficient_th3.csv");
          }else{
            indexkernel = input_layer_values("../layer_values/new_onlyefficient_th6.csv");
          }
          int l_prev = 0;
          for(int l=0;l<efficient[i][g].layers.size();l++){
            efficient[i][g].layers[l].l_api = 0;
            efficient[i][g].layers[l].l_mean = 0.0;
            efficient[i][g].layers[l].l_mem = 0.0;

            for(int w=0;w<indexkernel.size();w++){
              if(l==(int)indexkernel[w][0]){
                efficient[i][g].layers[l].l_mean = indexkernel[w][1];
                efficient[i][g].layers[l].l_api = (int)indexkernel[w][2];
                efficient[i][g].layers[l].l_mem = indexkernel[w][3];
                efficient[i][g].layers[l].l_identity = indexkernel[w][5];
                efficient[i][g].layers[l].l_prev = l_prev;
                efficient[i][g].layers[l_prev].l_next = l;
                if(l == efficient[i][g].last){
                  efficient[i][g].layers[l].l_next = l; //for last
                }
                l_prev = l;
                break;
              }
            }
          }
          for(int j=0;j<WARMING;j++){
            predict_efficientnet_warming(&efficient[i][g]);
            efficient[i][g].device->all_api = 0;
          }
          efficient[i][g].input = inputs3[efficient[i][g].g_index];
          efficient[i][g].warming = true;
          multig_efficient[g] = (&(efficient[i][g])); //하나의 net_index 0,1,2,3 

        }
      }
    }
    stream_index_H+=1;
    net_input_efficient[i]=multig_efficient;
  }

  std::cout<<"\n==================WARM UP END==================\n";


  for(int g=0;g<gpu_n;g++){
    gpu_list[gpu_idx[g]].load = 0;
    gpu_list[gpu_idx[g]].all_api = 0;
  }
  //cudaDeviceSynchronize();
  cudaProfilerStart();

  cudaEvent_t t_start, t_end;
  float t_time;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);

  cudaEventRecord(t_start);


  for(int i=0;i<n_dense;i++){
    if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))predict_densenet, &(net_input_dense[i])) < 0){
      perror("thread error");
      exit(0);
    }
    
    // #if CPU_PINNING

    //   #if CORE4
    //   // std::cout<<cpu_list[net_input_dense[i][0]->index_n%(cpu_list.size())]<<std::endl;
    //   CPU_SET(cpu_list[net_input_dense[i][0]->index_n%(cpu_list.size())], &cpuset);
    //   #else
    //   CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
    //   #endif

		// 	pthread_setaffinity_np(networkArray_dense[i], sizeof(cpu_set_t), &cpuset);
    //   n_cpu--;
    // #endif
  } 

  for(int i=0;i<n_res;i++){
    if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))predict_resnet, &(net_input_res[i])) < 0){
      perror("thread error");
      exit(0);
    }
    // #if CPU_PINNING
    //   #if CORE4
    //   CPU_SET(cpu_list[net_input_res[i][0]->index_n%(cpu_list.size())], &cpuset);
    //   #else
    //   CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
    //   #endif
		// 	pthread_setaffinity_np(networkArray_res[i], sizeof(cpu_set_t), &cpuset);
    //   n_cpu--;
    // #endif
  }

  for(int i=0;i<n_alex;i++){
    if (pthread_create(&networkArray_alex[i], NULL, (void *(*)(void*))predict_alexnet, &net_input_alex[i]) < 0){
      perror("thread error");
      exit(0);
    }
    // #if CPU_PINNING
    //   CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
    //   CPU_SET(cpu_list[i%(cpu_list.size())], &cpuset);
		// 	pthread_setaffinity_np(networkArray_alex[i], sizeof(cpu_set_t), &cpuset);
    //   n_cpu--;
    // #endif
  }
  // for(int i=0;i<n_vgg;i++){
	//   if (pthread_create(&networkArray_vgg[i], NULL, (void *(*)(void*))predict_vgg, &net_input_vgg[i]) < 0){
  //     perror("thread error");
  //     exit(0);
  //   }
  //   #if CPU_PINNING
  //     CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
	// 		pthread_setaffinity_np(networkArray_vgg[i], sizeof(cpu_set_t), &cpuset);
  //     n_cpu--;
  //   #endif
  // }
  for(int i=0;i<n_inception;i++){
	  if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))predict_inception, &net_input_inception[i]) < 0){
      perror("thread error");
      exit(0);
    }
    // #if CPU_PINNING
    //   #if CORE4
    //   CPU_SET(cpu_list[net_input_inception[i][0]->index_n%(cpu_list.size())], &cpuset);
    //   #else
    //   CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
    //   #endif
		// 	pthread_setaffinity_np(networkArray_inception[i], sizeof(cpu_set_t), &cpuset);
    //   n_cpu--;
    // #endif
  }
  for(int i=0;i<n_efficient;i++){
	  if (pthread_create(&networkArray_efficient[i], NULL, (void *(*)(void*))predict_efficientnet, &net_input_efficient[i]) < 0){
      perror("thread error");
      exit(0);
    }
    // #if CPU_PINNING
    //   #if CORE4
    //   CPU_SET(cpu_list[net_input_efficient[i][0]->index_n%(cpu_list.size())], &cpuset);
    //   #else
    //   CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
    //   #endif
		// 	pthread_setaffinity_np(networkArray_efficient[i], sizeof(cpu_set_t), &cpuset);
    //   n_cpu--;
    // #endif
  }

  
  for (int i = 0; i < n_dense; i++){
    pthread_join(networkArray_dense[i], NULL);
  }
  for (int i = 0; i < n_res; i++){
    pthread_join(networkArray_res[i], NULL);
  }
  for (int i = 0; i < n_alex; i++){
    pthread_join(networkArray_alex[i], NULL);
  }
  for (int i = 0; i < n_inception; i++){
    pthread_join(networkArray_inception[i], NULL);
  }
  for (int i = 0; i < n_efficient; i++){
    pthread_join(networkArray_efficient[i], NULL);
  }


  cudaDeviceSynchronize();
  cudaEventRecord(t_end);
  cudaEventSynchronize(t_end);
  cudaEventElapsedTime(&t_time, t_start, t_end);

	std::cout << "\n***** TOTAL EXECUTION TIME : "<<t_time/1000<<"s ***** \n";
  cudaProfilerStop();

  fclose(fp_res);
  free(cond_t);
  free(mutex_t);
  free(mutex_g);
  free(cond_i);
}
