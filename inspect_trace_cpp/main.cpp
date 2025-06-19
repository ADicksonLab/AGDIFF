#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#include <torchcluster/cluster.h>
#include <iostream>

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: hello-world <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(argv[1]);
    model.eval();
    std::cout << "Model loaded successfully!\n";
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }


  ////////////////////////////////////////////////////////////////////////////////////
  // Defining inputs (original)
auto pos_sample = torch::tensor({
    {-6.4286,  -2.2195,   1.2546},
    { 7.9375, -14.9490,  13.7540},
    {-1.3787,  16.4749,  11.7864},
    {31.4859,  -5.7569,  27.5555},
    {10.9687,   8.5556,   2.0881},
    {-16.5893,  -0.9832,  -5.6218},
    {-8.0981, -17.4447,  12.3680},
    {18.0933,  -4.5255,  -4.8151},
    { 3.5313, -17.6224,  -4.5693},
    {-4.4451,  13.3744, -10.3213},
    {-8.2149, -13.4401, -12.5153},
    {-0.3298,  17.3766,   3.6481},
    {-18.3561, -21.5394,  14.2432},
    {-7.7890, -15.8494,  -0.2349},
    {11.0490,  -4.6739,   0.5792},
    { 0.3784,   6.5527,  22.9655},
    {21.7426,  -0.7690,   3.8099},
    { 8.6980, -15.8056, -21.5284},
    {-8.5690,  -1.7754, -21.2267},
    { 6.1840,  11.3447,   2.3838},
    {-8.4851,   8.0373,   5.8482},
    {-18.1468,  -2.8499,  -4.6861}
}).to(torch::kCPU);


  auto t_sample = torch::tensor({0.5}).to(torch::kCPU); 

  // auto sigma_sample = torch::rand({5000}).to(torch::kCPU); 

  auto atom_type_sample = torch::tensor({1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1 }).to(torch::kCPU); 

  auto edge_index_sample = torch::tensor({
                                      { 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,
                                2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
                                4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,
                                6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,
                                8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
                                9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
                                11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
                                13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15,
                                15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                                17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19,
                                19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21},
                                {1,  2,  3,  4,  5,  6,  0,  2,  3,  4,  5,  6,  7,  8,  0,  1,  3,  4,
                                5,  6,  0,  1,  2,  4,  5,  6,  0,  1,  2,  3,  5,  6,  7,  8,  9, 10,
                                14,  0,  1,  2,  3,  4,  6,  7,  8,  0,  1,  2,  3,  4,  5,  7,  8,  9,
                                10, 11, 12, 13, 14, 15, 16,  1,  4,  5,  6,  8,  9, 10, 14,  1,  4,  5,
                                6,  7,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  4,  6,  7,  8, 10, 11,
                                12, 13, 14, 15, 16,  4,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16,  6,  8,
                                9, 10, 12, 13, 14,  6,  8,  9, 10, 11, 13, 14,  6,  8,  9, 10, 11, 12,
                                14,  4,  6,  7,  8,  9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,  6,
                                8,  9, 10, 14, 16, 17, 18,  6,  8,  9, 10, 14, 15, 17, 18, 19, 20, 21,
                                8, 14, 15, 16, 18, 19, 20, 21,  8, 14, 15, 16, 17, 19, 20, 21, 14, 16,
                                17, 18, 20, 21, 14, 16, 17, 18, 19, 21, 14, 16, 17, 18, 19, 20 }
                                      }).to(torch::kCPU); 

  auto edge_type_sample = torch::tensor({ 1, 23, 23, 23, 24, 24,  1,  1,  1,  1, 23, 23, 24, 24, 23,  1, 23, 23,
        24, 24, 23,  1, 23, 23, 24, 24, 23,  1, 23, 23,  2,  1, 23, 23, 24, 24,
        24, 24, 23, 24, 24,  2, 23, 24, 24, 24, 23, 24, 24,  1, 23,  1,  1, 23,
        23, 24, 24, 24, 23, 24, 24, 24, 23, 24,  1, 23, 24, 24, 24, 24, 23, 24,
        1, 23,  1,  1, 23, 23, 23,  1, 23, 23, 24, 24, 24, 23, 24,  1, 23, 24,
        24, 24, 23, 24, 24, 24, 23, 24,  1, 23,  1,  1,  1, 23, 24, 24, 24, 23,
        24,  1, 23, 23, 24, 24, 23, 24,  1, 23, 23, 24, 24, 23, 24,  1, 23, 23,
        24, 24, 23, 24,  1, 23, 23, 24, 24, 24,  2,  1, 23, 23, 24, 24, 24, 24,
        23, 24, 24,  2, 23, 24, 24, 24, 23, 24, 24,  1, 23,  1,  1, 23, 23, 23,
        24, 23, 24,  1, 23, 24, 24, 24, 24, 23, 24,  1, 23,  1,  1,  1, 24, 23,
        24,  1, 23, 23, 24, 23, 24,  1, 23, 23, 24, 23, 24,  1, 23, 23}).to(torch::kCPU); 

  auto batch_sample = torch::zeros_like(atom_type_sample, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));

  auto include_global_sample = torch::tensor({0}).to(torch::kCPU); 



  std::cout <<"pos_sample shape:" << pos_sample.sizes() << std::endl;
  std::cout <<"t_sample shape:" << t_sample.sizes() << std::endl;
  std::cout <<"atom_type_sample shape:" << atom_type_sample.sizes() << std::endl;
  std::cout <<"edge_index_sample shape:" << edge_index_sample.sizes() << std::endl;
  std::cout <<"edge_type_sample shape:" << edge_type_sample.sizes() << std::endl;
  std::cout <<"batch_sample shape:" << batch_sample.sizes() << std::endl;
  std::cout <<"include_global_sample" << include_global_sample.sizes() << std::endl;

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(pos_sample) ;
  inputs.push_back(t_sample) ;
  inputs.push_back(atom_type_sample) ;
  inputs.push_back(edge_index_sample) ;
  inputs.push_back(edge_type_sample) ;
  inputs.push_back(batch_sample) ;
  inputs.push_back(include_global_sample);

  auto get_diffusion_noise = model.get_method("get_diffusion_noise");
  auto out = get_diffusion_noise(inputs).toTensor();
  
  std::cout <<"output tensor shape:" << out.sizes() << std::endl;
  std::cout <<"output:" << out << std::endl;


  ////////////////////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////////
//   // Defining second inputs 

//   auto pos_sample2 = torch::tensor({
//                                   {-3.1595, -0.0915,  1.1380},
//                                   {-2.6371,  0.1743, -0.2724},
//                                   {-1.7372, -0.8189, -0.7360},
//                                   {-0.4360, -0.7348, -0.2102},
//                                   { 0.3126,  0.4802, -0.7309},
//                                   { 1.6709,  0.8246, -0.2945},
//                                   { 2.4149, -0.0089,  0.7262},
//                                   { 2.8783, -1.1726,  0.0769},
//                                   { 3.3983, -1.6934,  0.6980},
//                                   { 1.7422, -0.2596,  1.5567},
//                                   { 3.2553,  0.5776,  1.1240},
//                                   { 0.5610,  1.5577,  0.1323},
//                                   { 2.3141,  1.3369, -1.0058},
//                                   { 0.0204,  0.7628, -1.7406},
//                                   {-0.4508, -0.7196,  0.8865},
//                                   { 0.0746, -1.6398, -0.5507},
//                                   {-3.4660,  0.1442, -0.9843},
//                                   {-2.1684,  1.1652, -0.3160},
//                                   {-3.5535, -1.1028,  1.2032},
//                                   {-2.3716,  0.0280,  1.8777},
//                                   {-3.9530,  0.6140,  1.3681}
//                               }).to(torch::kCPU);



//   auto t_sample2 = torch::tensor({0.1}).to(torch::kCPU);


//   // auto sigma_sample2 = torch::rand({2000}).to(torch::kCPU);



//   auto atom_type_sample2 = torch::tensor({6, 6, 8, 6, 6, 6, 6, 8, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1}).to(torch::kCPU);

//   auto edge_index_sample2 = torch::tensor({
//                                           { 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,
//                                             5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  8,  9, 10, 11, 11, 12, 13, 14,
//                                           15, 16, 17, 18, 19, 20},
//                                           { 1, 18, 19, 20,  0,  2, 16, 17,  1,  3,  2,  4, 14, 15,  3,  5, 11, 13,
//                                             4,  6, 11, 12,  5,  7,  9, 10,  6,  8,  7,  6,  6,  4,  5,  5,  4,  3,
//                                             3,  1,  1,  0,  0,  0}
//                                       }).to(torch::kCPU);

  

//   auto edge_type_sample2 = torch::tensor({
//                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
//                                     }).to(torch::kCPU);


//   auto batch_sample2 = torch::tensor({
//                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
//                                 }).to(torch::kCPU);



  


//   std::cout <<"pos_sample2 shape:" << pos_sample2.sizes() << std::endl;
//   std::cout <<"t_sample2 shape:" << t_sample2.sizes() << std::endl;
//   // std::cout <<"sigma_sample2 shape:" << sigma_sample2.sizes() << std::endl;
//   std::cout <<"atom_type_sample2 shape:" << atom_type_sample2.sizes() << std::endl;
//   std::cout <<"edge_index_sample2 shape:" << edge_index_sample2.sizes() << std::endl;
//   std::cout <<"edge_type_sample2 shape:" << edge_type_sample2.sizes() << std::endl;
//   std::cout <<"batch_sample2 shape:" << batch_sample2.sizes() << std::endl;

//   std::vector<torch::jit::IValue> inputs2;
//   inputs2.push_back(pos_sample2) ;
//   inputs2.push_back(t_sample2) ;
//   // inputs2.push_back(sigma_sample2) ;
//   inputs2.push_back(atom_type_sample2) ;
//   inputs2.push_back(edge_index_sample2) ;
//   inputs2.push_back(edge_type_sample2) ;
//   inputs2.push_back(batch_sample2) ;

//   ////////////////////////////////////////////////////////////////////////////////////

//   auto get_diffusion_noise = model.get_method("get_diffusion_noise");
//   auto out = get_diffusion_noise(inputs).toTensor();
//   auto out2 = get_diffusion_noise(inputs2).toTensor();

//   std::cout <<"output tensor shape:" << out.sizes() << std::endl;
//   std::cout <<"output:" << out << std::endl;

//   std::cout <<"output2 tensor shape:" << out2.sizes() << std::endl;
//   std::cout <<"output2:" << out2 << std::endl;

}
