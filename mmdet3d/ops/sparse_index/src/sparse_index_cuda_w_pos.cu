#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <c10/util/Half.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

//__device__ __managed__ float *sparse_tensor_ptr[10];

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T_int>
__global__ void coor_point_to_window_idx_kernel(const T_int* coor,
                                         T_int* point_to_window_idx,
                                         //T_int* point_to_pointidx,
                                         const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if ((index >= num_points) || (coor_offset[0] == -1)) return;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      //if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
      }
    }
    point_to_window_idx[index] = num;
  }
}

template <typename T_int>
__global__ void point_to_window_idx_kernel(const T_int* coor,
                                         T_int* point_to_window_idx,
                                         //T_int* point_to_pointidx,
                                         const int num_points) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    int coor_index= coor[index];
    //if (index >= num_points) return;

    int num = 0;
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      int prev_coor = coor[i];
      //if (prev_coor == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if (prev_coor==coor_index)
        num++;
    }
    point_to_window_idx[index] = num;
  }
}
template <typename T, typename T_int>
__global__ void map_tensor_to_voxel_kernel(const int nthreads,
                                      T* points,
                                      T_int* point_to_window_idx,
                                      T_int* window_regions,
                                      T_int* window_to_tensor_idx,
                                      T_int* window_indexes,
                                      T** tensor_ptr,
                                      T_int* tensor_lengthes,
                                      const int num_features,
                                      const int num_points, const int NDim,const int tensor_num) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
  int index = thread_idx / num_features;
    int in_window_pos = point_to_window_idx[index];
    int window_index = window_indexes[index];
    int tensor_index = window_regions[window_index];
    int in_tensor_pos = window_to_tensor_idx[window_index];
    int tensor_length = tensor_lengthes[tensor_index];
    auto tensor_offset=
        tensor_ptr[tensor_index]+in_tensor_pos * tensor_length * num_features+in_window_pos*num_features;
    int k = thread_idx % num_features;
    points[thread_idx]= tensor_offset[k];
  }
}

template <typename T, typename T_int,typename T_bool>
__global__ void assign_voxel_to_sparse_tensor_kernel(const int nthreads,
                                      const T_int* coors,
                                      const T* points,
                                      T_int* point_to_window_idx,
                                      T_int* window_regions,
                                      T_int* window_to_tensor_idx,
                                      T_int* window_indexes,
                                      T** tensor_ptr,
                                      T_bool** tensor_index_ptr,
                                      T_int** points_index_ptr,
                                      T_int** relative_pos_ptr,
                                      T_int* tensor_lengthes,
                                      const int num_features,
                                      const int num_points, const int NDim,const int tensor_num) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;
    int in_window_pos = point_to_window_idx[index];
    int window_index = window_indexes[index];
    int tensor_index = window_regions[window_index];
    int in_tensor_pos = window_to_tensor_idx[window_index];
    int tensor_length = tensor_lengthes[tensor_index];
    //printf("%d %d %d %d %d\n",index,in_window_pos,window_index,tensor_index,tensor_length);
    //printf("%p %p %p\n",tensor_ptr,*(tensor_ptr+tensor_index),tensor_ptr[tensor_index]); //tensor_ptr+sizeof(float*)*tensor_index);
    //if(in_window_pos>-1){
    auto tensor_offset=
        tensor_ptr[tensor_index]+in_tensor_pos * tensor_length * num_features+in_window_pos*num_features;
    int k = thread_idx % num_features;
    tensor_offset[k] = points[thread_idx];

    auto tensor_index_offset=
        tensor_index_ptr[tensor_index]+in_tensor_pos * tensor_length;
    tensor_index_offset[in_window_pos]=true;

    auto points_index_offset=
        points_index_ptr[tensor_index]+in_tensor_pos * tensor_length;
    points_index_offset[in_window_pos]=index;

    auto relative_pos_offset=
        relative_pos_ptr[tensor_index]+in_tensor_pos * tensor_length*3+in_window_pos*3;
    auto coor_offset=coors+index*3;
    relative_pos_offset[0]=coor_offset[0];
    relative_pos_offset[1]=coor_offset[1];
    relative_pos_offset[2]=coor_offset[2];
    //}
    //auto tensor_offset=tensor_ptr[tensor_index]+
  }
}


template <typename T, typename T_int,typename T_bool>
__global__ void assign_voxel_to_sparse_tensor_kernel_wo_pos(const int nthreads,
                                      const T* points,
                                      T_int* point_to_window_idx,
                                      T_int* window_regions,
                                      T_int* window_to_tensor_idx,
                                      T_int* window_indexes,
                                      T** tensor_ptr,
                                      T_int* tensor_lengthes,
                                      const int num_features,
                                      const int num_points, const int NDim,const int tensor_num) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;
    int in_window_pos = point_to_window_idx[index];
    int window_index = window_indexes[index];
    int tensor_index = window_regions[window_index];
    int in_tensor_pos = window_to_tensor_idx[window_index];
    int tensor_length = tensor_lengthes[tensor_index];
    //printf("%d %d %d %d %d\n",index,in_window_pos,window_index,tensor_index,tensor_length);
    //printf("%p %p %p\n",tensor_ptr,*(tensor_ptr+tensor_index),tensor_ptr[tensor_index]); //tensor_ptr+sizeof(float*)*tensor_index);
    //if(in_window_pos>-1){
    auto tensor_offset=
        tensor_ptr[tensor_index]+in_tensor_pos * tensor_length * num_features+in_window_pos*num_features;
    int k = thread_idx % num_features;
    tensor_offset[k] = points[thread_idx];


    //}
    //auto tensor_offset=tensor_ptr[tensor_index]+
  }
}

template <typename T,typename T_int>
__global__ void assign_voxel_to_sparse_tensor_kernel_wo_pos_half(const int nthreads,
                                      const T* points,
                                      T_int* point_to_window_idx,
                                      T_int* window_regions,
                                      T_int* window_to_tensor_idx,
                                      T_int* window_indexes,
                                      at::Half** tensor_ptr,
                                      T_int* tensor_lengthes,
                                      const int num_features,
                                      const int num_points, const int NDim,const int tensor_num) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;
    int in_window_pos = point_to_window_idx[index];
    int window_index = window_indexes[index];
    int tensor_index = window_regions[window_index];
    int in_tensor_pos = window_to_tensor_idx[window_index];
    int tensor_length = tensor_lengthes[tensor_index];
    //printf("%d %d %d %d %d\n",index,in_window_pos,window_index,tensor_index,tensor_length);
    //printf("%p %p %p\n",tensor_ptr,*(tensor_ptr+tensor_index),tensor_ptr[tensor_index]); //tensor_ptr+sizeof(float*)*tensor_index);
    //if(in_window_pos>-1){
    at::Half* tensor_offset=
        tensor_ptr[tensor_index]+in_tensor_pos * tensor_length * num_features+in_window_pos*num_features;
    int k = thread_idx % num_features;
    tensor_offset[k] = points[thread_idx];


    //}
    //auto tensor_offset=tensor_ptr[tensor_index]+
  }
}


template <typename T_int,typename T_bool>
__global__ void assign_voxel_info_to_sparse_tensor_kernel(const int nthreads,
                                      const T_int* coors,
                                      T_int* point_to_window_idx,
                                      T_int* window_regions,
                                      T_int* window_to_tensor_idx,
                                      T_int* window_indexes,
                                      T_bool** tensor_index_ptr,
                                      T_int** points_index_ptr,
                                      T_int** pos_ptr,
                                      T_int* tensor_lengthes,
                                      const int num_points,const int tensor_num) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int in_window_pos = point_to_window_idx[thread_idx];
    int window_index = window_indexes[thread_idx];
    int tensor_index = window_regions[window_index];
    int in_tensor_pos = window_to_tensor_idx[window_index];
    int tensor_length = tensor_lengthes[tensor_index];

    auto tensor_index_offset=
        tensor_index_ptr[tensor_index]+in_tensor_pos * tensor_length;
    tensor_index_offset[in_window_pos]=true;

    auto points_index_offset=
        points_index_ptr[tensor_index]+in_tensor_pos * tensor_length;
    points_index_offset[in_window_pos]=thread_idx;
    auto pos_offset=
        pos_ptr[tensor_index]+in_tensor_pos * tensor_length*3 + in_window_pos*3;
    auto coor_offset = coors+thread_idx*3;
    pos_offset[0] = coor_offset[0];
    pos_offset[1] = coor_offset[1];
    pos_offset[2] = coor_offset[2];

    //}
    //auto tensor_offset=tensor_ptr[tensor_index]+
  }
}




namespace sparse_index{

//void sparse_index_backward_gpu(
//  at::Tensor & grad_voxel,
//  std::vector<at::Tensor> & grad_tensor_list,
//  const at::Tensor & point_to_window_idx,
//  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
//  const at::Tensor & tensor_lengthes,const int tensor_num=10){
//  CHECK_INPUT(grad_voxel);
//  at::cuda::CUDAGuard device_guard(grad_voxel.device());
//  const int num_points = grad_voxel.size(0);
//  const int num_features = grad_voxel.size(1);
//  float *grad_tensor_ptr[18];
//  for(int i=0;i<tensor_num;i++)
//  {
//    grad_tensor_ptr[i]=grad_tensor_list[i].contiguous().data_ptr<float>();
//  }
//  float** d_grad_tensor_ptr;
//  cudaMalloc((void***)&d_grad_tensor_ptr, sizeof(grad_tensor_ptr));
//  cudaMemcpy(d_grad_tensor_ptr,grad_tensor_ptr, sizeof(grad_tensor_ptr), cudaMemcpyHostToDevice);
//  auto pts_output_size = num_points * num_features;
//  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
//  dim3 cp_block(512);
//  AT_DISPATCH_ALL_TYPES(
//      grad_voxel.scalar_type(), "map_tensor_to_voxel", ([&] {
//        map_tensor_to_voxel_kernel<float, int>
//            <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
//                pts_output_size,
//                grad_voxel.contiguous().data_ptr<float>(),
//                point_to_window_idx.contiguous().data_ptr<int>(),
//                window_regions.contiguous().data_ptr<int>(),
//                window_to_tensor_idx.contiguous().data_ptr<int>(),
//                window_indexes.contiguous().data_ptr<int>(),
//                d_grad_tensor_ptr,
//                tensor_lengthes.contiguous().data_ptr<int>(),
//                num_features,num_points, 3,tensor_num);
//      }));
//  cudaFree(d_grad_tensor_ptr);
//  cudaDeviceSynchronize();
//  AT_CUDA_CHECK(cudaGetLastError());
//  return;
//}

void sparse_index_with_pos_gpu(
  std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  std::vector<at::Tensor> & pos_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,
  const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  CHECK_INPUT(voxel_features);
  at::cuda::CUDAGuard device_guard(voxel_features.device());
  const int num_points = voxel_features.size(0);
  const int num_features = voxel_features.size(1);

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);
  AT_DISPATCH_ALL_TYPES(
     point_to_window_idx.scalar_type(), "determin_point_to_window_idx", ([&] {
        point_to_window_idx_kernel<int><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            window_indexes.data_ptr<int>(), point_to_window_idx.data_ptr<int>(),
            num_points);
      }));

  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  //std::vector<float*>sparse_tensor_ptr(10);
  //printf("size %d\n",sparse_tensor_list.size());
  float *sparse_tensor_ptr[18];
  bool *sparse_tensor_index_ptr[18];
  int *points_index_ptr[18];
  int *pos_ptr[18];
  for(int i=0;i<tensor_num;i++)
  {
    //printf("%d %p %p\n",i,sparse_tensor_list[i].contiguous().data_ptr<float>(),sparse_tensor_list[i].data_ptr<float>()+sparse_tensor_list[i].numel());
    //printf('%d \n',sizeof(voxel_features.contiguous().data_ptr<float>()));
    //if(i>=tensor_num)
    //{
    //  sparse_tensor_ptr[i]=NULL;
    //  continue;
    //}
    sparse_tensor_ptr[i]=sparse_tensor_list[i].contiguous().data_ptr<float>();
    sparse_tensor_index_ptr[i]=sparse_tensor_index_list[i].contiguous().data_ptr<bool>();
    points_index_ptr[i]=points_index_list[i].contiguous().data_ptr<int>();
    pos_ptr[i]=pos_list[i].contiguous().data_ptr<int>();

    //printf("%d %p\n",i,sparse_tensor_ptr[i]);


    //printf("%d \n",i);
    //printf('%d\n',sparse_tensor_ptr[i]);
  }
  // printf("test %p %p %p %p %p %p %p\n",sparse_tensor_ptr,*sparse_tensor_ptr,&sparse_tensor_ptr,&sparse_tensor_ptr[1],sparse_tensor_ptr[1],(sparse_tensor_ptr+1),*(sparse_tensor_ptr+1));
  float** d_sparse_tensor_ptr;
  bool** d_sparse_tensor_index_ptr;
  int** d_points_index_ptr;
  int** d_pos_ptr;

  cudaMalloc((void***)&d_sparse_tensor_ptr, sizeof(sparse_tensor_ptr));
  cudaMalloc((void***)&d_sparse_tensor_index_ptr, sizeof(sparse_tensor_index_ptr));
  cudaMalloc((void***)&d_points_index_ptr, sizeof(points_index_ptr));
  cudaMalloc((void***)&d_pos_ptr, sizeof(pos_ptr));


  cudaMemcpy(d_sparse_tensor_ptr,sparse_tensor_ptr, sizeof(sparse_tensor_ptr), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sparse_tensor_index_ptr,sparse_tensor_index_ptr, sizeof(sparse_tensor_index_ptr), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points_index_ptr,points_index_ptr, sizeof(points_index_ptr), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos_ptr,pos_ptr, sizeof(pos_ptr), cudaMemcpyHostToDevice);
    //return;
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      voxel_features.scalar_type(), "assign_voxel_to_tensor_wo_pos", ([&] {
        assign_voxel_to_sparse_tensor_kernel_wo_pos<float, int, bool>
            <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                pts_output_size,
                voxel_features.contiguous().data_ptr<float>(),
                point_to_window_idx.contiguous().data_ptr<int>(),
                window_regions.contiguous().data_ptr<int>(),
                window_to_tensor_idx.contiguous().data_ptr<int>(),
                window_indexes.contiguous().data_ptr<int>(),
                d_sparse_tensor_ptr,
                tensor_lengthes.contiguous().data_ptr<int>(),
                num_features,num_points, NDim,tensor_num);
      }));

    dim3 coor_cp_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
    dim3 coor_cp_block(512);
    AT_DISPATCH_ALL_TYPES(
      voxel_features.scalar_type(), "assign_voxel_info_to_tensor", ([&] {
        assign_voxel_info_to_sparse_tensor_kernel<int, bool>
            <<<coor_cp_grid, coor_cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_points,
                coors.contiguous().data_ptr<int>(),
                point_to_window_idx.contiguous().data_ptr<int>(),
                window_regions.contiguous().data_ptr<int>(),
                window_to_tensor_idx.contiguous().data_ptr<int>(),
                window_indexes.contiguous().data_ptr<int>(),
                d_sparse_tensor_index_ptr,
                d_points_index_ptr,
                d_pos_ptr,
                tensor_lengthes.contiguous().data_ptr<int>(),
                num_points,tensor_num);
      }));
  cudaFree(d_sparse_tensor_ptr);
  cudaFree(d_sparse_tensor_index_ptr);
  cudaFree(d_points_index_ptr);
  cudaFree(d_pos_ptr);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return;
 }


void sparse_index_with_pos_gpu_half(
  std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  std::vector<at::Tensor> & pos_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,
  const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  CHECK_INPUT(voxel_features);
  at::cuda::CUDAGuard device_guard(voxel_features.device());
  const int num_points = voxel_features.size(0);
  const int num_features = voxel_features.size(1);

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);
  AT_DISPATCH_ALL_TYPES(
     point_to_window_idx.scalar_type(), "determin_point_to_window_idx", ([&] {
        point_to_window_idx_kernel<int><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            window_indexes.data_ptr<int>(), point_to_window_idx.data_ptr<int>(),
            num_points);
      }));

  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  //std::vector<float*>sparse_tensor_ptr(10);
  //printf("size %d\n",sparse_tensor_list.size());
  at::Half *sparse_tensor_ptr[18];
  bool *sparse_tensor_index_ptr[18];
  int *points_index_ptr[18];
  int *pos_ptr[18];
  for(int i=0;i<tensor_num;i++)
  {
    //printf("%d %p %p\n",i,sparse_tensor_list[i].contiguous().data_ptr<float>(),sparse_tensor_list[i].data_ptr<float>()+sparse_tensor_list[i].numel());
    //printf('%d \n',sizeof(voxel_features.contiguous().data_ptr<float>()));
    //if(i>=tensor_num)
    //{
    //  sparse_tensor_ptr[i]=NULL;
    //  continue;
    //}
    sparse_tensor_ptr[i]=sparse_tensor_list[i].contiguous().data_ptr<at::Half>();
    sparse_tensor_index_ptr[i]=sparse_tensor_index_list[i].contiguous().data_ptr<bool>();
    points_index_ptr[i]=points_index_list[i].contiguous().data_ptr<int>();
    pos_ptr[i]=pos_list[i].contiguous().data_ptr<int>();

    //printf("%d %p\n",i,sparse_tensor_ptr[i]);


    //printf("%d \n",i);
    //printf('%d\n',sparse_tensor_ptr[i]);
  }
  // printf("test %p %p %p %p %p %p %p\n",sparse_tensor_ptr,*sparse_tensor_ptr,&sparse_tensor_ptr,&sparse_tensor_ptr[1],sparse_tensor_ptr[1],(sparse_tensor_ptr+1),*(sparse_tensor_ptr+1));
  at::Half** d_sparse_tensor_ptr;
  bool** d_sparse_tensor_index_ptr;
  int** d_points_index_ptr;
  int** d_pos_ptr;

  cudaMalloc((void***)&d_sparse_tensor_ptr, sizeof(sparse_tensor_ptr));
  cudaMalloc((void***)&d_sparse_tensor_index_ptr, sizeof(sparse_tensor_index_ptr));
  cudaMalloc((void***)&d_points_index_ptr, sizeof(points_index_ptr));
  cudaMalloc((void***)&d_pos_ptr, sizeof(pos_ptr));


  cudaMemcpy(d_sparse_tensor_ptr,sparse_tensor_ptr, sizeof(sparse_tensor_ptr), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sparse_tensor_index_ptr,sparse_tensor_index_ptr, sizeof(sparse_tensor_index_ptr), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points_index_ptr,points_index_ptr, sizeof(points_index_ptr), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos_ptr,pos_ptr, sizeof(pos_ptr), cudaMemcpyHostToDevice);
    //return;
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Half,voxel_features.scalar_type(), "assign_voxel_to_tensor_wo_pos_half", ([&] {
        assign_voxel_to_sparse_tensor_kernel_wo_pos_half<scalar_t,int>
            <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                pts_output_size,
                voxel_features.contiguous().data_ptr<scalar_t>(),
                point_to_window_idx.contiguous().data_ptr<int>(),
                window_regions.contiguous().data_ptr<int>(),
                window_to_tensor_idx.contiguous().data_ptr<int>(),
                window_indexes.contiguous().data_ptr<int>(),
                d_sparse_tensor_ptr,
                tensor_lengthes.contiguous().data_ptr<int>(),
                num_features,num_points, NDim,tensor_num);
      }));

    dim3 coor_cp_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
    dim3 coor_cp_block(512);
    AT_DISPATCH_ALL_TYPES(
      coors.scalar_type(), "assign_voxel_info_to_tensor", ([&] {
        assign_voxel_info_to_sparse_tensor_kernel<int, bool>
            <<<coor_cp_grid, coor_cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_points,
                coors.contiguous().data_ptr<int>(),
                point_to_window_idx.contiguous().data_ptr<int>(),
                window_regions.contiguous().data_ptr<int>(),
                window_to_tensor_idx.contiguous().data_ptr<int>(),
                window_indexes.contiguous().data_ptr<int>(),
                d_sparse_tensor_index_ptr,
                d_points_index_ptr,
                d_pos_ptr,
                tensor_lengthes.contiguous().data_ptr<int>(),
                num_points,tensor_num);
      }));
  cudaFree(d_sparse_tensor_ptr);
  cudaFree(d_sparse_tensor_index_ptr);
  cudaFree(d_points_index_ptr);
  cudaFree(d_pos_ptr);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return;
 }
}

