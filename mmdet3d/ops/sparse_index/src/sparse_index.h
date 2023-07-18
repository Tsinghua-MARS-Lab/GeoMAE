#pragma once
#include <torch/extension.h>




namespace sparse_index {


void sparse_index_gpu(
  std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  std::vector<at::Tensor> & relative_pos_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10);



inline void sparse_index_test(std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  std::vector<at::Tensor> & relative_pos_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  if (coors.device().is_cuda()) {

    sparse_index_gpu(sparse_tensor_list,
                     sparse_tensor_index_list,
                     points_index_list,
                     relative_pos_list,
                     point_to_window_idx,
                     coors,voxel_features,
                     window_indexes,window_regions,window_to_tensor_idx,
                     tensor_lengthes,NDim,tensor_num);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
}
}

void sparse_index_backward_gpu(
  at::Tensor & grad_voxel,
  std::vector<at::Tensor> & grad_tensor_list,
  const at::Tensor & point_to_window_idx,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int tensor_num);



inline void sparse_index_backward_test(
  at::Tensor & grad_voxel,
  std::vector<at::Tensor> & grad_tensor_list,
  const at::Tensor & point_to_window_idx,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int tensor_num=10
  ){
  if (grad_voxel.device().is_cuda()) {

    sparse_index_backward_gpu(
      grad_voxel,
      grad_tensor_list,
      point_to_window_idx,
      window_indexes,window_regions,window_to_tensor_idx,
      tensor_lengthes,tensor_num);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
}
}


void sparse_index_backward_gpu_half(
  at::Tensor & grad_voxel,
  std::vector<at::Tensor> & grad_tensor_list,
  const at::Tensor & point_to_window_idx,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int tensor_num);



inline void sparse_index_backward_half(
  at::Tensor & grad_voxel,
  std::vector<at::Tensor> & grad_tensor_list,
  const at::Tensor & point_to_window_idx,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int tensor_num=10
  ){
  if (grad_voxel.device().is_cuda()) {

    sparse_index_backward_gpu_half(
      grad_voxel,
      grad_tensor_list,
      point_to_window_idx,
      window_indexes,window_regions,window_to_tensor_idx,
      tensor_lengthes,tensor_num);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
}
}


void sparse_index_gpu_(
  std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10);


inline void sparse_index_test_(std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  if (coors.device().is_cuda()) {

    sparse_index_gpu_(sparse_tensor_list,
                     sparse_tensor_index_list,
                     points_index_list,
                     point_to_window_idx,
                     coors,voxel_features,
                     window_indexes,window_regions,window_to_tensor_idx,
                     tensor_lengthes,NDim,tensor_num);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
}
}


void sparse_index_wo_pos_gpu(
  std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10);



inline void sparse_index_wo_pos(std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  if (voxel_features.device().is_cuda()) {

    sparse_index_wo_pos_gpu(sparse_tensor_list,
                     sparse_tensor_index_list,
                     points_index_list,
                     point_to_window_idx,
                     voxel_features,
                     window_indexes,window_regions,window_to_tensor_idx,
                     tensor_lengthes,NDim,tensor_num);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
}
}

void sparse_index_with_pos_gpu(
  std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  std::vector<at::Tensor> & pos_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,
  const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10);



inline void sparse_index_with_pos(std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  std::vector<at::Tensor> & pos_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,
  const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  if (voxel_features.device().is_cuda()) {

    sparse_index_with_pos_gpu(sparse_tensor_list,
                     sparse_tensor_index_list,
                     points_index_list,
                     pos_list,
                     point_to_window_idx,
                     coors,
                     voxel_features,
                     window_indexes,window_regions,window_to_tensor_idx,
                     tensor_lengthes,NDim,tensor_num);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
}
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
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10);



inline void sparse_index_with_pos_half(std::vector<at::Tensor> & sparse_tensor_list,
  std::vector<at::Tensor> & sparse_tensor_index_list,
  std::vector<at::Tensor> & points_index_list,
  std::vector<at::Tensor> & pos_list,
  at::Tensor & point_to_window_idx,
  const at::Tensor & coors,
  const at::Tensor & voxel_features,
  const at::Tensor & window_indexes,const at::Tensor & window_regions,const at::Tensor & window_to_tensor_idx,
  const at::Tensor & tensor_lengthes,const int NDim=3,const int tensor_num=10){
  if (voxel_features.device().is_cuda()) {

    sparse_index_with_pos_gpu_half(sparse_tensor_list,
                     sparse_tensor_index_list,
                     points_index_list,
                     pos_list,
                     point_to_window_idx,
                     coors,
                     voxel_features,
                     window_indexes,window_regions,window_to_tensor_idx,
                     tensor_lengthes,NDim,tensor_num);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
}
}

}