#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"
namespace caffe {

	template <typename Dtype>
	void DeepHandModelSolveGlobalHandScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
	}

	template <typename Dtype>
	void DeepHandModelSolveGlobalHandScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = 1;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelSolveGlobalHandScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* scale_norm_global_location_data = bottom[0]->cpu_data();
		const Dtype* stat_avg_bone_len_data = bottom[1]->cpu_data();
		
		Dtype* top_data = top[0]->mutable_cpu_data();

		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++) 
		{
			
			
			int Bid = t * JointNum_RHD * 3;
			int Sid = t * BoneNum_RHD;
			//solving scale
			double sum_of_square_bone_len = 0.0;
			for (int j = 0; j < BoneNum_RHD; j++) 
			{
				int u = bones_RHD[j][0], v = bones_RHD[j][1];
				double scale_norm_xku = scale_norm_global_location_data[Bid + u * 3];
				double scale_norm_yku = scale_norm_global_location_data[Bid + u * 3 + 1];
				double scale_norm_zku = scale_norm_global_location_data[Bid + u * 3 + 2];

				double scale_norm_xkv = scale_norm_global_location_data[Bid + v * 3];
				double scale_norm_ykv = scale_norm_global_location_data[Bid + v * 3 + 1];
				double scale_norm_zkv = scale_norm_global_location_data[Bid + v * 3 + 2];


				sum_of_square_bone_len += pow(scale_norm_xku - scale_norm_xkv, 2) + pow(scale_norm_yku - scale_norm_ykv, 2) + pow(scale_norm_zku - scale_norm_zkv, 2);
			}

			double solve_scale_argmin_equation_A = sum_of_square_bone_len;

			double solve_scale_argmin_equation_B = 0.0;
			for (int j = 0; j < BoneNum_RHD; j++) 
			{
				int u = bones_RHD[j][0], v = bones_RHD[j][1];
				double scale_norm_xku = scale_norm_global_location_data[Bid + u * 3];
				double scale_norm_yku = scale_norm_global_location_data[Bid + u * 3 + 1];
				double scale_norm_zku = scale_norm_global_location_data[Bid + u * 3 + 2];

				double scale_norm_xkv = scale_norm_global_location_data[Bid + v * 3];
				double scale_norm_ykv = scale_norm_global_location_data[Bid + v * 3 + 1];
				double scale_norm_zkv = scale_norm_global_location_data[Bid + v * 3 + 2];


				solve_scale_argmin_equation_B += sqrt(pow(scale_norm_xku - scale_norm_xkv, 2) + pow(scale_norm_yku - scale_norm_ykv, 2) + pow(scale_norm_zku - scale_norm_zkv, 2)) * stat_avg_bone_len_data[Sid + j];
			}
			solve_scale_argmin_equation_B *= -2.0;

			double solve_scale_argmin_equation_C = 0.0;
			for (int j = 0; j < BoneNum_RHD; j++) 
			{
				solve_scale_argmin_equation_C += pow(stat_avg_bone_len_data[Sid + j], 2);
			}

			//get min nadir to solve quadratic equation


			//-b+-sqrt(b*b-4*a*c) /2a
			//min: might not equal to zero
			//so is b/-2a
			double global_hand_scale = solve_scale_argmin_equation_B / (-2 * solve_scale_argmin_equation_A);
			
			int Tid = t;
			top_data[Tid] = global_hand_scale;
		}
	}


	template <typename Dtype>
	void DeepHandModelSolveGlobalHandScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const int batSize = (bottom[0]->shape())[0];

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelSolveGlobalHandScaleLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelSolveGlobalHandScaleLayer);
	REGISTER_LAYER_CLASS(DeepHandModelSolveGlobalHandScale);
}  // namespace caffe
