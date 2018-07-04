#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"
namespace caffe {

	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalLocationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		joint_num_ = this->layer_param_.deep_hand_model_solve_scale_normalized_global_location_param().joint_num();
		
	}

	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalLocationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = joint_num_ * 3;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalLocationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* joint_2d_data = bottom[0]->cpu_data();
		const Dtype* scale_norm_zkr = bottom[1]->cpu_data();
		const Dtype* fx_data = bottom[2]->cpu_data(); //focus x
		const Dtype* fy_data = bottom[3]->cpu_data(); //focus y
		const Dtype* u0_data = bottom[4]->cpu_data(); //u0 offset
		const Dtype* v0_data = bottom[5]->cpu_data(); //v0 offset
		const Dtype* scale_norm_zroot_data = bottom[6]->cpu_data(); //scale normed z root

		Dtype* top_data = top[0]->mutable_cpu_data();

		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++) {
			//camera parameter 3x3 matrix fx fy u0 v0 for each sample (read from file)
			double fx = fx_data[t];
			double fy = fy_data[t];
			double u0 = u0_data[t];
			double v0 = v0_data[t];

			double scale_norm_zroot = scale_norm_zroot_data[t];

			//maybe we should use joint_num_ instead of JointNum ?
			for (int i = 0; i < joint_num_; i++) 
			{
				int Bid = t * joint_num_ * 2;
				int Zid = t * joint_num_;
				//already raw projection data
				double uk = joint_2d_data[Bid + i * 2];
				double vk = joint_2d_data[Bid + i * 2 + 1];
				
				int Tid = t * joint_num_ * 3;
				
				/*scale_norm_zk[j] = scale_norm_zroot + scale_norm_zkr[j];
				scale_norm_xk[j] = (uk[j] - u0) * scale_norm_zk[j] / fx;
				scale_norm_yk[j] = (vk[j] - v0) * scale_norm_zk[j] / fy;*/

				//add scale norm z root to get scale norm absolute z
				top_data[Tid + i * 3 + 2] = scale_norm_zroot + scale_norm_zkr[Zid + i];
				top_data[Tid + i * 3 + 0] = (uk - u0) * top_data[Tid + i * 3 + 2] / fx;
				top_data[Tid + i * 3 + 1] = (vk - v0) * top_data[Tid + i * 3 + 2] / fy;

			}

		}
	}


	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalLocationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const int batSize = (bottom[0]->shape())[0];

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelSolveScaleNormalizedGlobalLocationLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelSolveScaleNormalizedGlobalLocationLayer);
	REGISTER_LAYER_CLASS(DeepHandModelSolveScaleNormalizedGlobalLocation);
}  // namespace caffe
