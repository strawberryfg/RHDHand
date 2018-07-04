#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"
namespace caffe {

	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalZRootLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		joint_num_ = this->layer_param_.deep_hand_model_solve_scale_normalized_global_z_root_param().joint_num();
		//joint (specific pair of keypoints) n_ and m_
		n_ = this->layer_param_.deep_hand_model_solve_scale_normalized_global_z_root_param().n();
		m_ = this->layer_param_.deep_hand_model_solve_scale_normalized_global_z_root_param().m();

		//scale constant
		C_ = this->layer_param_.deep_hand_model_solve_scale_normalized_global_z_root_param().c();
	}

	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalZRootLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = 1;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalZRootLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* joint_2d_data = bottom[0]->cpu_data();
		const Dtype* scale_norm_zkr = bottom[1]->cpu_data();
		const Dtype* fx_data = bottom[2]->cpu_data(); //focus x
		const Dtype* fy_data = bottom[3]->cpu_data(); //focus y
		const Dtype* u0_data = bottom[4]->cpu_data(); //u0 offset
		const Dtype* v0_data = bottom[5]->cpu_data(); //v0 offset
		
		Dtype* top_data = top[0]->mutable_cpu_data();

		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++) {
			//camera parameter 3x3 matrix fx fy u0 v0 for each sample (read from file)
			double fx = fx_data[t];
			double fy = fy_data[t];
			double u0 = u0_data[t];
			double v0 = v0_data[t];

			//maybe we should use joint_num_ instead of JointNum ?
			//for (int i = 0; i < joint_num_; i++) 
			{
				int Bid = t * joint_num_ * 2;
				//already raw projection data
				double ukn = joint_2d_data[Bid + n_ * 2];
				double vkn = joint_2d_data[Bid + n_ * 2 + 1];
				double ukm = joint_2d_data[Bid + m_ * 2];
				double vkm = joint_2d_data[Bid + m_ * 2 + 1];


				//calc scaled z root
				double scale_norm_zroot_equation_A = pow((ukn - ukm) / fx, 2) + pow((vkn - vkm) / fy, 2);
				double scale_norm_zroot_equation_B = 2 * (((ukn - ukm) / fx) * ((ukn - u0) / fx * scale_norm_zkr[Bid + n_] - (ukm - u0) / fx * scale_norm_zkr[Bid + m_]) +
					((vkn - vkm) / fy) * ((vkn - v0) / fy * scale_norm_zkr[Bid + n_] - (vkm - v0) / fy * scale_norm_zkr[Bid + m_]));
				double scale_norm_zroot_equation_C = pow((ukn - u0) / fx * scale_norm_zkr[Bid + n_] - (ukm - u0) / fx * scale_norm_zkr[Bid + m_], 2) + pow((vkn - v0) / fy * scale_norm_zkr[Bid + n_] - (vkm - v0) / fy * scale_norm_zkr[Bid + m_], 2) + pow(scale_norm_zkr[Bid + n_] - scale_norm_zkr[Bid + m_], 2) - pow(C_, 2);

				double scale_norm_zroot_0 = 0.5 * (-scale_norm_zroot_equation_B + sqrt(pow(scale_norm_zroot_equation_B, 2) - 4 * scale_norm_zroot_equation_A * scale_norm_zroot_equation_C)) / scale_norm_zroot_equation_A;
				double scale_norm_zroot_1 = 0.5 * (-scale_norm_zroot_equation_B - sqrt(pow(scale_norm_zroot_equation_B, 2) - 4 * scale_norm_zroot_equation_A * scale_norm_zroot_equation_C)) / scale_norm_zroot_equation_A;



				double scale_norm_zroot = 0.0;
				if (scale_norm_zroot_0 > scale_norm_zroot_1) scale_norm_zroot = scale_norm_zroot_0;

				//save global scale normed root z 
				int Tid = t * 1;
				top_data[Tid] = scale_norm_zroot;				
			}

		}
	}


	template <typename Dtype>
	void DeepHandModelSolveScaleNormalizedGlobalZRootLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const int batSize = (bottom[0]->shape())[0];

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelSolveScaleNormalizedGlobalZRootLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelSolveScaleNormalizedGlobalZRootLayer);
	REGISTER_LAYER_CLASS(DeepHandModelSolveScaleNormalizedGlobalZRoot);
}  // namespace caffe
