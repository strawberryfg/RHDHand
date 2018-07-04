#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DeepHandModelCalcNormScaleLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		n_ = this->layer_param_.deep_hand_model_calc_norm_scale_param().n();
		m_ = this->layer_param_.deep_hand_model_calc_norm_scale_param().m();
	}
	template <typename Dtype>
	void DeepHandModelCalcNormScaleLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelCalcNormScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_3d_mono = bottom[0]->cpu_data();    //ground truth 3d monocular (camera frame global 3D)
		
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) 
		{
			int Zid = t * JointNum_RHD;
			int Tid = t;
			double s = sqrt(pow(gt_3d_mono[n_ * 3 + 0] - gt_3d_mono[m_ * 3 + 0], 2) + pow(gt_3d_mono[n_ * 3 + 1] - gt_3d_mono[m_ * 3 + 1], 2) + pow(gt_3d_mono[n_ * 3 + 2] - gt_3d_mono[m_ * 3 + 2], 2));
			top_data[Tid] = 1.0 / s; //save scale computed from ground truth mono 3d
		}
	}

	template <typename Dtype>
	void DeepHandModelCalcNormScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelCalcNormScaleLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelCalcNormScaleLayer);
	REGISTER_LAYER_CLASS(DeepHandModelCalcNormScale);
}