#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "HandModel_RDH.h"
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
/*0501: visualize left and right hand (joint & bone)
        check camera parameters (correct)
		bbx (direct minmax & center of mass)
		discriminate between left and right
  0502: output to all_hands
        contain info about left/right hand (only 21 joints)
*/
//#define show_image
#define save_image
#define output_to_folder
#define wait_key 0
#define test_camera_parameters

#define direct_min_max_coord
#define also_calculate_genuine_center_of_mass

#define test
#define pad_scale 1.5
#define pad_scale_mask_bbx 1.25
#define opaque_ratio 0.5
#ifdef train
#define N 41258
#define color_rgb_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\color\\"
#define gt_joint_2d_in_raw_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\matlab_gt_2d_in_raw\\"
#define gt_camera_param_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\matlab_gt_camera_k\\"
#define gt_joint_3d_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\matlab_gt_3d\\"
#define mask_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\mask\\"
#define vis_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\matlab_gt_vis\\"

//only 21 joints
//done
#define gt_2d_in_raw_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\gt_2d_in_raw\\"
//done
#define gt_vis_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\gt_vis\\"
//in all image (starting from 0)
//done
#define image_index_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\image_index\\"
//done
#define is_left_hand_or_not_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\is_left_hand_or_not\\"
//done
#define crop_image_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\crop_image\\"
//done
#define crop_gt_2d_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\crop_gt_2d\\"
//done
#define gt_joint_3d_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\gt_3d\\"
#define gt_depth_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\gt_depth\\"
#define calib_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\calib\\"

#define bbx_x1_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\bbx_x1\\"
#define bbx_y1_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\bbx_y1\\"
#define bbx_x2_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\bbx_x2\\"
#define bbx_y2_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\bbx_y2\\"
#else
#define N 2728
#define color_rgb_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\color\\"
#define gt_joint_2d_in_raw_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\matlab_gt_2d_in_raw\\"
#define gt_camera_param_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\matlab_gt_camera_k\\"
#define gt_joint_3d_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\matlab_gt_3d\\"
#define mask_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\mask\\"
#define vis_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\matlab_gt_vis\\"

//only 21 joints
#define gt_2d_in_raw_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\gt_2d_in_raw\\"
#define gt_vis_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\gt_vis\\"
//in all image (starting from 0)
#define image_index_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\image_index\\"
#define is_left_hand_or_not_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\is_left_hand_or_not\\"
#define crop_image_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\crop_image\\"
#define crop_gt_2d_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\crop_gt_2d\\"
#define gt_joint_3d_all_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\gt_3d\\"
#define gt_depth_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\gt_depth\\"
#define calib_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\calib\\"

#define bbx_x1_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_x1\\"
#define bbx_y1_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_y1\\"
#define bbx_x2_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_x2\\"
#define bbx_y2_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_y2\\"
#endif

#define len_stats_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\sum_of_len_gt.txt"

#define maxlen 111
using namespace std;
using namespace cv;

//calculate sum of length gt
double sum_of_len_gt_all, sum_of_len_gt_left, sum_of_len_gt_right;
int valid_num_all, valid_num_left, valid_num_right; //counter (all/left hand only/right hand only)

double gt_joint_2d_all[N][JointNumAll_RHD * 2];
double gt_joint_3d_all[N][JointNumAll_RHD * 3];
double gt_depth_all[N][JointNumAll_RHD];
double gt_root[N][6];
int valid_all[N][2];
int main()
{
	FILE *fout_sum_of_len_gt = fopen(len_stats_prefix, "w");
	for (int i = 0; i < N; i++)
	{
		printf("%d\n", i);

		//---------left/right hand
#ifdef output_to_folder

		char left_right_all_name[maxlen];
		sprintf(left_right_all_name, "%s%d%s", is_left_hand_or_not_all_prefix, 2 * i, ".txt");
		FILE *fout_left_right_all = fopen(left_right_all_name, "w");
		fprintf(fout_left_right_all, "%d\n", 1); //left hand
		fclose(fout_left_right_all);

		sprintf(left_right_all_name, "%s%d%s", is_left_hand_or_not_all_prefix, 2 * i + 1, ".txt");
		fout_left_right_all = fopen(left_right_all_name, "w");
		fprintf(fout_left_right_all, "%d\n", 0); //right hand
		fclose(fout_left_right_all);


		//---------save image index raw (all)
		char image_index_all_name[maxlen];
		sprintf(image_index_all_name, "%s%d%s", image_index_all_prefix, 2 * i, ".txt");
		FILE *fout_image_index = fopen(image_index_all_name, "w");
		fprintf(fout_image_index, "%d\n", i);
		fclose(fout_image_index);

		sprintf(image_index_all_name, "%s%d%s", image_index_all_prefix, 2 * i + 1, ".txt");
		fout_image_index = fopen(image_index_all_name, "w");
		fprintf(fout_image_index, "%d\n", i);
		fclose(fout_image_index);

#endif

		//read raw rgb
		char color_rgb_name[maxlen];
		sprintf(color_rgb_name, "%s%05d%s", color_rgb_prefix, i, ".png");
		Mat img = imread(color_rgb_name);
		Mat img_save = img.clone();

		//read mask
		char mask_name[maxlen];
		sprintf(mask_name, "%s%05d%s", mask_prefix, i, ".png");
		Mat mask = imread(mask_name, 0); //read as gray-scale

		char gt_joint_2d_in_raw_name[maxlen];
		sprintf(gt_joint_2d_in_raw_name, "%s%d%s", gt_joint_2d_in_raw_prefix, i, ".txt");
		double gt_joint_2d[JointNumAll_RHD * 2];

		FILE *fin_gt_joint_2d_in_raw = fopen(gt_joint_2d_in_raw_name, "r");
		for (int j = 0; j < JointNumAll_RHD * 2; j++) fscanf(fin_gt_joint_2d_in_raw, "%lf", &gt_joint_2d[j]);
		fclose(fin_gt_joint_2d_in_raw);

		//read visible mask
		char vis_mask_name[maxlen];
		sprintf(vis_mask_name, "%s%d%s", vis_prefix, i, ".txt");
		FILE *fin_vis_mask = fopen(vis_mask_name, "r");
		int vis_mask[JointNumAll_RHD];
		for (int j = 0; j < JointNumAll_RHD; j++) fscanf(fin_vis_mask, "%d", &vis_mask[j]);
		fclose(fin_vis_mask);


#ifdef output_to_folder
		//---------save vis raw (all)
		char vis_all_name[maxlen];
		sprintf(vis_all_name, "%s%d%s", gt_vis_all_prefix, 2 * i, ".txt");
		FILE *fout_vis_all = fopen(vis_all_name, "w");
		for (int j = 0; j < JointNumAll_RHD / 2; j++) fprintf(fout_vis_all, "%d\n", vis_mask[j]);
		fclose(fout_vis_all);
		
		sprintf(vis_all_name, "%s%d%s", gt_vis_all_prefix, 2 * i + 1, ".txt");
		fout_vis_all = fopen(vis_all_name, "w");
		for (int j = 0; j < JointNumAll_RHD / 2; j++) fprintf(fout_vis_all, "%d\n", vis_mask[j + JointNumAll_RHD / 2]);
		fclose(fout_vis_all);

		//---------save gt 2d raw (all)
		char gt_2d_in_raw_all_name[maxlen];
		//left hand
		sprintf(gt_2d_in_raw_all_name, "%s%d%s", gt_2d_in_raw_all_prefix, 2 * i, ".txt");
		FILE *fout_gt_2d_left_raw = fopen(gt_2d_in_raw_all_name, "w");
		for (int j = 0; j < (JointNumAll_RHD / 2); j++)
		{
			fprintf(fout_gt_2d_left_raw, "%12.6f %12.6f\n", gt_joint_2d[j * 2], gt_joint_2d[j * 2 + 1]);
		}
		fclose(fout_gt_2d_left_raw);

		sprintf(gt_2d_in_raw_all_name, "%s%d%s", gt_2d_in_raw_all_prefix, 2 * i + 1, ".txt");
		fout_gt_2d_left_raw = fopen(gt_2d_in_raw_all_name, "w");
		for (int j = 0; j < (JointNumAll_RHD / 2); j++) 
		{
			fprintf(fout_gt_2d_left_raw, "%12.6f %12.6f\n", gt_joint_2d[(j + JointNumAll_RHD / 2) * 2], gt_joint_2d[(j + JointNumAll_RHD / 2) * 2 + 1]);
		}
		fclose(fout_gt_2d_left_raw);
#endif


#ifdef test_camera_parameters
		//read model param
		char gt_camera_param_name[maxlen];
		sprintf(gt_camera_param_name, "%s%d%s", gt_camera_param_prefix, i, ".txt");
		double camera_k[3][3];
		FILE *fin_gt_camera_param = fopen(gt_camera_param_name, "r");
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				fscanf(fin_gt_camera_param, "%lf", &camera_k[j][k]);
			}
		}
		fclose(fin_gt_camera_param);
		
		//read gt 3d
		char gt_joint_3d_name[maxlen];
		sprintf(gt_joint_3d_name, "%s%d%s", gt_joint_3d_prefix, i, ".txt");
		double gt_joint_3d[JointNumAll_RHD * 3];
		FILE *fin_gt_joint_3d = fopen(gt_joint_3d_name, "r");
		for (int j = 0; j < JointNumAll_RHD * 3; j++) fscanf(fin_gt_joint_3d, "%lf", &gt_joint_3d[j]);
		fclose(fin_gt_joint_3d);

		//multiply by 10000
		for (int j = 0; j < JointNumAll_RHD; j++) 
		{
			gt_joint_3d[j * 3 + 2] *= 1000.0;
			gt_joint_3d[j * 3 + 0] *= 1000.0;
			gt_joint_3d[j * 3 + 1] *= 1000.0;
		}


#ifdef output_to_folder
		//---------save gt 3d (all)
		char gt_joint_3d_all_name[maxlen];
		sprintf(gt_joint_3d_all_name, "%s%d%s", gt_joint_3d_all_prefix, 2 * i, ".txt");
		FILE *fout_gt_joint_3d_all = fopen(gt_joint_3d_all_name, "w");
		for (int j = 0; j < JointNumAll_RHD / 2; j++)
		{
			fprintf(fout_gt_joint_3d_all, "%12.6f %12.6f %12.6f\n", gt_joint_3d[j * 3], gt_joint_3d[j * 3 + 1], gt_joint_3d[j * 3 + 2]);
		}
		fclose(fout_gt_joint_3d_all);

		sprintf(gt_joint_3d_all_name, "%s%d%s", gt_joint_3d_all_prefix, 2 * i + 1, ".txt");
		fout_gt_joint_3d_all = fopen(gt_joint_3d_all_name, "w");
		for (int j = 0; j < JointNumAll_RHD / 2; j++) {
			fprintf(fout_gt_joint_3d_all, "%12.6f %12.6f %12.6f\n", gt_joint_3d[(j + JointNumAll_RHD / 2) * 3], gt_joint_3d[(j + JointNumAll_RHD / 2) * 3 + 1], gt_joint_3d[(j + JointNumAll_RHD / 2) * 3 + 2]);
		}
		fclose(fout_gt_joint_3d_all);
#endif


		//clear gt 2d
		for (int j = 0; j < JointNumAll_RHD * 2; j++) gt_joint_2d[j] = 0.0;
		for (int j = 0; j < JointNumAll_RHD; j++)
		{
			double x = gt_joint_3d[j * 3], y = gt_joint_3d[j * 3 + 1], z = gt_joint_3d[j * 3 + 2];
			gt_joint_2d[j * 2] = x / z * camera_k[0][0] + camera_k[0][2];     //x / z * focus + u0
			gt_joint_2d[j * 2 + 1] = y / z * camera_k[1][1] + camera_k[1][2]; //y / z * focus + v0
		}
#endif



		//show joint 2d (42 joints)
		for (int j = 0; j < JointNumAll_RHD; j++)
		{
			double x = gt_joint_2d[j * 2];
			double y = gt_joint_2d[j * 2 + 1];
			circle(img, Point2d(x, y), 3, Scalar(color_pred_joint_all_RHD[j][0], color_pred_joint_all_RHD[j][1], color_pred_joint_all_RHD[j][2]), -2);
		}

		//connect joints (bone) all 40 bones
		for (int j = 0; j < BoneNumAll_RHD; j++)
		{
			double x1 = gt_joint_2d[bones_all_RHD[j][0] * 2], y1 = gt_joint_2d[bones_all_RHD[j][0] * 2 + 1];
			double x2 = gt_joint_2d[bones_all_RHD[j][1] * 2], y2 = gt_joint_2d[bones_all_RHD[j][1] * 2 + 1];
			line(img, Point2d(x1, y1), Point2d(x2, y2), Scalar(color_pred_bone_all_RHD[j][0], color_pred_bone_all_RHD[j][1], color_pred_bone_all_RHD[j][2]), 3);
		}

		//tight bounding box
		double min_x_left = 1e30, min_y_left = 1e30, max_x_left = -1e30, max_y_left = -1e30;
		double min_x_right = 1e30, min_y_right = 1e30, max_x_right = -1e30, max_y_right = -1e30;
		for (int j = 0; j < JointNumAll_RHD / 2; j++)
		{
			min_x_left = min(min_x_left, gt_joint_2d[j * 2]);
			min_y_left = min(min_y_left, gt_joint_2d[j * 2 + 1]);
			max_x_left = max(max_x_left, gt_joint_2d[j * 2]);
			max_y_left = max(max_y_left, gt_joint_2d[j * 2 + 1]);

			min_x_right = min(min_x_right, gt_joint_2d[(j + JointNumAll_RHD / 2) * 2]);
			min_y_right = min(min_y_right, gt_joint_2d[(j + JointNumAll_RHD / 2) * 2 + 1]);
			max_x_right = max(max_x_right, gt_joint_2d[(j + JointNumAll_RHD / 2) * 2]);
			max_y_right = max(max_y_right, gt_joint_2d[(j + JointNumAll_RHD / 2) * 2 + 1]);
		}
		int H_left = max_y_left - min_y_left, W_left = max_x_left - min_x_left;
		int H_right = max_y_right - min_y_right, W_right = max_x_right - min_x_right;
		//left
		if (H_left > W_left)
		{
			min_x_left = min_x_left - (H_left - W_left) / 2.0;
			max_x_left = max_x_left + (H_left - W_left) / 2.0;
		}
		else 
		{
			min_y_left = min_y_left - (W_left - H_left) / 2.0;
			max_y_left = max_y_left + (W_left - H_left) / 2.0;
		}

		//right
		if (H_right > W_right) 
		{
			min_x_right = min_x_right - (H_right - W_right) / 2.0;
			max_x_right = max_x_right + (H_right - W_right) / 2.0;
		} else 
		{
			min_y_right = min_y_right - (W_right - H_right) / 2.0;
			max_y_right = max_y_right + (W_right - H_right) / 2.0;
		}
		int S_left = max(H_left, W_left) * pad_scale;
		int S_right = max(H_right, W_right) * pad_scale;
		int center_x_left = (min_x_left + max_x_left) / 2.0, center_y_left = (min_y_left + max_y_left) / 2.0;
		int center_x_right = (min_x_right + max_x_right) / 2.0, center_y_right = (min_y_right + max_y_right) / 2.0;
		min_x_left = center_x_left - S_left / 2.0; max_x_left = center_x_left + S_left / 2.0;
		min_y_left = center_y_left - S_left / 2.0; max_y_left = center_y_left + S_left / 2.0;

		min_x_right = center_x_right - S_right / 2.0; max_x_right = center_x_right + S_right / 2.0;
		min_y_right = center_y_right - S_right / 2.0; max_y_right = center_y_right + S_right / 2.0;

		//connect 4 lines
		//left
		/*line(img, Point2d(min_x_left, min_y_left), Point2d(max_x_left, min_y_left), Scalar(255, 255, 255), 3);
		line(img, Point2d(min_x_left, min_y_left), Point2d(min_x_left, max_y_left), Scalar(255, 255, 255), 3);
		line(img, Point2d(min_x_left, max_y_left), Point2d(max_x_left, max_y_left), Scalar(255, 255, 255), 3);
		line(img, Point2d(max_x_left, min_y_left), Point2d(max_x_left, max_y_left), Scalar(255, 255, 255), 3);
		*/

		//right
		/*line(img, Point2d(min_x_right, min_y_right), Point2d(max_x_right, min_y_right), Scalar(255, 255, 255), 3);
		line(img, Point2d(min_x_right, min_y_right), Point2d(min_x_right, max_y_right), Scalar(255, 255, 255), 3);
		line(img, Point2d(min_x_right, max_y_right), Point2d(max_x_right, max_y_right), Scalar(255, 255, 255), 3);
		line(img, Point2d(max_x_right, min_y_right), Point2d(max_x_right, max_y_right), Scalar(255, 255, 255), 3);
		*/
#ifdef show_image
		imshow("raw", img);
		waitKey(wait_key);
#endif

		Mat mask_img = Mat::zeros(Size(img.cols, img.rows), CV_8UC3);
		for (int row = 0; row < img.rows; row++) 
		{
			for (int col = 0; col < img.cols; col++) 
			{
				for (int c = 0; c < 3; c++) 
				{
					if (mask.at<uchar>(row, col) <= 1)
						mask_img.at<Vec3b>(row, col)[c] = 0;
					//opaque_ratio * img.at<Vec3b>(row, col)[c] + (1.0 - opaque_ratio) * mask.at<uchar>(row, col);
					else mask_img.at<Vec3b>(row, col)[c] = img.at<Vec3b>(row, col)[c];
				}
			}
		}
#ifdef show_image
		imshow("mask", mask_img);
		waitKey(wait_key);
#endif

		//use rough pad_scale * ground truth 2d tight bbx to rule out mask of another hand
		Mat mask_left_hand_raw = img.clone();
		Mat mask_left_hand = mask.clone();
		for (int row = 0; row < mask_left_hand.rows; row++)
		{
			for (int col = 0; col < mask_left_hand.cols; col++)
			{
				if (row < min_y_left || row > max_y_left || col < min_x_left || col > max_x_left)
				{
					for (int c = 0; c < 3; c++) mask_left_hand_raw.at<Vec3b>(row, col)[c] = 0;
					mask_left_hand.at<uchar>(row, col) = 0;
				}
			}
		}
#ifdef show_image
		imshow("mask_left", mask_left_hand_raw);
		waitKey(wait_key);
#endif

		Mat mask_right_hand_raw = img.clone();
		Mat mask_right_hand = mask.clone();
		for (int row = 0; row < mask_right_hand.rows; row++) 
		{
			for (int col = 0; col < mask_right_hand.cols; col++) 
			{
				if (row < min_y_right || row > max_y_right || col < min_x_right || col > max_x_right) 
				{
					for (int c = 0; c < 3; c++) mask_right_hand_raw.at<Vec3b>(row, col)[c] = 0;
					mask_right_hand.at<uchar>(row, col) = 0;
				}
			}
		}
#ifdef show_image
		imshow("mask_right", mask_right_hand_raw);
		waitKey(wait_key);
#endif

		//calculate center of mass using direct min max coordinates
		int min_mask_left_x = 11111, max_mask_left_x = -11111;
		int min_mask_left_y = 11111, max_mask_left_y = -11111;
		int H_mask_left = 0, W_mask_left = 0;
		int center_mask_left_x = 0, center_mask_left_y = 0;
		int size_mask_left = 0;

		int min_mask_right_x = 11111, max_mask_right_x = -11111;
		int min_mask_right_y = 11111, max_mask_right_y = -11111;
		int H_mask_right = 0, W_mask_right = 0;
		int center_mask_right_x = 0, center_mask_right_y = 0;
		int size_mask_right = 0;

		//centroid (center of mass) correct calculation
		double centroid_mask_left_x = 0.0, centroid_mask_left_y = 0.0;
		int valid_mask_left_cnt = 0;

		double centroid_mask_right_x = 0.0, centroid_mask_right_y = 0.0;
		int valid_mask_right_cnt = 0;
#ifdef direct_min_max_coord
		//left hand
		for (int row = 0; row < mask_left_hand.rows; row++)
		{
			for (int col = 0; col < mask_left_hand.cols; col++)
			{
				if (mask_left_hand.at<uchar>(row, col) > 1) 
				{
					min_mask_left_x = min(min_mask_left_x, col);
					max_mask_left_x = max(max_mask_left_x, col);

					min_mask_left_y = min(min_mask_left_y, row);
					max_mask_left_y = max(max_mask_left_y, row);

					centroid_mask_left_x += double(col);
					centroid_mask_left_y += double(row);
					valid_mask_left_cnt++;
				}
			}
		}

		//centroid divided by valid count
		centroid_mask_left_x /= double(valid_mask_left_cnt);
		centroid_mask_left_y /= double(valid_mask_left_cnt);

		H_mask_left = max_mask_left_y - min_mask_left_y;
		W_mask_left = max_mask_left_x - min_mask_left_x;
		center_mask_left_x = (max_mask_left_x + min_mask_left_x) / 2;
		center_mask_left_y = (max_mask_left_y + min_mask_left_y) / 2;
		size_mask_left = max(H_mask_left, W_mask_left) * pad_scale_mask_bbx;
		min_mask_left_x = center_mask_left_x - size_mask_left / 2;
		min_mask_left_y = center_mask_left_y - size_mask_left / 2;
		max_mask_left_x = center_mask_left_x + size_mask_left / 2;
		max_mask_left_y = center_mask_left_y + size_mask_left / 2;

		Mat img_mask_left = img_save.clone();
		line(img_mask_left, Point2d(min_mask_left_x, min_mask_left_y), Point2d(max_mask_left_x, min_mask_left_y), Scalar(255, 255, 255), 3);
		line(img_mask_left, Point2d(min_mask_left_x, min_mask_left_y), Point2d(min_mask_left_x, max_mask_left_y), Scalar(255, 255, 255), 3);
		line(img_mask_left, Point2d(min_mask_left_x, max_mask_left_y), Point2d(max_mask_left_x, max_mask_left_y), Scalar(255, 255, 255), 3);
		line(img_mask_left, Point2d(max_mask_left_x, min_mask_left_y), Point2d(max_mask_left_x, max_mask_left_y), Scalar(255, 255, 255), 3);
#ifdef show_image
		imshow("mask_left_accurate_bbx", img_mask_left);
		waitKey(wait_key);
#endif

		//right hand
		for (int row = 0; row < mask_right_hand.rows; row++) 
		{
			for (int col = 0; col < mask_right_hand.cols; col++) 
			{
				if (mask_right_hand.at<uchar>(row, col) > 1) 
				{
					min_mask_right_x = min(min_mask_right_x, col);
					max_mask_right_x = max(max_mask_right_x, col);

					min_mask_right_y = min(min_mask_right_y, row);
					max_mask_right_y = max(max_mask_right_y, row);

					centroid_mask_right_x += double(col);
					centroid_mask_right_y += double(row);
					valid_mask_right_cnt++;
				}
			}
		}
		centroid_mask_right_x /= double(valid_mask_right_cnt);
		centroid_mask_right_y /= double(valid_mask_right_cnt);

		H_mask_right = max_mask_right_y - min_mask_right_y;
		W_mask_right = max_mask_right_x - min_mask_right_x;
		center_mask_right_x = (max_mask_right_x + min_mask_right_x) / 2;
		center_mask_right_y = (max_mask_right_y + min_mask_right_y) / 2;
		size_mask_right = max(H_mask_right, W_mask_right) * pad_scale_mask_bbx;
		min_mask_right_x = center_mask_right_x - size_mask_right / 2;
		min_mask_right_y = center_mask_right_y - size_mask_right / 2;
		max_mask_right_x = center_mask_right_x + size_mask_right / 2;
		max_mask_right_y = center_mask_right_y + size_mask_right / 2;

		Mat img_mask_right = img_save.clone();
		line(img_mask_right, Point2d(min_mask_right_x, min_mask_right_y), Point2d(max_mask_right_x, min_mask_right_y), Scalar(255, 255, 255), 3);
		line(img_mask_right, Point2d(min_mask_right_x, min_mask_right_y), Point2d(min_mask_right_x, max_mask_right_y), Scalar(255, 255, 255), 3);
		line(img_mask_right, Point2d(min_mask_right_x, max_mask_right_y), Point2d(max_mask_right_x, max_mask_right_y), Scalar(255, 255, 255), 3);
		line(img_mask_right, Point2d(max_mask_right_x, min_mask_right_y), Point2d(max_mask_right_x, max_mask_right_y), Scalar(255, 255, 255), 3);
#ifdef show_image
		imshow("mask_right_accurate_bbx", img_mask_right);
		waitKey(wait_key);
#endif

#endif

#ifdef also_calculate_genuine_center_of_mass
		//left hand
		min_mask_left_x = centroid_mask_left_x - size_mask_left / 2;
		min_mask_left_y = centroid_mask_left_y - size_mask_left / 2;
		max_mask_left_x = centroid_mask_left_x + size_mask_left / 2;
		max_mask_left_y = centroid_mask_left_y + size_mask_left / 2;

		Mat img_mask_left_com = img_save.clone();
		line(img_mask_left_com, Point2d(min_mask_left_x, min_mask_left_y), Point2d(max_mask_left_x, min_mask_left_y), Scalar(255, 255, 255), 3);
		line(img_mask_left_com, Point2d(min_mask_left_x, min_mask_left_y), Point2d(min_mask_left_x, max_mask_left_y), Scalar(255, 255, 255), 3);
		line(img_mask_left_com, Point2d(min_mask_left_x, max_mask_left_y), Point2d(max_mask_left_x, max_mask_left_y), Scalar(255, 255, 255), 3);
		line(img_mask_left_com, Point2d(max_mask_left_x, min_mask_left_y), Point2d(max_mask_left_x, max_mask_left_y), Scalar(255, 255, 255), 3);
#ifdef show_image
		imshow("mask_left_accurate_bbx_com", img_mask_left_com);
		waitKey(wait_key);
#endif

		//right hand
		min_mask_right_x = centroid_mask_right_x - size_mask_right / 2;
		min_mask_right_y = centroid_mask_right_y - size_mask_right / 2;
		max_mask_right_x = centroid_mask_right_x + size_mask_right / 2;
		max_mask_right_y = centroid_mask_right_y + size_mask_right / 2;

		Mat img_mask_right_com = img_save.clone();
		line(img_mask_right_com, Point2d(min_mask_right_x, min_mask_right_y), Point2d(max_mask_right_x, min_mask_right_y), Scalar(255, 255, 255), 3);
		line(img_mask_right_com, Point2d(min_mask_right_x, min_mask_right_y), Point2d(min_mask_right_x, max_mask_right_y), Scalar(255, 255, 255), 3);
		line(img_mask_right_com, Point2d(min_mask_right_x, max_mask_right_y), Point2d(max_mask_right_x, max_mask_right_y), Scalar(255, 255, 255), 3);
		line(img_mask_right_com, Point2d(max_mask_right_x, min_mask_right_y), Point2d(max_mask_right_x, max_mask_right_y), Scalar(255, 255, 255), 3);
#ifdef show_image
		imshow("mask_right_accurate_bbx_com", img_mask_right_com);
		waitKey(wait_key);
#endif

#endif


		/*Till now the bounding box of left hand is 
		min_mask_left_x, max_mask_left_x
		min_mask_left_y, max_mask_left_x

		that of right hand is
		min_mask_right_x, max_mask_right_x
		min_mask_right_y, max_mask_right_x
		*/

#ifdef output_to_folder
		//left hand
		char bbx_x1_name[maxlen];
		sprintf(bbx_x1_name, "%s%d%s", bbx_x1_prefix, 2 * i, ".txt");
		char bbx_y1_name[maxlen];
		sprintf(bbx_y1_name, "%s%d%s", bbx_y1_prefix, 2 * i, ".txt");
		char bbx_x2_name[maxlen];
		sprintf(bbx_x2_name, "%s%d%s", bbx_x2_prefix, 2 * i, ".txt");
		char bbx_y2_name[maxlen];
		sprintf(bbx_y2_name, "%s%d%s", bbx_y2_prefix, 2 * i, ".txt");
		FILE *fout_x1 = fopen(bbx_x1_name, "w");
		fprintf(fout_x1, "%d\n", (int)min_mask_left_x);
		fclose(fout_x1);

		FILE *fout_y1 = fopen(bbx_y1_name, "w");
		fprintf(fout_y1, "%d\n", (int)min_mask_left_y);
		fclose(fout_y1);

		FILE *fout_x2 = fopen(bbx_x2_name, "w");
		fprintf(fout_x2, "%d\n", (int)max_mask_left_x);
		fclose(fout_x2);

		FILE *fout_y2 = fopen(bbx_y2_name, "w");
		fprintf(fout_y2, "%d\n", (int)max_mask_left_y);
		fclose(fout_y2);

		//right hand		
		sprintf(bbx_x1_name, "%s%d%s", bbx_x1_prefix, 2 * i + 1, ".txt");		
		sprintf(bbx_y1_name, "%s%d%s", bbx_y1_prefix, 2 * i + 1, ".txt");		
		sprintf(bbx_x2_name, "%s%d%s", bbx_x2_prefix, 2 * i + 1, ".txt");		
		sprintf(bbx_y2_name, "%s%d%s", bbx_y2_prefix, 2 * i + 1, ".txt");
		fout_x1 = fopen(bbx_x1_name, "w");
		fprintf(fout_x1, "%d\n", (int)min_mask_right_x);
		fclose(fout_x1);

		fout_y1 = fopen(bbx_y1_name, "w");
		fprintf(fout_y1, "%d\n", (int)min_mask_right_y);
		fclose(fout_y1);

		fout_x2 = fopen(bbx_x2_name, "w");
		fprintf(fout_x2, "%d\n", (int)max_mask_right_x);
		fclose(fout_x2);

		fout_y2 = fopen(bbx_y2_name, "w");
		fprintf(fout_y2, "%d\n", (int)max_mask_right_y);
		fclose(fout_y2);
#endif

		//save left hand
		Mat img_crop_left = Mat::zeros(Size(max_mask_left_x - min_mask_left_x + 1, max_mask_left_y - min_mask_left_y + 1), CV_8UC3);
		for (int row = min_mask_left_y; row <= max_mask_left_y; row++)
		{
			for (int col = min_mask_left_x; col <= max_mask_left_x; col++)
			{
				if (row >= 0 && row < img_save.rows && col >= 0 && col < img_save.cols)
				{
					for (int c = 0; c < 3; c++)
					{
						img_crop_left.at<Vec3b>(row - min_mask_left_y, col - min_mask_left_x)[c] = img_save.at<Vec3b>(row, col)[c];
					}					
				}
				else 
				{
					for (int c = 0; c < 3; c++)
					{
						img_crop_left.at<Vec3b>(row - min_mask_left_y, col - min_mask_left_x)[c] = 128;
					}
				}
			}
		}
#ifdef show_image
		imshow("crop_left_hand", img_crop_left);
		waitKey(wait_key);
#endif

#ifdef save_image
		char save_crop_left_hand_name[maxlen];
		sprintf(save_crop_left_hand_name, "%s%d%s", crop_image_all_prefix, 2 * i, ".png");
		imwrite(save_crop_left_hand_name, img_crop_left);
#endif

		//save right hand
		Mat img_crop_right = Mat::zeros(Size(max_mask_right_x - min_mask_right_x + 1, max_mask_right_y - min_mask_right_y + 1), CV_8UC3);
		for (int row = min_mask_right_y; row <= max_mask_right_y; row++) 
		{
			for (int col = min_mask_right_x; col <= max_mask_right_x; col++) 
			{
				if (row >= 0 && row < img_save.rows && col >= 0 && col < img_save.cols) 
				{
					for (int c = 0; c < 3; c++) 
					{
						img_crop_right.at<Vec3b>(row - min_mask_right_y, col - min_mask_right_x)[c] = img_save.at<Vec3b>(row, col)[c];
					}
				} 
				else 
				{
					for (int c = 0; c < 3; c++) 
					{
						img_crop_right.at<Vec3b>(row - min_mask_right_y, col - min_mask_right_x)[c] = 128;
					}
				}
			}
		}
#ifdef show_image
		imshow("crop_right_hand", img_crop_right);
		waitKey(wait_key);
#endif

#ifdef save_image
		char save_crop_right_hand_name[maxlen];
		sprintf(save_crop_right_hand_name, "%s%d%s", crop_image_all_prefix, 2 * i + 1, ".png");
		imwrite(save_crop_right_hand_name, img_crop_right);
#endif


		//save normalized gt joint 2d left hand
		
#ifdef output_to_folder
		char crop_gt_2d_name[maxlen];
		sprintf(crop_gt_2d_name, "%s%d%s", crop_gt_2d_all_prefix, 2 * i, ".txt");
		FILE *fout_crop_gt_2d = fopen(crop_gt_2d_name, "w");
#endif
		for (int j = 0; j < JointNumAll_RHD / 2; j++)
		{
			//normalize to [0, 1] crop coordinate
			gt_joint_2d[j * 2] = (gt_joint_2d[j * 2] - min_mask_left_x) / (max_mask_left_x - min_mask_left_x + 1);
			gt_joint_2d[j * 2 + 1] = (gt_joint_2d[j * 2 + 1] - min_mask_left_y) / (max_mask_left_y - min_mask_left_y + 1);
#ifdef output_to_folder
			fprintf(fout_crop_gt_2d, "%12.6f %12.6f\n", gt_joint_2d[j * 2], gt_joint_2d[j * 2 + 1]);
#endif
		}

#ifdef output_to_folder
		fclose(fout_crop_gt_2d);

		//save normalized gt joint 2d right hand		
		sprintf(crop_gt_2d_name, "%s%d%s", crop_gt_2d_all_prefix, 2 * i + 1, ".txt");
		fout_crop_gt_2d = fopen(crop_gt_2d_name, "w");
#endif

		for (int j = 0; j < JointNumAll_RHD / 2; j++) 
		{
			//normalize to [0, 1] crop coordinate
			gt_joint_2d[(j + JointNumAll_RHD / 2) * 2] = (gt_joint_2d[(j + JointNumAll_RHD / 2) * 2] - min_mask_right_x) / (max_mask_right_x - min_mask_right_x + 1);
			gt_joint_2d[(j + JointNumAll_RHD / 2) * 2 + 1] = (gt_joint_2d[(j + JointNumAll_RHD / 2) * 2 + 1] - min_mask_right_y) / (max_mask_right_y - min_mask_right_y + 1);
#ifdef output_to_folder
			fprintf(fout_crop_gt_2d, "%12.6f %12.6f\n", gt_joint_2d[(j + JointNumAll_RHD / 2) * 2], gt_joint_2d[(j + JointNumAll_RHD / 2) * 2 + 1]);
#endif
		}

#ifdef output_to_folder
		fclose(fout_crop_gt_2d);
#endif

		//gen gt depth
		char gt_depth_name[maxlen];		
		double gt_depth[JointNumAll_RHD];
		double s2d = 0.0;
		double s3d = 0.0;
		//------gen gt depth -> left hand
#ifdef output_to_folder
		sprintf(gt_depth_name, "%s%d%s", gt_depth_prefix, 2 * i, ".txt");
		FILE *fout_gt_depth = fopen(gt_depth_name, "w");
#endif

		for (int j = 0; j < BoneNum_RHD; j++)
		{
			int u = bones_RHD[j][0];
			int v = bones_RHD[j][1];


			// u v in range[0, 20] p1 p2 q1 q2 in range [0, 224]
			double p1 = 224 * gt_joint_2d[u * 2];
			double p2 = 224 * gt_joint_2d[v * 2];
			double q1 = 224 * gt_joint_2d[u * 2 + 1];
			double q2 = 224 * gt_joint_2d[v * 2 + 1];


			double x1 = (gt_joint_3d[u * 3] - gt_joint_3d[root_RHD * 3]);
			double y1 = (gt_joint_3d[u * 3 + 1] - gt_joint_3d[root_RHD * 3 + 1]);
			double x2 = (gt_joint_3d[v * 3] - gt_joint_3d[root_RHD * 3]);
			double y2 = (gt_joint_3d[v * 3 + 1] - gt_joint_3d[root_RHD * 3 + 1]);
			//image coordinate 
			s2d += sqrt(pow(p1 - p2, 2) + pow(q1 - q2, 2));
			//frame coordinate
			s3d += sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
		}
		
		//2d/3d
		double scale = s2d / s3d;		
		for (int j = 0; j < JointNumAll_RHD / 2; j++)
		{
			//gt depth ranges in [-1, 1] ((z-zroot)*scale+112- 112)/112 return a value in [-1, 1]
			gt_depth[j] = (gt_joint_3d[j * 3 + 2] - gt_joint_3d[root_RHD * 3 + 2]) * scale / 112;

#ifdef output_to_folder
			fprintf(fout_gt_depth, "%12.6f\n", gt_depth[j]);
#endif
		}

#ifdef output_to_folder
		fclose(fout_gt_depth);
		
		//------gen gt depth -> right hand
		sprintf(gt_depth_name, "%s%d%s", gt_depth_prefix, 2 * i + 1, ".txt");
		fout_gt_depth = fopen(gt_depth_name, "w");
#endif
		s2d = 0.0;
		s3d = 0.0;
		for (int j = 0; j < BoneNum_RHD; j++) 
		{
			int u = bones_RHD[j][0] + JointNumAll_RHD / 2;
			int v = bones_RHD[j][1] + JointNumAll_RHD / 2;


			// u v in range[0, 20] p1 p2 q1 q2 in range [0, 224]
			double p1 = 224 * gt_joint_2d[u * 2];
			double p2 = 224 * gt_joint_2d[v * 2];
			double q1 = 224 * gt_joint_2d[u * 2 + 1];
			double q2 = 224 * gt_joint_2d[v * 2 + 1];


			double x1 = (gt_joint_3d[u * 3] - gt_joint_3d[(root_RHD + JointNumAll_RHD / 2) * 3]);
			double y1 = (gt_joint_3d[u * 3 + 1] - gt_joint_3d[(root_RHD + JointNumAll_RHD / 2) * 3 + 1]);
			double x2 = (gt_joint_3d[v * 3] - gt_joint_3d[(root_RHD + JointNumAll_RHD / 2) * 3]);
			double y2 = (gt_joint_3d[v * 3 + 1] - gt_joint_3d[(root_RHD + JointNumAll_RHD / 2) * 3 + 1]);
			//image coordinate 
			s2d += sqrt(pow(p1 - p2, 2) + pow(q1 - q2, 2));
			//frame coordinate
			s3d += sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
		}

		//2d/3d
		scale = s2d / s3d;
		for (int j = 0; j < JointNumAll_RHD / 2; j++) 
		{
			//gt depth ranges in [-1, 1] ((z-zroot)*scale+112- 112)/112 return a value in [-1, 1]
			gt_depth[j + JointNumAll_RHD / 2] = (gt_joint_3d[(j + JointNumAll_RHD / 2) * 3 + 2] - gt_joint_3d[(root_RHD + JointNumAll_RHD / 2) * 3 + 2]) * scale / 112;
#ifdef output_to_folder
			fprintf(fout_gt_depth, "%12.6f\n", gt_depth[j + JointNumAll_RHD / 2]);
#endif
		}
#ifdef output_to_folder
		fclose(fout_gt_depth);
#endif

		//calc stats sum of len
		//---------left hand
		if (min_mask_left_x >= 0 && max_mask_left_x < img_save.cols && min_mask_left_y >= 0 && max_mask_left_y <= img_save.rows) 
		{
			bool not_valid = false;
			for (int j = 0; j < JointNumAll_RHD; j++) if (gt_joint_2d[j] < -5.0 || gt_joint_2d[j] > 5.0) { not_valid = true; break; }
			valid_all[i][0] = !not_valid;
			if (!not_valid)
			{
				//not out of bound
				double t_sum = 0.0;
				for (int j = 0; j < BoneNum_RHD; j++) 
				{
					int u = bones_RHD[j][0];
					int v = bones_RHD[j][1];
					t_sum += sqrt(pow(224 * gt_joint_2d[u * 2] - 224 * gt_joint_2d[v * 2], 2) + pow(224 * gt_joint_2d[u * 2 + 1] - 224 * gt_joint_2d[v * 2 + 1], 2) + pow((gt_depth[u] + 1.0) / 2.0 * 224.0 - (gt_depth[v] + 1.0) / 2.0 * 224.0, 2));
				}
				valid_num_all++;
				valid_num_left++;
				sum_of_len_gt_all += t_sum;
				sum_of_len_gt_left += t_sum;
			}

		}
		else valid_all[i][0] = 0; //not valid
		//---------right hand
		if (min_mask_right_x >= 0 && max_mask_right_x < img_save.cols && min_mask_right_y >= 0 && max_mask_right_y <= img_save.rows) 
		{
			bool not_valid = false;
			for (int j = JointNumAll_RHD; j < JointNumAll_RHD * 2; j++) if (gt_joint_2d[j] < -5.0 || gt_joint_2d[j] > 5.0) { not_valid = true; break; }
			valid_all[i][1] = !not_valid;
			if (!not_valid)
			{
				//not out of bound
				double t_sum = 0.0;
				for (int j = 0; j < BoneNum_RHD; j++) 
				{
					int u = bones_RHD[j][0] + JointNumAll_RHD / 2;
					int v = bones_RHD[j][1] + JointNumAll_RHD / 2;
					t_sum += sqrt(pow(224 * gt_joint_2d[u * 2] - 224 * gt_joint_2d[v * 2], 2) + 
						          pow(224 * gt_joint_2d[u * 2 + 1] - 224 * gt_joint_2d[v * 2 + 1], 2) + 
								   pow((gt_depth[u] + 1.0) / 2.0 * 224.0 - (gt_depth[v] + 1.0) / 2.0 * 224.0, 2));
				}				
				valid_num_all++;
				valid_num_right++;
				sum_of_len_gt_all += t_sum;
				sum_of_len_gt_right += t_sum;
			}			
		}
		else valid_all[i][1] = 0;
		//save gt joint 2d 3d root depth to all
		
		for (int j = 0; j < JointNumAll_RHD * 2; j++) gt_joint_2d_all[i][j] = gt_joint_2d[j];
		for (int j = 0; j < JointNumAll_RHD * 3; j++) gt_joint_3d_all[i][j] = gt_joint_3d[j];
		for (int j = 0; j < JointNumAll_RHD; j++) gt_depth_all[i][j] = gt_depth[j];
		for (int j = 0; j < 3; j++) gt_root[i][j] = gt_joint_3d[root_RHD * 3 + j];		
		for (int j = 0; j < 3; j++) gt_root[i][j + 3] = gt_joint_3d[(root_RHD + JointNumAll_RHD / 2) * 3 + j];

	}

	//divide by num
	printf("sum of len gt all  :  %12.6f\n", sum_of_len_gt_all);
	printf("sum of len gt left :  %12.6f\n", sum_of_len_gt_left);
	printf("sum of len gt right:  %12.6f\n", sum_of_len_gt_right);

	sum_of_len_gt_all /= double(valid_num_all);
	sum_of_len_gt_left /= double(valid_num_left);
	sum_of_len_gt_right /= double(valid_num_right);
	printf("sum of len gt all  :  %12.6f\n", sum_of_len_gt_all);
	printf("sum of len gt left :  %12.6f\n", sum_of_len_gt_left);
	printf("sum of len gt right:  %12.6f\n", sum_of_len_gt_right);

	fprintf(fout_sum_of_len_gt, "sum of len gt all  :  %12.6f\n", sum_of_len_gt_all);
	fprintf(fout_sum_of_len_gt, "sum of len gt left :  %12.6f\n", sum_of_len_gt_left);
	fprintf(fout_sum_of_len_gt, "sum of len gt right:  %12.6f\n", sum_of_len_gt_right);




	//Test calibration
	for (int i = 0; i < N; i++)
	{
		
		double new_calib_gt_joint_3d[JointNumAll_RHD * 3];
		for (int hand = 0; hand < 2; hand++) //left/right hand
		{
#ifdef output_to_folder
			char calib_name[maxlen];
			sprintf(calib_name, "%s%d%s", calib_prefix, 2 * i + hand, ".txt");
			FILE *fout_calib = fopen(calib_name, "w");
#endif
			double sum_of_len_pred = 0.0;
			for (int j = 0; j < BoneNum_RHD; j++)
			{
				int u = bones_RHD[j][0] + hand * JointNum_RHD;
				int v = bones_RHD[j][1] + hand * JointNum_RHD;
				sum_of_len_pred += sqrt(pow(224 * gt_joint_2d_all[i][u * 2 + 0] - 224 * gt_joint_2d_all[i][v * 2 + 0], 2) + 
					                    pow(224 * gt_joint_2d_all[i][u * 2 + 1] - 224 * gt_joint_2d_all[i][v * 2 + 1], 2) + 
										pow((gt_depth_all[i][u] + 1.0) / 2.0 * 224.0 - (gt_depth_all[i][v] + 1.0) / 2.0 * 224.0, 2));
			}
#ifdef output_to_folder
			fprintf(fout_calib, "%12.6f\n", sum_of_len_pred);
			fprintf(fout_calib, "all:\n");
#endif
			for (int j = 0; j < JointNum_RHD; j++) 
			{
				int u = j + hand * JointNum_RHD;
				double x1 = 224 * (gt_joint_2d_all[i][u * 2] - gt_joint_2d_all[i][(root_RHD + hand * JointNum_RHD) * 2]);
				double y1 = 224 * (gt_joint_2d_all[i][u * 2 + 1] - gt_joint_2d_all[i][(root_RHD + hand * JointNum_RHD) * 2 + 1]);
				double z1 = (gt_depth_all[i][u] + 1.0) / 2.0 * 224.0 - (gt_depth_all[i][(root_RHD + hand * JointNum_RHD)] + 1.0) / 2.0 * 224.0;
				//(t + 1) / 2.0 * resolution

				new_calib_gt_joint_3d[u * 3] = x1 / sum_of_len_pred * sum_of_len_gt_all + gt_root[i][0 + hand * 3];
				new_calib_gt_joint_3d[u * 3 + 1] = y1 / sum_of_len_pred * sum_of_len_gt_all + gt_root[i][1 + hand * 3];
				new_calib_gt_joint_3d[u * 3 + 2] = z1 / sum_of_len_pred * sum_of_len_gt_all + gt_root[i][2 + hand * 3];
				
				double euc = 0.0;
				for (int k = 0; k < 3; k++) 
				{
#ifdef output_to_folder
					fprintf(fout_calib, "%12.6f ", new_calib_gt_joint_3d[u * 3 + k]);
#endif
					euc += pow(new_calib_gt_joint_3d[u * 3 + k] - gt_joint_3d_all[i][u * 3 + k], 2);
				}
				euc = sqrt(euc);
#ifdef output_to_folder
				for (int k = 0; k < 3; k++) fprintf(fout_calib, "%12.6f ", gt_joint_3d_all[i][u * 3 + k]);
				fprintf(fout_calib, "%12.6f ", euc);
				fprintf(fout_calib, "\n");
#endif
			}
#ifdef output_to_folder
			fprintf(fout_calib, "\n\nleft/right:\n\n");
#endif
			double sum_of_len_gt_left_right = (hand == 0 ? sum_of_len_gt_left : sum_of_len_gt_right);

			for (int j = 0; j < JointNum_RHD; j++) 
			{
				int u = j + hand * JointNum_RHD;
				double x1 = 224 * (gt_joint_2d_all[i][u * 2] - gt_joint_2d_all[i][(root_RHD + hand * JointNum_RHD) * 2]);
				double y1 = 224 * (gt_joint_2d_all[i][u * 2 + 1] - gt_joint_2d_all[i][(root_RHD + hand * JointNum_RHD) * 2 + 1]);
				double z1 = (gt_depth_all[i][u] + 1.0) / 2.0 * 224.0 - (gt_depth_all[i][(root_RHD + hand * JointNum_RHD)] + 1.0) / 2.0 * 224.0;
				//(t + 1) / 2.0 * resolution

				new_calib_gt_joint_3d[u * 3] = x1 / sum_of_len_pred * sum_of_len_gt_left_right + gt_root[i][0 + hand * 3];
				new_calib_gt_joint_3d[u * 3 + 1] = y1 / sum_of_len_pred * sum_of_len_gt_left_right + gt_root[i][1 + hand * 3];
				new_calib_gt_joint_3d[u * 3 + 2] = z1 / sum_of_len_pred * sum_of_len_gt_left_right + gt_root[i][2 + hand * 3];

				double euc = 0.0;
				for (int k = 0; k < 3; k++) 
				{
#ifdef output_to_folder
					fprintf(fout_calib, "%12.6f ", new_calib_gt_joint_3d[u * 3 + k]);
#endif
					euc += pow(new_calib_gt_joint_3d[u * 3 + k] - gt_joint_3d_all[i][u * 3 + k], 2);
				}
				euc = sqrt(euc);
#ifdef output_to_folder
				for (int k = 0; k < 3; k++) fprintf(fout_calib, "%12.6f ", gt_joint_3d_all[i][u * 3 + k]);
				fprintf(fout_calib, "%12.6f ", euc);
				fprintf(fout_calib, "\n");
#endif
			}

#ifdef output_to_folder
			fclose(fout_calib);
#endif
		}
	}
	fclose(fout_sum_of_len_gt);
	return 0;
}