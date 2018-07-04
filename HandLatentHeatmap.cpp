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
#define train

//#define print_to_console
#define save_calib

#define avg_bone_file "D:\\handpose\\CalcRHDBoneStats\\avg_bone.txt"

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

//this line is different from another version of code
#define calib_prefix "C:\\RHD_v1-1\\RHD_published_v2\\training\\all_hand\\calib_latent_heatmap\\"


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


#define calib_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\calib_latent_heatmap\\"

#define bbx_x1_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_x1\\"
#define bbx_y1_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_y1\\"
#define bbx_x2_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_x2\\"
#define bbx_y2_prefix "C:\\RHD_v1-1\\RHD_published_v2\\evaluation\\all_hand\\bbx_y2\\"
#endif
#define maxlen 111

#define C 1.0 //as in raw paper

using namespace std;
using namespace cv;

//2.5D normalized coord
double uk[JointNum_RHD], vk[JointNum_RHD], zkr[JointNum_RHD];

//recover scale back absolute 3D pose
double recover_back_scale_norm_xk[JointNum_RHD], recover_back_scale_norm_yk[JointNum_RHD], recover_back_scale_norm_zk[JointNum_RHD];

//scale-normalized absolute 3D pose
double scale_norm_xk[JointNum_RHD], scale_norm_yk[JointNum_RHD], scale_norm_zk[JointNum_RHD];

//scale-normalized root-relative 2.5D pose
double scale_norm_uk[JointNum_RHD], scale_norm_vk[JointNum_RHD], scale_norm_zkr[JointNum_RHD];

//specific pair of keypoints index MCP -> wrist 
const int n = bones_RHD[ref_bone_RHD][0];
const int m = bones_RHD[ref_bone_RHD][1];

double stats_avg_bone[BoneNum_RHD];

void ParseAverageBoneLen()
{
	FILE *fin_bone = fopen(avg_bone_file, "r");
	for (int j = 0; j < BoneNum_RHD; j++) fscanf(fin_bone, "%lf", &stats_avg_bone[j]);
	fclose(fin_bone);
}

int main()
{
	ParseAverageBoneLen();

	for (int i = 0; i < N; i++)
	{
		for (int hand = 0; hand < 2; hand++) 
			//left right hand
		{
#ifdef print_to_console
			printf("%5d %5d\n", i, hand);
#endif
			if (i == 956 && hand == 1)
			{
				cout << "fdsfd\n";
			}

			char bbx_x1_name[maxlen];
			sprintf(bbx_x1_name, "%s%d%s", bbx_x1_prefix, 2 * i + hand, ".txt");
			FILE *fin_bbx_x1 = fopen(bbx_x1_name, "r");
			int bbx_x1;
			fscanf(fin_bbx_x1, "%d", &bbx_x1);
			fclose(fin_bbx_x1);
			if (bbx_x1 <= 0 || bbx_x1 >= 640) continue;
			

			char bbx_y1_name[maxlen];
			sprintf(bbx_y1_name, "%s%d%s", bbx_y1_prefix, 2 * i + hand, ".txt");
			FILE *fin_bbx_y1 = fopen(bbx_y1_name, "r");
			int bbx_y1;
			fscanf(fin_bbx_y1, "%d", &bbx_y1);
			fclose(fin_bbx_y1);
			if (bbx_y1 <= 0 || bbx_y1 >= 640) continue;
			

			char bbx_x2_name[maxlen];
			sprintf(bbx_x2_name, "%s%d%s", bbx_x2_prefix, 2 * i + hand, ".txt");
			FILE *fin_bbx_x2 = fopen(bbx_x2_name, "r");
			int bbx_x2;
			fscanf(fin_bbx_x2, "%d", &bbx_x2);
			fclose(fin_bbx_x2);
			if (bbx_x2 <= 0 || bbx_x2 >= 640) continue;
			

			char bbx_y2_name[maxlen];
			sprintf(bbx_y2_name, "%s%d%s", bbx_y2_prefix, 2 * i + hand, ".txt");
			FILE *fin_bbx_y2 = fopen(bbx_y2_name, "r");
			int bbx_y2;
			fscanf(fin_bbx_y2, "%d", &bbx_y2);
			fclose(fin_bbx_y2);
			if (bbx_y2 <= 0 || bbx_y2 >= 640) continue;
			

			//read crop gt 2d 
			char crop_gt_2d_name[maxlen];
			sprintf(crop_gt_2d_name, "%s%d%s", crop_gt_2d_all_prefix, 2 * i + hand, ".txt");
			FILE *fin_crop_gt_2d = fopen(crop_gt_2d_name, "r");

			//crop 2d -> global 2d
			for (int j = 0; j < JointNum_RHD; j++)
			{
				fscanf(fin_crop_gt_2d, "%lf %lf", &uk[j], &vk[j]);

				uk[j] = uk[j] * (bbx_x2 - bbx_x1) + bbx_x1;
				vk[j] = vk[j] * (bbx_y2 - bbx_y1) + bbx_y1;
			}
			fclose(fin_crop_gt_2d);

			//read gt camera
			char gt_camera_name[maxlen];
			sprintf(gt_camera_name, "%s%d%s", gt_camera_param_prefix, i, ".txt");
			double camera_k[3][3];
			FILE *fin_camera = fopen(gt_camera_name, "r");
			for (int j = 0; j < 3; j++) 
			{
				for (int k = 0; k < 3; k++)
				{
					fscanf(fin_camera, "%lf", &camera_k[j][k]);
				}
			}
				
			fclose(fin_camera);
			double fx = camera_k[0][0], fy = camera_k[1][1], u0 = camera_k[0][2], v0 = camera_k[1][2];


			//read gt 3d mono
			char gt_3d_mono_name[maxlen];
			double gt_3d_mono[JointNum_RHD * 3];
			sprintf(gt_3d_mono_name, "%s%d%s", gt_joint_3d_all_prefix, 2 * i + hand, ".txt");
			FILE *fin_gt_3d_mono = fopen(gt_3d_mono_name, "r");
			for (int j = 0; j < JointNum_RHD; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					fscanf(fin_gt_3d_mono, "%lf", &gt_3d_mono[j * 3 + k]);				
					gt_3d_mono[j * 3 + k] /= 1000.0; //initially * 1000 -> mm
				}
			}
			fclose(fin_gt_3d_mono);

			for (int j = 0; j < JointNum_RHD; j++)
			{
				zkr[j] = gt_3d_mono[j * 3 + 2] - gt_3d_mono[root_RHD * 3 + 2];
				//root relative (not scaled till now)
			}
			//compute s (according to relative bone)
			double s = 0.0;			
			
			s = sqrt(pow(gt_3d_mono[n * 3 + 0] - gt_3d_mono[m * 3 + 0], 2) + pow(gt_3d_mono[n * 3 + 1] - gt_3d_mono[m * 3 + 1], 2) + pow(gt_3d_mono[n * 3 + 2] - gt_3d_mono[m * 3 + 2], 2));
			//s = 1.0;
			//scale normalize
			for (int j = 0; j < JointNum_RHD; j++)
			{
				scale_norm_uk[j] = uk[j];
				scale_norm_vk[j] = vk[j];
				
				scale_norm_zkr[j] = zkr[j] * C / s;
			}
			fclose(fin_gt_3d_mono);

			/*get xn xm yn ym*/
			double xn = (uk[n] - u0) * (scale_norm_zkr[n] + gt_3d_mono[root_RHD * 3 + 2] * C / s) / fx; //norm gt root 3d (z dimension)
			double xm = (uk[m] - u0) * (scale_norm_zkr[m] + gt_3d_mono[root_RHD * 3 + 2] * C / s) / fx;
			double yn = (vk[n] - v0) * (scale_norm_zkr[n] + gt_3d_mono[root_RHD * 3 + 2] * C / s) / fy;
			double ym = (vk[m] - v0) * (scale_norm_zkr[m] + gt_3d_mono[root_RHD * 3 + 2] * C / s) / fy;
			double zn = scale_norm_zkr[n] + gt_3d_mono[root_RHD * 3 + 2] * C / s;
			double zm = scale_norm_zkr[m] + gt_3d_mono[root_RHD * 3 + 2] * C / s;


			/*pow(, 2) = pow(C, 2)*/
			double pow2 = pow(xn - xm, 2) + pow(yn - ym, 2) + pow(zn - zm, 2) - pow(C, 2);

			//05/07: tested pow2 close to 0.000

			//calc scaled z root
			double scale_norm_zroot_equation_A = pow((uk[n] - uk[m]) / fx, 2) + pow((vk[n] - vk[m]) / fy, 2);
			double scale_norm_zroot_equation_B = 2 * (((uk[n] - uk[m]) / fx) * ((uk[n] - u0) / fx * scale_norm_zkr[n] - (uk[m] - u0) / fx * scale_norm_zkr[m]) +
				((vk[n] - vk[m]) / fy) * ((vk[n] - v0) / fy * scale_norm_zkr[n] - (vk[m] - v0) / fy * scale_norm_zkr[m]));
			double scale_norm_zroot_equation_C = pow((uk[n] - u0) / fx * scale_norm_zkr[n] - (uk[m] - u0) / fx * scale_norm_zkr[m], 2) + pow((vk[n] - v0) / fy * scale_norm_zkr[n] - (vk[m] - v0) / fy * scale_norm_zkr[m], 2) + pow(scale_norm_zkr[n] - scale_norm_zkr[m], 2) - pow(C, 2);

			double scale_norm_zroot_0 = 0.5 * (-scale_norm_zroot_equation_B + sqrt(pow(scale_norm_zroot_equation_B, 2) - 4 * scale_norm_zroot_equation_A * scale_norm_zroot_equation_C)) / scale_norm_zroot_equation_A;
			double scale_norm_zroot_1 = 0.5 * (-scale_norm_zroot_equation_B - sqrt(pow(scale_norm_zroot_equation_B, 2) - 4 * scale_norm_zroot_equation_A * scale_norm_zroot_equation_C)) / scale_norm_zroot_equation_A;

#ifdef print_to_console
			cout << 2 * i + hand << " " << pow2 << " " << "\n";
			cout << "---\n";
			printf("Solving scale normalized zroot\n");
			printf("Root 1 vs Ground truth scale normalized root: %12.6f %12.6f\n", scale_norm_zroot_0, gt_3d_mono[root_RHD * 3 + 2] * C / s);
			printf("Root 2 vs Ground truth scale normalized root: %12.6f %12.6f\n", scale_norm_zroot_1, gt_3d_mono[root_RHD * 3 + 2] * C / s);
			printf("---------------------\n");			
#endif

#ifdef save_calib
			char calib_file_name[maxlen];
			sprintf(calib_file_name, "%s%d%s", calib_prefix, 2 * i + hand, ".txt");
			FILE *fout_calib = fopen(calib_file_name, "w");
			fprintf(fout_calib, "%d %12.6f \n", 2 * i + hand, pow2);
			fprintf(fout_calib, "---\n");
			fprintf(fout_calib, "Solving scale normalized zroot\n");
			fprintf(fout_calib, "Root 1 vs Ground truth scale normalized root: %12.6f %12.6f\n", scale_norm_zroot_0, gt_3d_mono[root_RHD * 3 + 2] * C / s);
			fprintf(fout_calib, "Root 2 vs Ground truth scale normalized root: %12.6f %12.6f\n", scale_norm_zroot_1, gt_3d_mono[root_RHD * 3 + 2] * C / s);
			fprintf(fout_calib, "---------------------\n");
#endif
			
			double scale_norm_zroot = 0.0;
			if (scale_norm_zroot_0 > scale_norm_zroot_1) scale_norm_zroot = scale_norm_zroot_0;

			//Get scale-normalized root -> get scale-normalized absolute 3D pose
			for (int j = 0; j < JointNum_RHD; j++) 
			{
				//add root (since is scale normalized root relative coordinate)
				scale_norm_zk[j] = scale_norm_zroot + scale_norm_zkr[j];
				scale_norm_xk[j] = (uk[j] - u0) * scale_norm_zk[j] / fx;
				scale_norm_yk[j] = (vk[j] - v0) * scale_norm_zk[j] / fy;
			}

			//solving scale
			double sum_of_square_bone_len = 0.0;
			for (int j = 0; j < BoneNum_RHD; j++)
			{
				int u = bones_RHD[j][0], v = bones_RHD[j][1];
				sum_of_square_bone_len += pow(scale_norm_xk[u] - scale_norm_xk[v], 2) + pow(scale_norm_yk[u] - scale_norm_yk[v], 2) + pow(scale_norm_zk[u] - scale_norm_zk[v], 2);
			}

			double solve_scale_argmin_equation_A = sum_of_square_bone_len;
			double solve_scale_argmin_equation_B = 0.0;
			for (int j = 0; j < BoneNum_RHD; j++)
			{
				int u = bones_RHD[j][0], v = bones_RHD[j][1];
				solve_scale_argmin_equation_B += sqrt(pow(scale_norm_xk[u] - scale_norm_xk[v], 2) + pow(scale_norm_yk[u] - scale_norm_yk[v], 2) + pow(scale_norm_zk[u] - scale_norm_zk[v], 2)) * stats_avg_bone[j];
			}
			solve_scale_argmin_equation_B *= -2.0;

			double solve_scale_argmin_equation_C = 0.0;
			for (int j = 0; j < BoneNum_RHD; j++)
			{
				solve_scale_argmin_equation_C += pow(stats_avg_bone[j], 2);
			}

			//-b+-sqrt(b*b-4*a*c) /2a
			//min: might not equal to zero
			//so is b/-2a
			double global_hand_scale = solve_scale_argmin_equation_B / (-2 * solve_scale_argmin_equation_A);

#ifdef print_to_console
			printf("Solving global hand scale which is\n");
			printf("Solved global scale vs real global scale: %12.6f %12.6f\n", global_hand_scale, s);
			printf("---------------------\n");
#endif

#ifdef save_calib
			fprintf(fout_calib, "Solving global hand scale which is\n");
			fprintf(fout_calib, "Solved global scale vs real global scale: %12.6f %12.6f\n", global_hand_scale, s);
			fprintf(fout_calib, "---------------------\n");
#endif

			//next step: test scale to see if the recovered 3D is exactly...
			//add scale-normalized root
			//calcualte scale norm xyz


			//recover scale back using the computed scale
			for (int j = 0; j < JointNum_RHD; j++)
			{
				recover_back_scale_norm_xk[j] = global_hand_scale / C * scale_norm_xk[j];
				recover_back_scale_norm_yk[j] = global_hand_scale / C * scale_norm_yk[j];
				recover_back_scale_norm_zk[j] = global_hand_scale / C * scale_norm_zk[j];
			}

#ifdef print_to_console
			printf("Comparing recovered 3d with ground truth monocular 3D pose\n");
#endif

#ifdef save_calib
			fprintf(fout_calib, "Comparing recovered 3d with ground truth monocular 3D pose\n");
#endif
			//output recovered xyz and mono camera frame global xyz
			double all_joint_err = 0.0;
			for (int j = 0; j < JointNum_RHD; j++)
			{
				double cur_joint_err = 0.0;
#ifdef print_to_console
				printf("%5d predicted: ", 2 * i + hand);
				printf("%12.6f %12.6f %12.6f ", recover_back_scale_norm_xk[j], recover_back_scale_norm_yk[j], recover_back_scale_norm_zk[j]);				
				printf("    ground truth mono 3d: ");
#endif

#ifdef save_calib
				fprintf(fout_calib, "%5d predicted: ", 2 * i + hand);
				fprintf(fout_calib, "%12.6f %12.6f %12.6f ", recover_back_scale_norm_xk[j], recover_back_scale_norm_yk[j], recover_back_scale_norm_zk[j]);
				fprintf(fout_calib, "    ground truth mono 3d: ");
#endif
				for (int k = 0; k < 3; k++) 
				{
#ifdef print_to_console
					printf("%12.6f ", gt_3d_mono[j * 3 + k]);
#endif

#ifdef save_calib
					fprintf(fout_calib, "%12.6f ", gt_3d_mono[j * 3 + k]);
#endif
				}
				cur_joint_err += pow(recover_back_scale_norm_xk[j] - gt_3d_mono[j * 3], 2) + pow(recover_back_scale_norm_yk[j] - gt_3d_mono[j * 3 + 1], 2) + pow(recover_back_scale_norm_zk[j] - gt_3d_mono[j * 3 + 2], 2);
				cur_joint_err = sqrt(cur_joint_err);
#ifdef print_to_console
				printf(" %12.6f \n", cur_joint_err);
#endif

#ifdef save_calib
				fprintf(fout_calib, " %12.6f \n", cur_joint_err);
#endif
				all_joint_err += cur_joint_err;
#ifdef print_to_console
				printf("\n");
#endif

#ifdef save_calib
				fprintf(fout_calib, "\n");
#endif
			}
			all_joint_err /= double(JointNum_RHD);
#ifdef print_to_console
			printf("All joint error: %12.6f\n", all_joint_err);
			printf("---------------------\n");
#endif

#ifdef save_calib
			fprintf(fout_calib, "All joint error: %12.6f\n", all_joint_err);
			fprintf(fout_calib, "---------------------\n");
#endif
			
#ifdef save_calib
			fclose(fout_calib);
#endif
			if ((2 * i + hand) % 100 == 0) cout << 2 * i + hand << "\n";

		}
	}
	return 0;
}