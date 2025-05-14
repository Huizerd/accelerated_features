#!/usr/bin/env python3
# This script was converted from Jupyter notebook xfeat_matching_end2end_onnxruntime.ipynb

import numpy as np
import os
import onnxruntime as ort
import tqdm
import cv2
import matplotlib.pyplot as plt
import time

# Path to the ONNX model
model_path = './xfeat_matching.onnx'    # python ./export.py --dynamic --export_path ./xfeat_matching.onnx

# Load example images
im1 = cv2.imread('./assets/ref.png', cv2.IMREAD_COLOR)
im2 = cv2.imread('./assets/tgt.png', cv2.IMREAD_COLOR)
size = 1024
im1 = cv2.resize(im1, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
im2 = cv2.resize(im2, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

# Simple function that fits an homography in a set of matches and draw the homography transform
def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

def main():
    # Initialize ONNX Runtime session to check model info
    tmp_ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Print the input and output names and shapes
    print("Model information:")
    for i in range(len(tmp_ort_session.get_inputs())):
        print(f"Input name: {tmp_ort_session.get_inputs()[i].name}, shape: {tmp_ort_session.get_inputs()[i].shape}")
    for i in range(len(tmp_ort_session.get_outputs())):
        print(f"Output name: {tmp_ort_session.get_outputs()[i].name}, shape: {tmp_ort_session.get_outputs()[i].shape}")

    # Define providers for inference
    providers = [
        # The TensorrtExecutionProvider is the fastest.
        # ('TensorrtExecutionProvider', { 
        #     'device_id': 0,
        #     'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
        #     'trt_fp16_enable': True,
        #     'trt_engine_cache_enable': True,
        #     'trt_engine_cache_path': './trt_engine_cache',
        #     'trt_engine_cache_prefix': 'model',
        #     'trt_dump_subgraphs': False,
        #     'trt_timing_cache_enable': True,
        #     'trt_timing_cache_path': './trt_engine_cache',
        #     #'trt_builder_optimization_level': 3,
        # }),

        # The CUDAExecutionProvider is slower than PyTorch, 
        # possibly due to performance issues with large matrix multiplication "cossim = torch.bmm(feats1, feats2.permute(0,2,1))"
        # Reducing the top_k value when exporting to ONNX can decrease the matrix size.
        # ('CUDAExecutionProvider', { 
        #     'device_id': 0,
        #     'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
        # }),
        ('CPUExecutionProvider',{})
    ]
    
    # Create the ONNX Runtime session with the specified providers
    ort_session = ort.InferenceSession(model_path, providers=providers)

    # Prepare the input tensor
    # TODO: proper resize
    # im1 = cv2.resize(im1, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
    # im2 = cv2.resize(im2, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)

    input_array_1 = im1.transpose(2, 0, 1).astype(np.float32)
    input_array_1 = np.expand_dims(input_array_1, axis=0)
    input_array_2 = im2.transpose(2, 0, 1).astype(np.float32)
    input_array_2 = np.expand_dims(input_array_2, axis=0)

    batch_size = 1

    # Pseudo-batch the input images
    input_array_1 = np.concatenate([input_array_1 for _ in range(batch_size)], axis=0)
    input_array_2 = np.concatenate([input_array_2 for _ in range(batch_size)], axis=0)

    inputs = {
        ort_session.get_inputs()[0].name: input_array_1,
        ort_session.get_inputs()[1].name: input_array_2
    }

    # Run matching
    print("Running initial inference...")
    outputs = ort_session.run(None, inputs)

    # Validate the outputs of the pseudo-batched inputs
    matches = outputs[0]
    batch_indexes = outputs[1]

    matches_0 = matches[batch_indexes == 0]
    valid = []
    for i in range(1, input_array_1.shape[0]):
        valid.append(np.all(matches_0 == matches[batch_indexes == i]))
    print(f"All batched outputs equal: {valid}")

    # Benchmark performance
    print("Benchmarking performance over 100 runs...")
    times = []
    for i in tqdm.tqdm(range(100)):
        start = time.time()
        outputs = ort_session.run(None, inputs)
        times.append(time.time() - start)

    print(f"Average time per batch: {np.mean(times):.4f} seconds")
    print(f"Average time per image: {np.mean(times)/batch_size:.4f} seconds")
    print(f"Average FPS per image: {batch_size/np.mean(times):.4f}")

    # Draw the matches
    print("Drawing matches...")
    matches = outputs[0]
    batch_indexes = outputs[1]
    mkpts_0, mkpts_1 = matches[batch_indexes == 0][..., :2], matches[batch_indexes == 0][..., 2:]

    canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
    
    # Show and save the visualization
    plt.figure(figsize=(12,12))
    plt.imshow(canvas[..., ::-1])
    plt.title("Feature Matches with Homography")
    plt.savefig("matching_visualization.png")
    plt.show()

if __name__ == "__main__":
    main()
