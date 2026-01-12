#!/usr/bin/env python3
"""
Comprehensive patch for dnn_backend_torch.cpp to enable zero-copy CUDA frame processing.

This patch:
1. Fixes the input path to properly handle CUDA frames without memory leaks
2. Adds zero-copy output path for CUDA output frames
3. Adds necessary includes and structures
"""
import sys
import re

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_dnn_backend_torch.cpp>", file=sys.stderr)
    sys.exit(1)

content = open(sys.argv[1]).read()
patches_applied = []

# 1. Add cuda_runtime.h include if not present (for cudaMemcpy)
if "#include <cuda_runtime.h>" not in content:
    # Add after torch/script.h
    old = '#include <torch/script.h>'
    new = '''#include <torch/script.h>
#include <cuda_runtime.h>'''
    if old in content:
        content = content.replace(old, new, 1)
        patches_applied.append("Added cuda_runtime.h include")

# 2. Fix the input path - the current code has issues:
#    - Memory allocated for input.data before the CUDA check (memory leak)
#    - Need to restructure to avoid allocation when using zero-copy
#
# Find the fill_model_input_th function and restructure it

# Look for the problematic section where we allocate memory before checking for CUDA
old_alloc_section = '''    input.dims[height_idx] = task->in_frame->height;
    input.dims[width_idx] = task->in_frame->width;
    input.data = av_malloc(input.dims[height_idx] * input.dims[width_idx] *
                           input.dims[channel_idx] * sizeof(float));
    if (!input.data)
        return AVERROR(ENOMEM);
    infer_request->input_tensor = new torch::Tensor();
    infer_request->output = new torch::Tensor();

    // Check for CUDA hardware frames (zero-copy path)
    if (task->in_frame->hw_frames_ctx && task->in_frame->format == AV_PIX_FMT_CUDA) {
        CUdeviceptr cuda_ptr = (CUdeviceptr)task->in_frame->data[0];

        if (cuda_ptr) {
            av_log(ctx, AV_LOG_INFO, "CUDA frame detected - using zero-copy path\\n");

            // Create tensor directly from CUDA memory
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            *infer_request->input_tensor = torch::from_blob(
                (void*)cuda_ptr,
                {1, input.dims[channel_idx], input.dims[height_idx], input.dims[width_idx]},
                options
            );

            return 0;
        }
    }

    switch (th_model->model.func_type) {'''

new_alloc_section = '''    input.dims[height_idx] = task->in_frame->height;
    input.dims[width_idx] = task->in_frame->width;

    infer_request->input_tensor = new torch::Tensor();
    infer_request->output = new torch::Tensor();

    // Check for CUDA hardware frames (zero-copy input path)
    if (task->in_frame->hw_frames_ctx && task->in_frame->format == AV_PIX_FMT_CUDA) {
        AVHWFramesContext *hw_frames = (AVHWFramesContext *)task->in_frame->hw_frames_ctx->data;
        CUdeviceptr cuda_ptr = (CUdeviceptr)task->in_frame->data[0];
        int linesize = task->in_frame->linesize[0];
        int height = task->in_frame->height;
        int width = task->in_frame->width;

        if (cuda_ptr) {
            av_log(ctx, AV_LOG_DEBUG, "CUDA frame detected (sw_format=%s) - using zero-copy input\\n",
                   av_get_pix_fmt_name(hw_frames->sw_format));

            // For RGB24 sw_format, the data is packed as HWC (height, width, channels)
            // We need to convert to CHW format for the model
            // Create tensor options for CUDA
            auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

            // Create tensor from CUDA memory - data is in HWC format (packed RGB)
            torch::Tensor input_hwc = torch::from_blob(
                (void*)cuda_ptr,
                {height, width, 3},
                {linesize, 3, 1},  // strides: linesize for rows, 3 for columns, 1 for channels
                options
            );

            // Convert HWC -> CHW and normalize to float [0,1]
            *infer_request->input_tensor = input_hwc.permute({2, 0, 1})  // HWC -> CHW
                                                     .unsqueeze(0)       // Add batch dim: CHW -> NCHW
                                                     .to(torch::kFloat32)
                                                     .div(255.0);        // Normalize to [0,1]

            return 0;
        }
    }

    // Standard CPU path - allocate memory for input data
    input.data = av_malloc(input.dims[height_idx] * input.dims[width_idx] *
                           input.dims[channel_idx] * sizeof(float));
    if (!input.data)
        return AVERROR(ENOMEM);

    switch (th_model->model.func_type) {'''

if old_alloc_section in content:
    content = content.replace(old_alloc_section, new_alloc_section)
    patches_applied.append("Fixed input path for zero-copy CUDA frames")

# 3. Update the output path in infer_completion_callback
#    Need to handle zero-copy output to CUDA frames
old_output_section = '''    switch (th_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            // Post process can only deal with CPU memory.
            if (output->device() != torch::kCPU)
                *output = output->to(torch::kCPU);
            outputs.scale = 255;
            outputs.data = output->data_ptr();
            if (th_model->model.frame_post_proc != NULL) {
                th_model->model.frame_post_proc(task->out_frame, &outputs, th_model->model.filter_ctx);
            } else {
                ff_proc_from_dnn_to_frame(task->out_frame, &outputs, th_model->ctx);
            }
        } else {
            task->out_frame->width = outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];
            task->out_frame->height = outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
        }
        break;'''

new_output_section = '''    switch (th_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        // Check for CUDA output frames (zero-copy output path)
        if (task->out_frame->hw_frames_ctx && task->out_frame->format == AV_PIX_FMT_CUDA) {
            CUdeviceptr out_cuda_ptr = (CUdeviceptr)task->out_frame->data[0];
            int out_linesize = task->out_frame->linesize[0];
            int out_height = outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
            int out_width = outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];

            if (out_cuda_ptr) {
                av_log(th_model->ctx, AV_LOG_DEBUG, "Zero-copy output to CUDA frame %dx%d\\n",
                       out_width, out_height);

                // Ensure output tensor is on CUDA and contiguous
                if (!output->is_cuda()) {
                    *output = output->to(torch::kCUDA);
                }

                // Model output is NCHW float [0,1], need to convert to HWC uint8 [0,255]
                // Remove batch dim, convert CHW -> HWC, scale to [0,255], convert to uint8
                torch::Tensor output_hwc = output->squeeze(0)        // NCHW -> CHW
                                                  .permute({1, 2, 0}) // CHW -> HWC
                                                  .mul(255.0)
                                                  .clamp(0, 255)
                                                  .to(torch::kUInt8)
                                                  .contiguous();

                // Copy to output frame with proper stride handling
                // If linesize matches width*3, we can do a single copy
                if (out_linesize == out_width * 3) {
                    cudaMemcpy((void*)out_cuda_ptr, output_hwc.data_ptr(),
                               out_height * out_width * 3, cudaMemcpyDeviceToDevice);
                } else {
                    // Copy row by row to handle stride mismatch
                    for (int y = 0; y < out_height; y++) {
                        cudaMemcpy((void*)(out_cuda_ptr + y * out_linesize),
                                   (char*)output_hwc.data_ptr() + y * out_width * 3,
                                   out_width * 3, cudaMemcpyDeviceToDevice);
                    }
                }

                task->out_frame->width = out_width;
                task->out_frame->height = out_height;
                break;
            }
        }

        // Standard CPU output path
        if (task->do_ioproc) {
            // Post process can only deal with CPU memory.
            if (output->device() != torch::kCPU)
                *output = output->to(torch::kCPU);  // This is the expensive GPU->CPU copy
            outputs.scale = 255;
            outputs.data = output->data_ptr();
            if (th_model->model.frame_post_proc != NULL) {
                th_model->model.frame_post_proc(task->out_frame, &outputs, th_model->model.filter_ctx);
            } else {
                ff_proc_from_dnn_to_frame(task->out_frame, &outputs, th_model->ctx);
            }
        } else {
            task->out_frame->width = outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];
            task->out_frame->height = outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
        }
        break;'''

if old_output_section in content:
    content = content.replace(old_output_section, new_output_section)
    patches_applied.append("Added zero-copy output path for CUDA frames")

# Output
if patches_applied:
    print(content)
    print(f"Applied patches: {', '.join(patches_applied)}", file=sys.stderr)
else:
    print("No patches applied - file may already be patched or has unexpected format", file=sys.stderr)
    print("Checking for partial matches...", file=sys.stderr)

    # Debug: show what we're looking for vs what's in the file
    if "// Check for CUDA hardware frames (zero-copy path)" in content:
        print("Found old zero-copy comment - patch may be partially applied", file=sys.stderr)
    if "// Check for CUDA hardware frames (zero-copy input path)" in content:
        print("Found new zero-copy input comment - input already patched", file=sys.stderr)
    if "// Check for CUDA output frames (zero-copy output path)" in content:
        print("Found zero-copy output comment - output already patched", file=sys.stderr)

    sys.exit(1)
