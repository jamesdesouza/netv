#!/usr/bin/env python3
"""
Patch dnn_backend_torch.cpp to add zero-copy CUDA input path.
"""
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_dnn_backend_torch.cpp>", file=sys.stderr)
    sys.exit(1)

content = open(sys.argv[1]).read()

# Find the allocation section and add CUDA check before it
old_section = '''    input.dims[height_idx] = task->in_frame->height;
    input.dims[width_idx] = task->in_frame->width;
    input.data = av_malloc(input.dims[height_idx] * input.dims[width_idx] *
                           input.dims[channel_idx] * sizeof(float));
    if (!input.data)
        return AVERROR(ENOMEM);
    infer_request->input_tensor = new torch::Tensor();
    infer_request->output = new torch::Tensor();

    switch (th_model->model.func_type) {'''

new_section = '''    input.dims[height_idx] = task->in_frame->height;
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
            av_log(ctx, AV_LOG_DEBUG, "CUDA input frame detected (sw_format=%s) - using zero-copy\\n",
                   av_get_pix_fmt_name(hw_frames->sw_format));

            // For RGB24 sw_format, the data is packed as HWC (height, width, channels)
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

if old_section in content:
    content = content.replace(old_section, new_section)
    print(content)
    print("Applied CUDA input path patch", file=sys.stderr)
else:
    print("Pattern not found!", file=sys.stderr)
    sys.exit(1)
