#!/usr/bin/env python3
"""
Patch vf_dnn_processing.c to add CUDA frame support for zero-copy GPU inference.

This enables the pipeline: hwupload_cuda -> dnn_processing -> hwdownload
avoiding expensive GPU<->CPU memory transfers.
"""
import sys
import re

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_vf_dnn_processing.c>", file=sys.stderr)
    sys.exit(1)

content = open(sys.argv[1]).read()

# Track what patches were applied
patches_applied = []

# 1. Add hwcontext includes after existing includes
old_includes = '#include "libswscale/swscale.h"'
new_includes = '''#include "libswscale/swscale.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda.h"'''

if old_includes in content and "hwcontext_cuda.h" not in content:
    content = content.replace(old_includes, new_includes)
    patches_applied.append("Added hwcontext includes")

# 2. Extend DnnProcessingContext struct with CUDA fields
old_struct = '''typedef struct DnnProcessingContext {
    const AVClass *class;
    DnnContext dnnctx;
    struct SwsContext *sws_uv_scale;
    int sws_uv_height;
} DnnProcessingContext;'''

new_struct = '''typedef struct DnnProcessingContext {
    const AVClass *class;
    DnnContext dnnctx;
    struct SwsContext *sws_uv_scale;
    int sws_uv_height;
    AVBufferRef *hw_device_ctx;    // CUDA device context
    AVBufferRef *hw_frames_ctx;    // CUDA frames context for output
    int use_cuda;                  // Flag: using CUDA frames
} DnnProcessingContext;'''

if old_struct in content:
    content = content.replace(old_struct, new_struct)
    patches_applied.append("Extended DnnProcessingContext with CUDA fields")

# 3. Add AV_PIX_FMT_CUDA to pix_fmts array
old_pix_fmts = '''static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};'''

new_pix_fmts = '''static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_CUDA,  // CUDA hardware frames for zero-copy GPU inference
    AV_PIX_FMT_NONE
};'''

if old_pix_fmts in content:
    content = content.replace(old_pix_fmts, new_pix_fmts)
    patches_applied.append("Added AV_PIX_FMT_CUDA to pix_fmts")

# 4. Replace config_input to handle CUDA frames
old_config_input = '''static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *context     = inlink->dst;
    DnnProcessingContext *ctx = context->priv;
    int result;
    DNNData model_input;
    int check;

    result = ff_dnn_get_input(&ctx->dnnctx, &model_input);
    if (result != 0) {
        av_log(ctx, AV_LOG_ERROR, "could not get input from the model\\n");
        return result;
    }

    check = check_modelinput_inlink(&model_input, inlink);
    if (check != 0) {
        return check;
    }

    return 0;
}'''

new_config_input = '''static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *context     = inlink->dst;
    DnnProcessingContext *ctx = context->priv;
    int result;
    DNNData model_input;
    int check;

    // Check if input is CUDA hardware frames
    if (inlink->format == AV_PIX_FMT_CUDA) {
        AVHWFramesContext *hw_frames;

        if (!inlink->hw_frames_ctx) {
            av_log(context, AV_LOG_ERROR, "CUDA format requires hw_frames_ctx\\n");
            return AVERROR(EINVAL);
        }

        hw_frames = (AVHWFramesContext *)inlink->hw_frames_ctx->data;
        ctx->hw_device_ctx = av_buffer_ref(hw_frames->device_ref);
        if (!ctx->hw_device_ctx) {
            return AVERROR(ENOMEM);
        }

        ctx->use_cuda = 1;
        av_log(context, AV_LOG_INFO, "Using CUDA frames for zero-copy DNN inference\\n");

        // For CUDA frames, check the underlying software format
        if (hw_frames->sw_format != AV_PIX_FMT_RGB24 &&
            hw_frames->sw_format != AV_PIX_FMT_BGR24 &&
            hw_frames->sw_format != AV_PIX_FMT_RGBF32 &&
            hw_frames->sw_format != AV_PIX_FMT_0RGB32) {
            av_log(context, AV_LOG_WARNING,
                   "CUDA frame sw_format is %s, model may expect RGB format\\n",
                   av_get_pix_fmt_name(hw_frames->sw_format));
        }

        return 0;  // Skip standard model input check for CUDA frames
    }

    ctx->use_cuda = 0;

    result = ff_dnn_get_input(&ctx->dnnctx, &model_input);
    if (result != 0) {
        av_log(ctx, AV_LOG_ERROR, "could not get input from the model\\n");
        return result;
    }

    check = check_modelinput_inlink(&model_input, inlink);
    if (check != 0) {
        return check;
    }

    return 0;
}'''

if old_config_input in content:
    content = content.replace(old_config_input, new_config_input)
    patches_applied.append("Updated config_input for CUDA frames")

# 5. Replace config_output to handle CUDA output frames
old_config_output = '''static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *context = outlink->src;
    DnnProcessingContext *ctx = context->priv;
    int result;
    AVFilterLink *inlink = context->inputs[0];

    // have a try run in case that the dnn model resize the frame
    result = ff_dnn_get_output(&ctx->dnnctx, inlink->w, inlink->h, &outlink->w, &outlink->h);
    if (result != 0) {
        av_log(ctx, AV_LOG_ERROR, "could not get output from the model\\n");
        return result;
    }

    prepare_uv_scale(outlink);

    return 0;
}'''

new_config_output = '''static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *context = outlink->src;
    DnnProcessingContext *ctx = context->priv;
    int result;
    AVFilterLink *inlink = context->inputs[0];

    // have a try run in case that the dnn model resize the frame
    result = ff_dnn_get_output(&ctx->dnnctx, inlink->w, inlink->h, &outlink->w, &outlink->h);
    if (result != 0) {
        av_log(ctx, AV_LOG_ERROR, "could not get output from the model\\n");
        return result;
    }

    // For CUDA frames, set up output hw_frames_ctx
    if (ctx->use_cuda) {
        AVHWFramesContext *out_frames;

        ctx->hw_frames_ctx = av_hwframe_ctx_alloc(ctx->hw_device_ctx);
        if (!ctx->hw_frames_ctx) {
            return AVERROR(ENOMEM);
        }

        out_frames = (AVHWFramesContext *)ctx->hw_frames_ctx->data;
        out_frames->format = AV_PIX_FMT_CUDA;
        out_frames->sw_format = AV_PIX_FMT_RGB24;  // DNN output format
        out_frames->width = outlink->w;
        out_frames->height = outlink->h;
        out_frames->initial_pool_size = 4;

        result = av_hwframe_ctx_init(ctx->hw_frames_ctx);
        if (result < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to init CUDA output frames context\\n");
            av_buffer_unref(&ctx->hw_frames_ctx);
            return result;
        }

        outlink->hw_frames_ctx = av_buffer_ref(ctx->hw_frames_ctx);
        if (!outlink->hw_frames_ctx) {
            return AVERROR(ENOMEM);
        }

        av_log(context, AV_LOG_INFO, "CUDA output frames: %dx%d\\n", outlink->w, outlink->h);
        return 0;
    }

    prepare_uv_scale(outlink);

    return 0;
}'''

if old_config_output in content:
    content = content.replace(old_config_output, new_config_output)
    patches_applied.append("Updated config_output for CUDA frames")

# 6. Update uninit to clean up CUDA contexts
old_uninit = '''static av_cold void uninit(AVFilterContext *ctx)
{
    DnnProcessingContext *context = ctx->priv;

    sws_freeContext(context->sws_uv_scale);
    ff_dnn_uninit(&context->dnnctx);
}'''

new_uninit = '''static av_cold void uninit(AVFilterContext *ctx)
{
    DnnProcessingContext *context = ctx->priv;

    sws_freeContext(context->sws_uv_scale);
    av_buffer_unref(&context->hw_frames_ctx);
    av_buffer_unref(&context->hw_device_ctx);
    ff_dnn_uninit(&context->dnnctx);
}'''

if old_uninit in content:
    content = content.replace(old_uninit, new_uninit)
    patches_applied.append("Updated uninit for CUDA cleanup")

# Output result
if patches_applied:
    print(content)
    print(f"Applied patches: {', '.join(patches_applied)}", file=sys.stderr)
else:
    print("No patches applied - file may already be patched or has unexpected format", file=sys.stderr)
    sys.exit(1)
