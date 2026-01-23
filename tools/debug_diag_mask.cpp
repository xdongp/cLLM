/**
 * @file debug_diag_mask.cpp
 * @brief 调试 ggml_diag_mask_inf 的行为
 * 
 * 对比批量推理和增量推理时 causal mask 的差异
 */

#include <ggml.h>
#include <ggml-cpu.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

void printTensor2D(const char* name, ggml_tensor* t, size_t maxRows = 5, size_t maxCols = 5) {
    if (!t || !t->data) {
        std::cout << name << ": NULL" << std::endl;
        return;
    }
    
    std::cout << name << " [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3] << "]:" << std::endl;
    
    const float* data = static_cast<const float*>(t->data);
    size_t ne0 = t->ne[0];
    size_t ne1 = t->ne[1];
    size_t nb0 = t->nb[0] / sizeof(float);
    size_t nb1 = t->nb[1] / sizeof(float);
    
    for (size_t j = 0; j < std::min(ne1, maxCols); ++j) {
        std::cout << "  col " << j << ": [";
        for (size_t i = 0; i < std::min(ne0, maxRows); ++i) {
            float v = data[i * nb0 + j * nb1];
            if (std::isinf(v)) {
                std::cout << (v < 0 ? "-inf" : "+inf");
            } else {
                std::cout << std::fixed << std::setprecision(4) << v;
            }
            if (i < std::min(ne0, maxRows) - 1) std::cout << ", ";
        }
        if (ne0 > maxRows) std::cout << " ...";
        std::cout << "]" << std::endl;
    }
    if (ne1 > maxCols) std::cout << "  ..." << std::endl;
}

void testDiagMask() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing ggml_diag_mask_inf behavior" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 创建 GGML 上下文
    size_t memSize = 64 * 1024 * 1024;  // 64MB
    struct ggml_init_params params = {
        .mem_size   = memSize,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    ggml_context* ctx = ggml_init(params);
    
    // ========== 测试 1: 批量推理场景 ==========
    std::cout << "\n=== Test 1: Batch Inference (seqLen=3, n_past=0) ===" << std::endl;
    {
        // scores: [totalLen=3, seqLen=3, nHeads=1]
        ggml_tensor* scores_batch = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 3, 1);
        
        // 初始化为简单的值（用于调试）
        float* data = static_cast<float*>(scores_batch->data);
        for (int i = 0; i < 9; ++i) {
            data[i] = 1.0f;  // 全1
        }
        
        std::cout << "Before mask:" << std::endl;
        printTensor2D("scores_batch", scores_batch);
        
        // 应用 diag_mask_inf (n_past=0)
        ggml_tensor* masked_batch = ggml_diag_mask_inf(ctx, scores_batch, 0);
        
        // 构建并执行图
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, masked_batch);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
        std::cout << "After mask (n_past=0):" << std::endl;
        printTensor2D("masked_batch", masked_batch);
        
        // 应用 softmax
        ggml_tensor* attn_batch = ggml_soft_max(ctx, masked_batch);
        
        ggml_cgraph* graph2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph2, attn_batch);
        ggml_graph_compute_with_ctx(ctx, graph2, 1);
        
        std::cout << "After softmax:" << std::endl;
        printTensor2D("attn_batch", attn_batch);
        
        // 打印 query 2 (最后一列) 的注意力权重
        const float* attn_data = static_cast<const float*>(attn_batch->data);
        std::cout << "\nQuery 2 (col 2) attention weights: [";
        for (int i = 0; i < 3; ++i) {
            std::cout << std::fixed << std::setprecision(4) << attn_data[i * 1 + 2 * 3];
            if (i < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // ========== 测试 2: 增量推理场景 ==========
    std::cout << "\n=== Test 2: Incremental Inference (seqLen=1, n_past=2) ===" << std::endl;
    {
        // scores: [totalLen=3, seqLen=1, nHeads=1]
        ggml_tensor* scores_incr = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, 1);
        
        float* data = static_cast<float*>(scores_incr->data);
        for (int i = 0; i < 3; ++i) {
            data[i] = 1.0f;  // 全1
        }
        
        std::cout << "Before mask:" << std::endl;
        printTensor2D("scores_incr", scores_incr);
        
        // 应用 diag_mask_inf (n_past=2)
        ggml_tensor* masked_incr = ggml_diag_mask_inf(ctx, scores_incr, 2);
        
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, masked_incr);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
        std::cout << "After mask (n_past=2):" << std::endl;
        printTensor2D("masked_incr", masked_incr);
        
        // 应用 softmax
        ggml_tensor* attn_incr = ggml_soft_max(ctx, masked_incr);
        
        ggml_cgraph* graph2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph2, attn_incr);
        ggml_graph_compute_with_ctx(ctx, graph2, 1);
        
        std::cout << "After softmax:" << std::endl;
        printTensor2D("attn_incr", attn_incr);
        
        // 打印 query 0 的注意力权重
        const float* attn_data = static_cast<const float*>(attn_incr->data);
        std::cout << "\nQuery 0 (col 0) attention weights: [";
        for (int i = 0; i < 3; ++i) {
            std::cout << std::fixed << std::setprecision(4) << attn_data[i];
            if (i < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // ========== 测试 3: 检查 GGML 的 softmax 维度 ==========
    std::cout << "\n=== Test 3: Check softmax dimension ===" << std::endl;
    {
        // 创建一个 [3, 2] 的张量
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
        float* data = static_cast<float*>(t->data);
        // 第一列: [1, 2, 3]
        data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;
        // 第二列: [4, 5, 6]
        data[3] = 4.0f; data[4] = 5.0f; data[5] = 6.0f;
        
        std::cout << "Input [3, 2]:" << std::endl;
        printTensor2D("t", t);
        
        ggml_tensor* sm = ggml_soft_max(ctx, t);
        
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sm);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        
        std::cout << "After softmax:" << std::endl;
        printTensor2D("sm", sm);
        
        // 验证：softmax 应该在 ne[0] 维度上做
        const float* sm_data = static_cast<const float*>(sm->data);
        float sum_col0 = sm_data[0] + sm_data[1] + sm_data[2];
        float sum_col1 = sm_data[3] + sm_data[4] + sm_data[5];
        
        std::cout << "Sum of col 0: " << sum_col0 << " (should be 1.0)" << std::endl;
        std::cout << "Sum of col 1: " << sum_col1 << " (should be 1.0)" << std::endl;
    }
    
    ggml_free(ctx);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Analysis Complete" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main() {
    testDiagMask();
    return 0;
}
