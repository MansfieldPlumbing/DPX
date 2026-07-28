#include "sys_types.h"
#include "sys_graph_orchestrator.h"
#include <vector>
#include <cstring>

void cpu_concat(DpxGpuTensor** inputs, int num_inputs, DpxGpuTensor& out, int axis) {
    if (out.cpu_data && num_inputs > 0) {
        int rank = inputs[0]->shape.size();
        if (rank == 0) return;
        if (axis < 0) axis += rank;

        // C++ Shape Inference
        if (out.shape.empty()) {
            out.shape = inputs[0]->shape;
            int sum_axis = 0;
            for (int i = 0; i < num_inputs; i++) {
                if (inputs[i]->shape.size() > axis) {
                    sum_axis += inputs[i]->shape[axis];
                }
            }
            if (out.shape.size() > axis) out.shape[axis] = sum_axis;
        }

        long blocks = 1;
        for (int k = 0; k < axis; k++) {
            blocks *= (inputs[0]->shape[k] > 0 ? inputs[0]->shape[k] : 1);
        }

        long out_str_axis = 1;
        for (int k = axis + 1; k < rank; k++) {
            out_str_axis *= (out.shape[k] > 0 ? out.shape[k] : 1);
        }

        float* dst = (float*)out.cpu_data;
        long off_axis = 0;

        for (int i = 0; i < num_inputs; i++) {
            auto* t = inputs[i];
            if (!t->cpu_data) continue;
            
            long current_axis_dim = (t->shape.size() > axis && t->shape[axis] > 0) ? t->shape[axis] : 1;
            
            float* src = (float*)t->cpu_data;
            long in_row = current_axis_dim * out_str_axis;
            
            for (long bl = 0; bl < blocks; bl++) {
                memcpy(dst + bl * (out.shape[axis] * out_str_axis) + off_axis * out_str_axis, 
                       src + bl * in_row, 
                       in_row * sizeof(float));
            }
            off_axis += current_axis_dim;
        }
    }
}
