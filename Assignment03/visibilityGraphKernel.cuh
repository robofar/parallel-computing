#ifndef VISIBILITY_GRAPH_GPU_HEADER
#define VISIBILITY_GRAPH_GPU_HEADER

#include "visibilityGraph.cuh"


__global__
void visibilityGraphKernel(SuperPoint* vertices, Line* obstacles, Line* visibilityGraph, int n, int m, int* realSize) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
    
    if(i<n && j<n && k<m) {
        if(VisibilityGraph::concaveVertex(vertices[i]) == false) {
            if(j > i) {
                if(VisibilityGraph::concaveVertex(vertices[j]) == false) {
                    bool direction1 = VisibilityGraph::tangentialEdge(vertices[i], vertices[j]);
                    bool direction2 = VisibilityGraph::tangentialEdge(vertices[j], vertices[i]);
                    if(direction1==true && direction2==true) {
                        Line visibilityGraphEdge = {vertices[i].current, vertices[j].current};

                        bool isVisible = true;
                        if(VisibilityGraph::doIntersect(visibilityGraphEdge, obstacles[k])) isVisible=false;
                        
                        if(isVisible) {
                            // push_back imitation
                            // int index = atomicAdd(realSize, 1);
                            visibilityGraph[*realSize] = visibilityGraphEdge;
                            *realSize = ((*realSize) + 1);
                        }
                        
                    }
                }
            }
        }
    }
}
















#endif // VISIBILITY_GRAPH_GPU_HEADER