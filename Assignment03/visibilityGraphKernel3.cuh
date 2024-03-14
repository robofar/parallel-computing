#ifndef VISIBILITY_GRAPH_GPU_HEADER
#define VISIBILITY_GRAPH_GPU_HEADER

#include "visibilityGraph.cuh"


__global__
void visibilityGraphKernel(SuperPoint* vertices, Line* obstacles, int n, int m, int* indices) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

    int gridDimY = gridDim.y;
    int blockDimX = blockDim.x;

    int threadIndex = i*gridDimY*blockDimX + j*blockDimX + k;
    
    if(i<n && j<n && k<m) {
        if(VisibilityGraph::concaveVertex(vertices[i]) == false) {
            if(j > i) {
                if(VisibilityGraph::concaveVertex(vertices[j]) == false) {
                    bool direction1 = VisibilityGraph::tangentialEdge(vertices[i], vertices[j]);
                    bool direction2 = VisibilityGraph::tangentialEdge(vertices[j], vertices[i]);
                    if((direction1==true) && (direction2==true)) {
                        Line visibilityGraphEdge = {vertices[i].current, vertices[j].current};
                        bool isVisible = true;
                        if(VisibilityGraph::doIntersect(visibilityGraphEdge, obstacles[k])) isVisible=false;

                        if(isVisible) {
                            indices[threadIndex] = 1;
                            // in this approach we cannot use this (because visGraph is shape n*n and indices n*n*m)
                            //visibilityGraph[threadIndex] = visibilityGraphEdge;
                        }
                        
                    }
                }
            }
        }
    }
}



#endif // VISIBILITY_GRAPH_GPU_HEADER