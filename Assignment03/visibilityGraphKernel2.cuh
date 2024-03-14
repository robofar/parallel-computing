#ifndef VISIBILITY_GRAPH_GPU_HEADER
#define VISIBILITY_GRAPH_GPU_HEADER

#include "visibilityGraph.cuh"


__global__
void visibilityGraphKernel(SuperPoint* vertices, Line* obstacles, Line* visibilityGraph, int n, int m, int* indices) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int gridDimY = gridDim.y;
    int threadIndex = i*gridDimY + j;
    
    if(i<n && j<n) {
        if(VisibilityGraph::concaveVertex(vertices[i]) == false) {
            if(j > i) {
                if(VisibilityGraph::concaveVertex(vertices[j]) == false) {
                    bool direction1 = VisibilityGraph::tangentialEdge(vertices[i], vertices[j]);
                    bool direction2 = VisibilityGraph::tangentialEdge(vertices[j], vertices[i]);
                    if((direction1==true) && (direction2==true)) {
                        Line visibilityGraphEdge = {vertices[i].current, vertices[j].current};

                        bool isVisible = true;
                        for(int k=0 ; k<m ; k++) {
                            if(VisibilityGraph::doIntersect(visibilityGraphEdge, obstacles[k])) {
                                isVisible = false;
                                break;
                            }
                        }

                        if(isVisible) {
                            // push_back imitation
                            indices[threadIndex] = 1;
                            visibilityGraph[threadIndex] = visibilityGraphEdge;
                        }
                        
                    }
                }
            }
        }
    }
}
















#endif // VISIBILITY_GRAPH_GPU_HEADER