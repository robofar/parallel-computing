#ifndef A_STAR_KERNEL_HEADER
#define A_STAR_KERNEL_HEADER

#include "nodeClass2.cuh"


__global__
void iterateNodeNeighbors(Node** neighbors, Node* current, Node* goal, int size, Node** openSet) {
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("%d %d\n", threadIndex, size);
    if (threadIndex < size) {
        Node* neighbor = *(neighbors + threadIndex); // neighbors_pointer[threadIndex];
        double tentative_gScore = (*current).gScore + calculateDistance(*current, *neighbor);
        bool worsePath = (tentative_gScore >= (*neighbor).gScore);

        if((*neighbor).inClosedSet && worsePath) {
        }

        else if((*neighbor).inOpenSet && worsePath) {
        }

        else {
            (*neighbor).inClosedSet = false;
            (*neighbor).updateScores(tentative_gScore, calculateHeuristic(*neighbor, *goal));
            (*neighbor).inOpenSet = true;
            (*neighbor).parent = current;

            // (*openSet).push(neighbor);
            openSet[threadIndex] = neighbor; // save address
        }
    }
}















#endif