#include "visibilityGraph.cuh"
#include "visibilityGraphKernel3.cuh"

int main() {
    Graph graph("polygons_large.txt");
    bool gpu = false;


    if(gpu == false) {
        VisibilityGraph vGraph(graph, gpu);
        VisibilityGraph::printVisibilityGraph(vGraph.visibilityGraph);
        VisibilityGraph::saveVisibilityGraph(vGraph.visibilityGraph);
    }
    else {
        try{
            // Copy the vectors from host to device
            thrust::device_vector<SuperPoint> verticesGPU = graph.vertices;
            thrust::device_vector<Line> obstaclesGPU = graph.obstacles;
            int n = verticesGPU.size();
            int m = obstaclesGPU.size();

            // Vector in which we will store (we don't know actual size but we have to allocate because of kernel)
            //thrust::device_vector<Line> visibilityGraph(n*n);
            thrust::device_vector<int> vGIndices(n*n*m, 0);

            // Obtain pointers (raw)
            SuperPoint* v = thrust::raw_pointer_cast(verticesGPU.data());
            Line* o = thrust::raw_pointer_cast(obstaclesGPU.data());
            //Line* vg = thrust::raw_pointer_cast(visibilityGraph.data());
            int* indices = thrust::raw_pointer_cast(vGIndices.data());

            // Define kernel configuration
            dim3 blocksPerGrid(n, n);
            int threadsPerBlock = m;

            // Execute kernel
            visibilityGraphKernel<<<blocksPerGrid, threadsPerBlock>>>(v, o, n, m, indices);
            cudaDeviceSynchronize();  // Ensure that the kernel is finished before continuing

            // Only push edges that has index 1 -> those are part of visibilityGraph
            thrust::host_vector<Line> visibilityGraph;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    // Calculate the index corresponding to the pair (i, j) in the flattened 1D array
                    int index = i * n + j;

                    // Check if the edge (created from the pair of vertices i and j) has at least one intersection with obstacles
                    bool doIntersect = std::any_of(vGIndices.begin() + index * m, vGIndices.begin() + (index + 1) * m, [](int x) { return x == 0; });

                    // If there is no intersection, then that edge is part of the visibility graph
                    if (!doIntersect) {
                        SuperPoint ei = verticesGPU[i];
                        SuperPoint ej = verticesGPU[j];
                        Line visibilityGraphEdge = {ei.current, ej.current};
                        visibilityGraph.push_back(visibilityGraphEdge);
                    }
                }
            }

            // Print it
            VisibilityGraph::printVisibilityGraph(visibilityGraph);

            // output to txt file
            const char* file_path = "vg_large.txt";
            std::ofstream file(file_path);

            for(int i = 0 ; i < visibilityGraph.size() ; i++) {
                file << visibilityGraph[i].start.x << " " << visibilityGraph[i].start.y << " " << visibilityGraph[i].end.x << " " << visibilityGraph[i].end.y << "\n" ;
            }


        }
        catch(const thrust::system::detail::bad_alloc& e) {
            std::cout << e.what() << std::endl;
            return 0;
        }
        catch(const std::exception& e) {
            std::cout << e.what() << std::endl;
            return 0;
        }
        catch(...) {
            std::cout << "Some other exception occured\n";
            return 0;
        }

    }

    return 0;
}