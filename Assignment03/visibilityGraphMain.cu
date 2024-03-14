#include "visibilityGraph.cuh"
#include "visibilityGraphKernel.cuh"

int main() {
    Graph graph("../polygons_small.txt");
    bool gpu = true;
    

    if(gpu == false) {
        VisibilityGraph vGraph(graph, gpu);
        VisibilityGraph::printVisibilityGraph(vGraph.visibilityGraph);
    }
    else {
        try{
            // Copy the vectors from host to device
            thrust::device_vector<SuperPoint> verticesGPU = graph.vertices;
            thrust::device_vector<Line> obstaclesGPU = graph.obstacles;
            int n = verticesGPU.size();
            int m = obstaclesGPU.size();

            // Vector in which we will store (we don't know actual size but we have to allocate because of kernel)
            thrust::device_vector<Line> visibilityGraph(n*n);
            int realSize = 0;

            // Obtain pointers (raw)
            SuperPoint* v = thrust::raw_pointer_cast(verticesGPU.data());
            Line* o = thrust::raw_pointer_cast(obstaclesGPU.data());
            Line* vg = thrust::raw_pointer_cast(visibilityGraph.data());

            // Define kernel configuration
            dim3 blocksPerGrid(n, n);
            int threadsPerBlock = m;

            // Execute kernel
            visibilityGraphKernel<<<blocksPerGrid, threadsPerBlock>>>(v, o, vg, n, m, &realSize);
            cudaDeviceSynchronize();  // Ensure that the kernel is finished before continuing
            std::cout << "Kernel executed successfully." << std::endl;
            thrust::device_vector<Line> realVisibilityGraph(visibilityGraph.begin(), visibilityGraph.begin() + realSize);


            std::cout << visibilityGraph.size() << std::endl;
            std::cout << realVisibilityGraph.size() << std::endl;

            std::cout << "END" << std::endl;

            // Print it
            //VisibilityGraph::printVisibilityGraph(realVisibilityGraph);
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