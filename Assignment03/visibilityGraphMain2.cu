#include "visibilityGraph.cuh"
#include "visibilityGraphKernel2.cuh"

int main() {
    Graph graph("polygons_indoor.txt");
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
            thrust::device_vector<Line> visibilityGraph(n*n);
            thrust::device_vector<int> visibilityGraphIndices(n*n, 0);

            // Obtain pointers (raw)
            SuperPoint* v = thrust::raw_pointer_cast(verticesGPU.data());
            Line* o = thrust::raw_pointer_cast(obstaclesGPU.data());
            Line* vg = thrust::raw_pointer_cast(visibilityGraph.data());
            int* indices = thrust::raw_pointer_cast(visibilityGraphIndices.data());

            // Define kernel configuration
            dim3 blocksPerGrid(n, n);
            int threadsPerBlock = 1;

            // Execute kernel
            visibilityGraphKernel<<<blocksPerGrid, threadsPerBlock>>>(v, o, vg, n, m, indices);
            cudaDeviceSynchronize();  // Ensure that the kernel is finished before continuing

            // Only push edges that has index 1 -> those are part of visibilityGraph
            thrust::host_vector<Line> realVisibilityGraph;
            for(int i=0; i<visibilityGraphIndices.size();i++) {
                if(visibilityGraphIndices[i] == 1) realVisibilityGraph.push_back(visibilityGraph[i]);
            }

            // Print it
            VisibilityGraph::printVisibilityGraph(realVisibilityGraph);

            // output to txt file
            const char* file_path = "vis_graph.txt";
            std::ofstream file(file_path);

            for(int i = 0 ; i < realVisibilityGraph.size() ; i++) {
                file << realVisibilityGraph[i].start.x << " " << realVisibilityGraph[i].start.y << " " << realVisibilityGraph[i].end.x << " " << realVisibilityGraph[i].end.y << "\n" ;
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