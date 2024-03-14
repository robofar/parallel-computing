#include "nodeClass2.cuh"
#include "kernelAStar2.cuh"

std::map<Node, Node*> storeNodes(std::string filepath) {
    std::map<Node, Node*> nodes;

    std::ifstream inputFile(filepath);
    std::string line;

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        double x1, y1, x2, y2;
        iss >> x1 >> y1;
        iss >> x2 >> y2;

        /* For some WEIRD reason, I have to have this comment here, if its not here it does not load properly. */
        /* I suppose its some buffer thing, where cout empties the buffer or similar - please ignore it in output txt file */
        std::cout << "hahaha" << std::endl;

        /* If I use CPU memory, it allocates every time the same memory space, in every iteration - maybe because this is not 
        dynamic allocation, and then (somehow) he decides to each time use same cpu memory space - check it */
        Node p1(x1,y1);
        Node p2(x2,y2);

        /* Pointers to allocated GPU memory */
        Node *p1GPU, *p2GPU; 
        
        // Now we can see perk of using UNified Memory Space instead of separate GPU/CPU memory because
        // If I now use GPU memory here to store objects, I cannot dereference poiter that points to GPU memory from host code because
        // In that case I would be accessing GPU memory from CPU code which is not possible (also vice-versa is not possible)

        // Allocate memory for the vectors in the Unified Memory Space
        cudaMallocManaged(&p1GPU, sizeof(Node));
        cudaMallocManaged(&p2GPU, sizeof(Node));

        // No need for cudaMemcpy for unified memory
        //cudaMemcpy(p1GPU, &p1, sizeof(Node), cudaMemcpyHostToDevice);
        //cudaMemcpy(p2GPU, &p2, sizeof(Node), cudaMemcpyHostToDevice);

        *p1GPU = p1;
        *p2GPU = p2;

        //nodes[*p1GPU] = *p1GPU; // kopiranje se desava, i onda ce objekat iz unified memorije biti kopiran u cpu memoriju (vjerovatno je ovo)
        // zbog ovog razloga iznad, nmg storati objekte, jer nikad necu biti u mogucnosti storati objekat koji je u GPU, moram storati opet pokazivace
        //nodes[*p2GPU] = *p2GPU;

        nodes[*p1GPU] = p1GPU;
        nodes[*p2GPU] = p2GPU;

    }

    return nodes;
}

void printNodes(const std::map<Node, Node*>& nodes) {
    for (const auto& pair : nodes) {
        std::cout << "Key: (" << pair.first.x << ", " << pair.first.y << ")";
        std::cout << " - Value: (" << (*(pair.second)).x << ", " << (*(pair.second)).y << ")\n";
        std::cout << "Adress Key: " << &(pair.first) << std::endl;
        std::cout << "Adress Value: " << (pair.second) << std::endl;
        std::cout << "========================================================" << std::endl;
    }
}

std::map<Node, std::vector<Node*>> createGraph(std::string filepath, std::map<Node, Node*>& nodes) {
    std::map<Node, std::vector<Node*>> dataMap;
    std::ifstream inputFile(filepath);
    std::string line;

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        double x1, y1, x2, y2;
        iss >> x1 >> y1;
        iss >> x2 >> y2;

        Node p1(x1,y1);
        Node p2(x2,y2);

        dataMap[p1].push_back(nodes.at(p2));
        dataMap[p2].push_back(nodes.at(p1));

    }

    inputFile.close();

    return dataMap;

}

void printGraph(const std::map<Node, std::vector<Node*>>& graph) {
    for (const auto& pair : graph) {
        const Node& node = pair.first;
        const std::vector<Node*>& neighbors = pair.second;

        std::cout << "Node: (" << node.x << ", " << node.y << ") - Neighbors: ";
        
        for (const Node* neighbor : neighbors) {
            std::cout << "(" << neighbor->x << ", " << neighbor->y << ") " << ", Address: " << neighbor << " ";
        }

        std::cout << std::endl;
    }
}

std::vector<Node> astar(std::map<Node, std::vector<Node*>>& graph, Node* start, Node* goal, bool gpu) {
    std::priority_queue<Node*, std::vector<Node*>, CompareNodes> openSet;
    (*start).updateScores(0.0, calculateHeuristic(*start, *goal));
    (*start).inOpenSet = true;
    openSet.push(start);

    while (!openSet.empty()) {
        Node* current = openSet.top();
        openSet.pop();
        (*current).inOpenSet = false;
        (*current).inClosedSet = true;

        // Check if the current node is the goal
        if(current == goal) {
            // Reconstruct the path from the goal to the start
            std::vector<Node> path;
            while ((*current).parent != nullptr) {
                path.push_back(*current);
                current = (*current).parent;
            }
            path.push_back(*current);
            std::reverse(path.begin(), path.end());
            return path;
        }

        /* Elements of vector in C++ are stored in memory one after another */
        /* Therefore it is possible to use pointer arthimetic on vector */
        /* On the other side elements of deque in C++ are NOT guarenteed that are stored one after another in memory */
        /* Therefore it is NOT possible to use pointer arthimetic on deque */

        /* 1) Pointer to whole vector - does not work, because when we dereference pointer we get vector, which we cannot use in kernel */
        /*
        std::vector<Node*>* neighborsPointer = &graph.at(*current);
        int size = (*neighborsPointer).size();

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        iterateNodeNeighbors<<<gridSize, blockSize>>>(neighborsPointer, current, size);
        cudaDeviceSynchronize();
        */

        /* 2) Pointer to first element of vector */
        if(gpu == true) {
            thrust::device_vector<Node*> neighbors = graph.at(*current);
            Node** neighbors_pointer = thrust::raw_pointer_cast(neighbors.data());

            int size = neighbors.size();

            /* Create intermediate vector just so its easier to manipulate in kernel */
            thrust::device_vector<Node*> neighborsOpenSet(size, nullptr);
            Node** openSetPtr = thrust::raw_pointer_cast(neighborsOpenSet.data());

            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;
            iterateNodeNeighbors<<<gridSize, blockSize>>>(neighbors_pointer, current, goal, size, openSetPtr);
            cudaDeviceSynchronize();

            /* Now push to priority queue all neighbor with flags that is NOT nullptr */
            for(Node* n : neighborsOpenSet) {
                if(n != nullptr) {
                    openSet.push(n);
                }
            }

        }

        if(gpu == false) {
            for (Node* neighbor : graph.at(*current)) {
                double tentative_gScore = (*current).gScore + calculateDistance(*current, *neighbor);
                bool worsePath = (tentative_gScore >= (*neighbor).gScore);

                if((*neighbor).inClosedSet && worsePath) {
                    continue;
                }

                if((*neighbor).inOpenSet && worsePath) {
                    continue;
                }

                (*neighbor).inClosedSet = false;
                (*neighbor).updateScores(tentative_gScore, calculateHeuristic(*neighbor, *goal));
                (*neighbor).inOpenSet = true;
                (*neighbor).parent = current;

                openSet.push(neighbor);
                
            }
        }

    }

    return {};

}

void printPath(const std::vector<Node>& path) {
    std::cout << "Shortest path (nodes)..." << std::endl << std::endl;
    // Print the result
    if (!path.empty()) {
        std::cout << "Path is found, and nodes of the path are:" << std::endl;
        int counter = 1;
        for (const Node& node : path) {
            std::cout << counter << ": (" << node.x << ", " << node.y << ")" << std::endl;
            counter++;
        }
    } else {
        std::cout << "Path is not found." << std::endl;
    }
}




int main() {

    /* Load start and end from original file - I need to load them so I know from where to start - because map is ordered container */
    std::ifstream f("../Assignment03/polygons_large.txt");
    std::string l;

    std::getline(f, l);
    std::istringstream aaa(l);
    double start_x, start_y;
    aaa >> start_x >> start_y;

    std::getline(f, l);
    std::istringstream bbb(l);
    double end_x, end_y;
    bbb >> end_x >> end_y;

    f.close();

    Node s(start_x, start_y);
    Node e(end_x, end_y);

    /* ======================================================================================================== */


    std::string filepath = "../Assignment03/vg_large.txt";
    std::map<Node, Node*> nodes = storeNodes(filepath);
    printNodes(nodes);
    std::cout << "=======================================" << std::endl;

    std::map<Node, std::vector<Node*>> graph = createGraph(filepath, nodes);
    printGraph(graph);
    std::cout << "=======================================" << std::endl;

    Node* end = nodes.at(e);
    Node* start = nodes.at(s);

    std::cout << "Adress start: " << start << std::endl;
    std::cout << "Adress end: " << end << std::endl;

    /* ======================================================================================================== */

    bool gpu = true;

    std::cout << "Calculating A*..." << std::endl << std::endl;
    std::vector<Node> path = astar(graph, start, end, gpu);

    std::cout << "Visibility graph number of nodes: " << graph.size() << std::endl;
    std::cout << "Shortest path number of nodes: " << path.size() << std::endl;
    printPath(path);
    std::cout << "=======================================" << std::endl;

    std::cout << "Saving data for visualization..." << std::endl;
    const char* file_path = "shortest_large.txt";
    std::ofstream file(file_path);
    int pathSize = path.size();
    for(int i = 0 ; i < (pathSize - 1) ; i++) {
        file << path[i].x << " " << path[i].y << " " << path[(i+1)].x << " " << path[(i+1)].y << "\n";
    }

    std::cout << "Done..." << std::endl;
    


    return 0;
}