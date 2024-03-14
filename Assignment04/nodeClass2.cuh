#ifndef GRAPH_NODE_CLASS_HEADER
#define GRAPH_NODE_CLASS_HEADER

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <cmath>
#include <unordered_map>
#include <functional>
#include <algorithm>  // Include <algorithm> for std::reverse
#include <typeinfo>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



class Node {
    public:
        double x, y;

        double gScore;  // Cost from the start node to this node
        double hScore;  // Heuristic (estimated cost) from this node to the goal
        double fScore; // f = g + h

        bool inOpenSet;
        bool inClosedSet;

        Node* parent;

        __device__ __host__
        Node(double x = 0.0, double y = 0.0) {
            Node::x = x;
            Node::y = y;

            Node::gScore = 0.0;
            Node::hScore = 0.0;
            fScore = gScore + hScore;

            inOpenSet = false;
            inClosedSet = false;

            parent = nullptr;
        }

        __device__ __host__
        void updateScores(double g, double h) {
            Node::gScore = g;
            Node::hScore = h;
            Node::fScore = g + h;
        }

        // Comparison function for priority queue (min heap based on total cost)
        __device__ __host__
        bool operator >(const Node& other) const {
            return (gScore + hScore) > (other.gScore + other.hScore);
        }

        // Define comparison operators
        __device__ __host__
        bool operator<(const Node& other) const {
            if (x < other.x)
                return true;
            else if (x > other.x)
                return false;
            else
                return y < other.y;
        }

        __device__ __host__
        bool operator==(const Node& other) const {
            const double epsilon = 1e-3;
            return std::abs(x - other.x) < epsilon && std::abs(y - other.y) < epsilon;
        }
};

struct CompareNodes {
    bool operator()(const Node* a, const Node* b) const {
        return a->fScore > b->fScore;
    }
};

/*
class PriorityQueue {
public:
    std::vector<Node*> nodes;

    __device__ __host__
    void push(Node* node) {
        nodes.push_back(node);
        std::push_heap(nodes.begin(), nodes.end(), CompareNodes()); // std::greater<>() -> does not work because now we have pointer to Node not object itself
    }

    __device__ __host__
    Node* top() {
        return nodes.front();
    }

    __device__ __host__
    void pop() {
        std::pop_heap(nodes.begin(), nodes.end(), CompareNodes()); // std::greater<>() -> does not work because now we have pointer to Node not object itself
        nodes.pop_back();
    }

    __device__ __host__
    bool empty() const {
        return nodes.empty();
    }

    __device__ __host__
    size_t size() const {
        return nodes.size();
    }
};
*/


// Helper function to calculate the Euclidean distance between two nodes
__device__ __host__
double calculateDistance(const Node& node1, const Node& node2) {
    double dx = node1.x - node2.x;
    double dy = node1.y - node2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Helper function to calculate the Euclidean distance between two nodes
__device__ __host__
double calculateHeuristic(const Node& node1, const Node& node2) {
    double dx = node1.x - node2.x;
    double dy = node1.y - node2.y;
    return std::sqrt(dx * dx + dy * dy);
}














#endif