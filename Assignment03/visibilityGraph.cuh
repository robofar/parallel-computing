#ifndef VISIBILITY_GRAPH_CPU_HEADER
#define VISIBILITY_GRAPH_CPU_HEADER



#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream> 

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class Point {
    public:
        float x;
        float y;

        __host__ __device__
        Point() {
            this->x = 0;
            this->y = 0;
        }

        __host__ __device__
        Point(float x, float y) {
            this->x = x;
            this->y = y;
        }
};

class SuperPoint {
    public:
        Point current;
        Point previous;
        Point next;
        bool hasPrevious;
        bool hasNext;

        __host__ __device__
        SuperPoint() {}

        __host__ __device__
        SuperPoint(Point current) {
            this->current = current;
            this->hasPrevious = false;
            this->hasNext = false;
        }

        __host__ __device__
        SuperPoint(Point current, Point previous, Point next) {
            this->current = current;
            this->previous = previous;
            this->next = next;
            this->hasPrevious = true;
            this->hasNext = true;
        }
};

class Line {
    public:
        Point start;
        Point end;

        __host__ __device__
        Line() {}

        __host__ __device__
        Line(Point start, Point end) {
            this->start = start;
            this->end = end;
        }
};

class Graph {
    public:
        thrust::host_vector<SuperPoint> vertices;
        thrust::host_vector<Line> obstacles;

        
        Graph(std::string filename) {
            readGraph(filename);
            printGraph();
        }


        void readGraph(std::string filename) {
            std::ifstream input(filename);

            /* Load start and goal point*/
            Point point_start;
            Point point_goal;
            input >> point_start.x >> point_start.y >> point_goal.x >> point_goal.y;
            SuperPoint start(point_start);
            SuperPoint goal(point_goal);
            vertices.push_back(start);
            vertices.push_back(goal);

            /* Load rest of the graph */
            std::string line;
            float x;
            float y;
            thrust::host_vector<Point> points_buffer;

            bool newPolygon = false;
            while(std::getline(input, line)) {
                if (!line.empty()) {
                    std::istringstream iss(line);
                    iss >> x >> y;
                    Point vertex(x,y);
                    points_buffer.push_back(vertex);
                    newPolygon = true;
                }
                else {
                    if(newPolygon) {
                        int n_vertices_polygon = points_buffer.size();
                        for(int i=0 ; i<n_vertices_polygon ; i++) {
                            int current_idx = i;
                            int previous_idx = ((i-1) < 0) ? (n_vertices_polygon-1):(i-1);
                            int next_idx = ((i+1) == n_vertices_polygon) ? 0:(i+1);
                            SuperPoint current_point(points_buffer[current_idx], points_buffer[previous_idx], points_buffer[next_idx]);
                            Line edge(points_buffer[current_idx], points_buffer[next_idx]);
                            vertices.push_back(current_point);
                            obstacles.push_back(edge);
                        }
                        points_buffer.clear();
                    }
                }
            }
            input.close();
        }

        void printGraph() {
            std::cout << vertices.size() << std::endl << std::endl; 
            for(int i = 0 ; i<vertices.size() ; i++) {
                std::cout<<"======================"<<std::endl;
                std::cout << "Current point: (" << vertices[i].current.x << ", " << vertices[i].current.y << ")" << std::endl;
                if(vertices[i].hasPrevious) std::cout << "Previous point: (" << vertices[i].previous.x << ", " << vertices[i].previous.y << ")" << std::endl;
                if(vertices[i].hasNext) std::cout << "Next point: (" << vertices[i].next.x << ", " << vertices[i].next.y << ")" << std::endl;
            }
        }
};

class VisibilityGraph {
    public:
        thrust::host_vector<Line> visibilityGraph;

        VisibilityGraph(const Graph& graph, bool gpu) {
            if(gpu == false) computeVisibilityGraph(graph);
        }

        // Function to find the orientation of triplet (p, q, r).
        // The function returns the following values:
        // 0: Collinear points
        // 1: Clockwise points
        // 2: Counterclockwise points
        __host__ __device__
        static float orientation(const Point& p, const Point& q, const Point& r) {
            float eps = 1e-3; // use this next time (also in all other methods that compare floats or similar)
            float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
            if (val == 0.0) return 0;  // collinear
            return (val > 0.0)? 1: 2; // clock or counterclock wise 
        }

        __host__ __device__
        static bool doIntersect(Line l1, Line l2) {
            Point p1 = l1.start;
            Point q1 = l1.end;
            Point p2 = l2.start;
            Point q2 = l2.end;

            int o1 = orientation(p1, q1, p2);
            int o2 = orientation(p1, q1, q2);
            int o3 = orientation(p2, q2, p1);
            int o4 = orientation(p2, q2, q1);

            // General case
            if((o1 != o2) && (o3 != o4) && o1!=0 && o2!=0 && o3!=0 && o4!=0) return true;

            return false;
        }

        __host__ __device__
        static double scalarProduct(const Point& A, const Point& B, const Point& C) {
            double ab_x = (B.x - A.x);
            double ab_y = (B.y - A.y);

            // 2nd quadrrant: neg x, pos y
            // 4th quadrant: pos x, neg y
            double ac_x = (-1.0) * (C.x - A.x);
            double ac_y = (-1.0) * (C.y - A.y);

            double scalar = (ab_x*ac_x) + (ab_y*ac_y);
            return scalar;
        }

        __host__ __device__
        static double crossProduct2D(const Point& A, const Point& B, const Point& C) {
            double ab_x = (B.x - A.x);
            double ab_y = (B.y - A.y);

            double ac_x = (C.x - A.x);
            double ac_y = (C.y - A.y);

            double cross = (ab_x*ac_y) - (ab_y*ac_x);

            return cross;
        }

        __host__ __device__
        static bool concaveVertex(const SuperPoint& vertex) {
            if(vertex.hasPrevious == false) return false;
            double cross = crossProduct2D(vertex.current, vertex.previous, vertex.next);
            // Since vertices in graph.txt are defined in counter-clock-wise direction, everything is opposite
            // In this case, magnitude of cross product will be negative or 0 when vertex is convex, that's why here we have >0 (that is cocave vertex then)
            // We found cross product more suitable then scalar/dot product for this testing
            return cross > 0;
        }

        // tangentialEdge test, wrt to end point neighbours (for other direction, just call function with reverse indices)
        __host__ __device__
        static bool tangentialEdge(const SuperPoint& start, const SuperPoint& end) {
            if(end.hasPrevious == false) return true;
            double scalar1 = crossProduct2D(end.current, start.current, end.previous);
            double scalar2 = crossProduct2D(end.current, start.current, end.next);

            return (scalar1 * scalar2) >= 0;
        }

        void computeVisibilityGraph(const Graph& graph) {
            for(int i=0 ; i<graph.vertices.size() ; i++) {
                if(concaveVertex(graph.vertices[i]) == false) {
                    for(int j=i+1 ; j<graph.vertices.size() ; j++) {
                        if(concaveVertex(graph.vertices[j]) == false) {
                            // check tangentialEdge condition for both directions (i.e. in both ends): vi,vj and vj,vi (OR or AND) ???
                            bool direction1 = tangentialEdge(graph.vertices[i], graph.vertices[j]);
                            bool direction2 = tangentialEdge(graph.vertices[j], graph.vertices[i]);
                            if((direction1 == true) && (direction2 == true)) {
                                Line visibilityGraphEdge = {graph.vertices[i].current, graph.vertices[j].current};
                                
                                bool isVisible = true;
                                for (const Line& obstacle : graph.obstacles) {
                                    if (doIntersect(visibilityGraphEdge, obstacle)) {
                                        isVisible = false;
                                        break;
                                    }
                                }

                                if(isVisible) visibilityGraph.push_back(visibilityGraphEdge);
                                
                            }
                        }
                    }
                }
            }
        }

        static void printVisibilityGraph(const thrust::host_vector<Line>& visibilityGraph) {
            std::cout << "Visibility Graph stored...\n" << std::endl;
            std::cout << "Number of edges in Visibility Graph is: " << visibilityGraph.size() << std::endl << std::endl;

            for (const Line& edge : visibilityGraph) {
                std::cout << "(" << edge.start.x << ", " << edge.start.y << ") - ("
                        << edge.end.x << ", " << edge.end.y << ")\n";
            }
        }

        static void saveVisibilityGraph(const thrust::host_vector<Line>& visibilityGraph) {
            std::cout << "Saving visibility graph..." << std::endl << std::endl;
            const char* file_path = "vg_large.txt";
            std::ofstream file(file_path);

            for(int i = 0 ; i < visibilityGraph.size() ; i++) {
                file << visibilityGraph[i].start.x << " " << visibilityGraph[i].start.y << " " << visibilityGraph[i].end.x << " " << visibilityGraph[i].end.y << "\n";
            }
        }


};

#endif // VISIBILITY_GRAPH_CPU_H
