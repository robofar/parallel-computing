#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream> 

class Point {
    public:
        int x;
        int y;
        Point() {}
        Point(int x, int y) {
            this -> x = x;
            Point::y = y;
        }
};

class Line {
    public:
        Point start;
        Point end;

        Line(Point s, Point e) {
            start = s;
            end = e;
        }
};

class Polygon {
    public:
        std::vector<Point> vertices;
        std::vector<Line> edges;

        Polygon(std::vector<Point> v, std::vector<Line> e) {
            vertices = v;
            edges = e;
        }
};

class Graph {
    public:
        Point start;
        Point end;
        std::vector<Polygon> polygons;
        

        Graph(std::string filepath) {
            readGraph(filepath);
        }

        void readGraph(std::string filepath) {
            std::ifstream input(filepath);
            input >> start.x >> start.y >> end.x >> end.y;

            std::vector<Point> polygonVertices;
            std::vector<Line> polygonEdges;
            
            std::string line;
            int x;
            int y;

            // Fake polygon - contains only start and end vertices (without edge between them - that would be fake obstacle) 
            // because it is easier then to loop (since my approach is looping through polygons of graph)
            polygonVertices.push_back(start);
            polygonVertices.push_back(end);
            Polygon polygon(polygonVertices, polygonEdges);
            polygons.push_back(polygon);
            polygonVertices.clear();
            polygonEdges.clear();


            bool newPolygon = false;
            while(std::getline(input, line)) {
                if (!line.empty()) {
                    std::istringstream iss(line);
                    iss >> x >> y;
                    Point vertex(x,y);
                    polygonVertices.push_back(vertex);
                    newPolygon = true;
                }
                else {
                    if(newPolygon) {
                        int n_vertices_polygon = polygonVertices.size();
                        for(int i=0 ; i<n_vertices_polygon ; i++) {
                            int idx_start = i % n_vertices_polygon;
                            int idx_end = (i+1) % n_vertices_polygon;
                            Line edge(polygonVertices[idx_start], polygonVertices[idx_end]);
                            polygonEdges.push_back(edge);   
                        }
                        Polygon polygon(polygonVertices, polygonEdges);
                        polygons.push_back(polygon);
                        polygonVertices.clear();
                        polygonEdges.clear();
                    }
                }
            }

            input.close();
        }
};

class VisibilityGraph {
    public:
        std::vector<Line> visibilityGraph;
        std::vector<bool> concaveCorners;

        VisibilityGraph() {}

        // Function to find the orientation of triplet (p, q, r).
        // The function returns the following values:
        // 0: Collinear points
        // 1: Clockwise points
        // 2: Counterclockwise points
        static int orientation(const Point& p, const Point& q, const Point& r) {
            int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
            if (val == 0) return 0;  // collinear
            return (val > 0)? 1: 2; // clock or counterclock wise 
        }

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

        // Function to calculate cross product of two vectors (AB and BC)
        static double crossProduct(const Point& A, const Point& B, const Point& C) {
            double cross = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
            return cross;
        }

        static bool concaveVertex(Point previous, Point current, Point following) {
            double cross = crossProduct(previous, current, following);
            return cross > 0; // Concave if cross product is positive (for this setup of cross-product)
        }

        void concaveVertices(const Graph& graph) {
            for(int i=0 ; i<graph.polygons.size() ; i++) {
                for(int j=0 ; j<graph.polygons[i].vertices.size() ; j++) {
                    int current = j;
                    int previous = ((j-1) < 0)? (graph.polygons[i].vertices.size()-1):(j-1);
                    int following = ((j+1) == graph.polygons[i].vertices.size())? 0:(j+1);
                    if(!concaveVertex(graph.polygons[i].vertices[previous], graph.polygons[i].vertices[current], graph.polygons[i].vertices[following])) {
                        concaveCorners.push_back(false);
                    }
                    else concaveCorners.push_back(true);
                }
            }
        }

        static bool tangentialEdge(const Graph& graph, Line edge, int polygon_idx, int vertex_idx) {
            int current = vertex_idx;
            int previous = ((vertex_idx-1) < 0)? (graph.polygons[polygon_idx].vertices.size()-1):(vertex_idx-1);
            int following = ((vertex_idx+1) == graph.polygons[polygon_idx].vertices.size())? 0:(vertex_idx+1);

            Point v_i = edge.start;
            Point v_j = edge.end;
            Point v_prev = graph.polygons[polygon_idx].vertices[previous];
            Point v_next = graph.polygons[polygon_idx].vertices[following];

            double crossPrev = crossProduct(v_i, v_j, v_prev);
            double crossNext = crossProduct(v_i, v_j, v_next);

            //std::cout << polygon_idx << " " << vertex_idx << std::endl;
            //std::cout << previous << " " << following << std::endl;
            //std::cout << crossPrev << " " << crossNext << std::endl;
            //std::cout << "================================" << std::endl;

            return (crossPrev * crossNext) >= 0; // Same sign means they are on the same side
        }

        void computeVisibilityGraph(const Graph& graph) {
            // For each vertex in the graph, give it "flag" if it is concave or convex vertex (stored in vector)
            // I could also put attribute concave in class Vertex but does not matter now
            concaveVertices(graph);

            std::vector<Line> obstacles;
            for(int i=0 ; i<graph.polygons.size() ; i++) {
                for(int j=0 ; j<graph.polygons[i].edges.size() ; j++) {
                    obstacles.push_back(graph.polygons[i].edges[j]);
                }
            }
            
            /*
            std::cout << "Polygons numbers: " << graph.polygons.size() << std::endl;
            for(int q = 0 ; q<graph.polygons.size() ; q++) {
                std::cout << graph.polygons[q].vertices.size() << std::endl;
            }
            */


            
            int index1 = 0;
            // For each polygon
            for(int i=0 ; i<graph.polygons.size() ; i++) {
                // For each vertex in that polygon
                for(int k=0 ; k<graph.polygons[i].vertices.size() ; k++) {
                    // If corner is concave, do not check that corner with all other corners in the graph, because it cannot be part of shortest path
                    if(concaveCorners[index1] == false) {
                        int index2 = 0;
                        for(int j=0 ; j<graph.polygons.size() ; j++) {
                            for(int l=0 ; l<graph.polygons[j].vertices.size() ; l++) {
                                if(concaveCorners[index2] == false) {
                                    // Do not check same vertices in same polygon
                                    if(index1 != index2) {
                                        //std::cout << i << " " << k << " " << j << " " << l << std::endl;
                                        Line visibilityGraphEdge = {graph.polygons[i].vertices[k], graph.polygons[j].vertices[l]};

                                        // Check if this possible visibilityGraphEdge is non-tangential in its ending point
                                        // if it is tangential, then compare intesection with all other edges
                                        // if it is non-tangential, dont compare intersection with all other edges
                                        if(tangentialEdge(graph, visibilityGraphEdge, j, l)) {

                                            // Check if the visibility edge intersects with any obstacle edge
                                            bool isVisible = true;
                                            for (const Line& obstacle : obstacles) {
                                                if (doIntersect(visibilityGraphEdge, obstacle)) {
                                                    isVisible = false;
                                                    break;
                                                }
                                            }

                                            if (isVisible) {
                                                // Store edge as part of the visibilityGraph
                                                visibilityGraph.push_back(visibilityGraphEdge);
                                            }

                                        }
                                    }
                                }
                                index2++;
                            }
                        }
                    }
                    index1++;
                }
            }
        }

        void printVisibilityGraph() {
            std::cout << "Visibility Graph stored...\n";
            for (const Line& edge : visibilityGraph) {
                std::cout << "(" << edge.start.x << ", " << edge.start.y << ") - ("
                        << edge.end.x << ", " << edge.end.y << ")\n";
            }
        }

};

int main() {
    Graph graph("../polygons_small.txt");
    VisibilityGraph visibilityGraph;
    visibilityGraph.computeVisibilityGraph(graph);
    visibilityGraph.printVisibilityGraph();

    return 0;
}