#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using std::vector;
using std::string;
using std::ifstream;
using std::cout;
using std::getline;
using std::stoi;
using std::istringstream;

vector<vector<int>> parse_file_to_matrix(ifstream& file) {
    vector<vector<int>> matrix;
    string str_line;
    while (getline(file, str_line)) {
        vector<int> row = {};
        istringstream iss(str_line);
        string token;

        while (iss >> token) {
            row.push_back(stoi(token));
        };

        matrix.push_back(row);
    }
    return matrix;
};

ifstream get_file(string filename) {
    ifstream input_file;
    input_file.open(filename, std::ios::binary);
    return input_file;
};

int main() {
    ifstream infile = get_file("matrix.txt");
    vector<vector<int>> m = parse_file_to_matrix(infile);
    
    for (int i = 0; i < m.size(); i++) {
        vector<int> row = m[i];
        for (int j = 0; j < row.size(); j++) {
            cout << row[j] << " ";
        }
        cout << "\n";
    }

    return 0;
}