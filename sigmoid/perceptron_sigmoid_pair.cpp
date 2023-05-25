/**
 * PROYECTO I - RED NEURONAL PARA DIFERENCIAR LETRAS VOCALES MANUSCRITAS
 * Hecha por: Kevin Rojas - C.I 29.582.382
 * Materia: Modelos de Programación Emergente (MPE)
 *   
 * Enunciado: Desarrolle una RNA basada en el Perceptron Simple que permita el reconocimiento 
 * de letras minúsculas (vocales: a, e, i, o, u) manuscritas. La entrada será un archivo contentivo 
 * del caracter fotografiado, o una matriz 16x10 que simule el caracter, a analizar.
 * 
 * Entrada: Para iniciar el reconocimiento de una vocal debemos tener un archivo .txt llamado "input.txt"
 * en el mismo directorio donde se encuentra nuestro archivo perceptron_sigmoid_pair.cpp, además de la 
 * base de conocimiento "base.txt". El formato que debe tener esta entrada es:
 * 
 * 0 0 0 0 0 0 0 0 0 0 
 * 0 0 0 0 0 0 0 0 0 0 
 * 0 0 0 0 0 0 0 0 0 0 
 * 0 0 0 0 0 0 0 0 0 0 
 * 0 0 0 0 0 0 0 0 0 0 
 * 0 0 0 0 0 0 0 0 0 0 
 * 0 1 0 0 0 0 0 1 0 0 
 * 0 1 0 0 0 0 0 1 0 0 
 * 0 1 0 0 0 0 0 1 0 0 
 * 0 1 0 0 0 0 0 1 0 0 
 * 0 1 1 0 0 0 0 1 0 0 
 * 0 0 1 0 0 0 0 1 0 0 
 * 0 0 1 1 0 0 1 1 1 0 
 * 0 0 0 0 1 1 0 0 1 0 
 * 0 0 0 0 0 0 0 0 0 1 
 * 0 0 0 0 0 0 0 0 0 0
 * 
 * No debe haber saltos de línea en la parte de arriba. Es decir, la línea número 1 es la 
 * fila número 1 de la matriz.
*/
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <ctime>

using std::vector;
using std::string;
using std::fstream;
using std::cout;
using std::getline;
using std::stoi;
using std::stof;
using std::istringstream;
using std::to_string;
using std::mt19937;
using std::uniform_real_distribution;
using std::exp;
using std::round;
using std::pair;
using std::srand;
using std::rand;

const int COLUMNS_NUM = 10;
const int ROWS_NUM = 16;
const int PATTERNS_NUM = 30;
const int RANDOM_STATE = 42;
const float MIN_WEIGHT_VALUE = -1.0;
const float MAX_WEIGHT_VALUE = 1.0;
const float LEARNING_RATE = 0.05;
const int ITERATION_NUMBER = 2000;
mt19937 generator(RANDOM_STATE);
uniform_real_distribution<> dist(MIN_WEIGHT_VALUE, MAX_WEIGHT_VALUE);

/**
 * @brief Clase FileManager que se encarga de manejar la apertura, lectura y escritura de archivos .txt.  
*/
class FileManager {
    public:
        fstream file;
        string mode;
        string filename;

        /**
         * @brief Constructor de la clase FileManager.
         * 
         * @param aFilename Parámetro de tipo cadena de caracteres que representa el nombre del archivo. 
         * @param aMode Parámetro de tipo cadena de caracteres que representa el modo de apertura del 
         * archivo (lectura o escritura).
        */
        FileManager(string aFilename, string aMode) {
            filename = aFilename;
            mode = aMode;
            get_file();
        };

        /**
         * @brief Método utilizado para parsear una matriz de números de coma flotante dentro de un archivo 
         * .txt.
         * 
         * @return Vector de vectores de números de coma flotante (matriz de números reales).
        */
        vector<vector<float>> parse_float_matrix() {
            vector<vector<float>> matrix;
            string str_line;
            for(int i = 0; i < 16; i++) {
                getline(file, str_line);
                vector<float> row = {};
                istringstream iss(str_line);
                string token;

                while (iss >> token) {
                    row.push_back(stof(token));
                };

                matrix.push_back(row);
            }
            return matrix;
        };

        /**
         * @brief Método utilizado para parsear una matriz de números enteros dentro de un archivo .txt.
         * 
         * @return Vector de vectores de números enteros (matriz de números enteros).
        */
        vector<vector<int>> parse_integer_matrix() {
            vector<vector<int>> matrix;
            string str_line;
            for(int i = 0; i < 16; i++) {
                getline(file, str_line);
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

        /**
         * @brief Método utilizado para parsear un número de coma flotante dentro de un archivo .txt.
         * 
         * @return Número de coma flotante (número real).
        */
        float parse_float() {
            float number;
            string str_line;
            getline(file, str_line);
            number = stof(str_line);

            return number;
        }

        // Extras

        /**
         * @brief Método utilizado para obtener el archivo .txt en el modo especificado por la clase.
        */
        void get_file() {
            if (mode == "write") {
                file.open(filename, std::ios::out);
            } else {
                file.open(filename, std::ios::in);
            }
            
        };

        /**
         * @brief Método utilizado para escribir en el archivo .txt dado por la clase.
        */
        void write(string words) {
            file << words;
        };

        /**
         * @brief Método utilizado para saltar de línea en el archivo .txt dado por la clase.
        */
        void line_break() {
            string str_line;
            getline(file, str_line);
        }
};

/**
 * @brief Clase Perceptron que modela la neurona propuesta por Frank Rosenblatt.
 * 
 * Se utiliza como una neuorna individual en una red neuronal. Posee N número de entradas (dendritas), con
 * sus respectivos pesos, una razón de aprendizaje, la sumatoria ponderada y una función de activación.   
*/
class Perceptron {
    public:
        int randomState;
        float bias;
        vector<vector<float>> weights;
        float learningRate;

        /**
         * @brief Constructor de la clase Perceptron.
         * 
         * @param aLearningRate Parámetro de tipo flotante necesario para otorgar la razón de aprendizaje de la neurona. 
         * @param aRandomState Parámetro de tipo entero necesario para otorgar la semilla de aleatoriedad para generar 
         * los pesos de la neurona.
        */
        Perceptron(float aLearningRate, int aRandomState) {
            learningRate = aLearningRate;
            randomState = aRandomState;
            initialize_bias();
            initialize_weights();
        };

        /**
         * @brief Método utilizado para calcular la suma ponderada en base a las entradas y los pesos de la neurona. 
         * Se utiliza un producto punto entre el vector de entradas y el vector de pesos y, al final, se le suma el
         * sesgo (o bias). Fórmula: sum(x * w) + bias
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros (matriz de enteros) que hace referencia 
         * a los valores de entrada de la neurona.
         * 
         * @return Número de coma flotante (resultado de la suma ponderada).
        */
        float net_input(vector<vector<int>> inputValues) {
            float result = 0;
            for (int i = 0; i < ROWS_NUM; i++) {
                for (int j = 0; j < COLUMNS_NUM; j++) {
                    result += inputValues[i][j] * weights[i][j];
                }
            }
            return result + bias;
        };

        /**
         * @brief Método utilizado para describir la función de activación de la neurona (función sigmoide). 
         * Se utiliza para determinar si las entradas son capaces de activar (excitar) o no (inhibir) a la neurona.
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros (matriz de enteros) que hace referencia 
         * a los valores de entrada de la neurona.
         * 
         * @return Número de coma flotante (salida de la función sigmoide evaluada en el resultado de la suma ponderada).
        */
        float activation_function(vector<vector<int>> inputValues) {
            float weightedSum = net_input(inputValues);
            return 1 / (1 + exp(-weightedSum));
        };

        // AJUSTES DE LA NEURONA

        /**
         * @brief Método utilizado para ajustar los pesos del perceptron (neuorna) a través de la Regla Delta. 
         * Fórmula: w + L(s - y)x
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros (matriz de enteros) que hace referencia 
         * a los valores de entrada de la neurona.
         * @param expectedValue Parámetro de tipo entero que hace referencia al valor esperado en la salida de
         * la neurona.
         * @param outputValue Parámetro de tipo número de coma flotante que hace referencia al valor obtenido 
         * en la salida de la neurona.
        */
        void adjust_weights(vector<vector<int>> inputValues, int expectedValue, int outputValue) {
            for (int i = 0; i < ROWS_NUM; i++) {
                for (int j = 0; j < COLUMNS_NUM; j++) {
                    weights[i][j] += learningRate * (expectedValue - outputValue) * inputValues[i][j];
                };
            };
        };

        /**
         * @brief Método utilizado para ajustar el sesgo (o bias) del perceptron (neuorna) a través de la Regla Delta. 
         * Fórmula: b + L(s - y)1
         * 
         * @param expectedValue Parámetro de tipo entero que hace referencia al valor esperado en la salida de
         * la neurona.
         * @param outputValue Parámetro de tipo número de coma flotante que hace referencia al valor obtenido 
         * en la salida de la neurona.
        */
        void adjust_bias(int expectedValue, int outputValue) {
            bias += learningRate * (expectedValue - outputValue) * 1;
        };

        // MÉTODOS DE UTILIDAD DEL PERCEPTRON

        /**
         * @brief Método utilizado para inicializar el vector de pesos de la neurona aleatoriamente.
        */
        void initialize_weights() {
            for (int i = 0; i < ROWS_NUM; i++) {
                vector<float> new_row = {};
                for (int i = 0; i < COLUMNS_NUM; i++) {
                    new_row.push_back(dist(generator));
                };
                weights.push_back(new_row);
            };
        };

        /**
         * @brief Método utilizado para inicializar el valor del bias (o sesgo ) de la neurona aleatoriamente.
        */
        void initialize_bias() {
            bias = dist(generator);
        };

        /**
         * @brief Método utilizado para mostrar el vector de pesos de la neurona en la consola.
        */
        void show_weights() {
            for (vector<float> weight : weights) {
                for (float w : weight) {
                    cout << w << ", ";
                };
                cout << "\n";
            };
        };

        /**
         * @brief Método utilizado para retornar el vector de pesos de la neurona en una cadena de caracteres.
         * 
         * @return Cadena de caracteres que representan la matriz de pesos y el bias (o sesgo) de la neurona.
        */
        string weights_to_string() {
            string result = "";
            for (vector<float> weight : weights) {
                string row = "";
                for (float w : weight) {
                    row += to_string(w) + " ";
                };
                row = row.substr(0, row.length() - 1);
                result += row + "\n";
            };
            string bias_str = to_string(bias);
            result += bias_str + "\n\n";
            return result;
        };

        // SETTERS

        /**
         * @brief Método utilizado para asignar un nuevo vector de pesos a la neurona.
        */
        void set_weights(vector<vector<float>> newWeights) {
            weights = newWeights;
        };

        /**
         * @brief Método utilizado para asignar un nuevo sesgo (o bias) a la neurona.
        */
        void set_bias(float newBias) {
            bias = newBias;
        };
};

/**
 * @brief Clase NeuralNetwork que modela una red neuronal de N perceptrones simples.
 * 
 * Se utiliza para analizar el problema planteado en el enunciado: identificar o reconocer
 * letras vocales minúsculas (a, e, i, o, u), a través de una matriz de entrada. La salida 
 * de la red neuronal es un vector de enteros de tamaño N.    
*/
class NeuralNetwork {
    public:
        vector<Perceptron> perceptrons;
        int iterationsNumber;

        /**
         * @brief Constructor de la clase NeuralNetwork.
         * 
         * @param aIterationsNumber Parámetro de tipo entero que determina el número de iteraciones que hará la red neuronal
         * para entrenar y aprender los patrones.
         * @param aPerceptronsNumber Parámetro de tipo entero necesario para determinar el número de perceptrones (neuronas) 
         * en la red neuronal. 
        */
        NeuralNetwork(int aPerceptronsNumber, int aIterationsNumber) {
            for (int i = 0; i < aPerceptronsNumber; i++) {
                Perceptron newPerceptron(LEARNING_RATE, 42);
                perceptrons.push_back(newPerceptron);
            };
            iterationsNumber = aIterationsNumber;
        }

        /**
         * @brief Método utilizado para procesar un patrón específico con la red neuronal.
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros que representa la entrada de la red neuronal.
         * 
         * @return Vector de números de coma flotante (salidas de la función de activación de cada neurona de la red neuronal).
        */
        vector<float> process_input(vector<vector<int>> inputValues) {
            vector<float> output = {};
            for (int i = 0; i < perceptrons.size(); i++) {
                float predict = perceptrons[i].activation_function(inputValues);
                output.push_back(predict);
            };
            return output;
        };

        /**
         * @brief Método utilizado para entrenar a la red neuronal con un conjunto de patrones. Durante un número de iteraciones
         * determinado, se eligirá aleatoriamente el patrón con el que se desea entrenar y la red neuronal empezará a procesarlo con 
         * todas los perceptrones (neuronas); reajustando sus pesos cuando sea necesario.
         * 
         * @param patterns Parámetro de tipo vector de pares de matrices y vectores utilizado para representar una entrada
         * y su salida esperada.
        */
        void training(vector<pair<vector<vector<int>>, vector<int>>> patterns) {
            int n = patterns.size();
            for (int iteration = 0; iteration < iterationsNumber; iteration++) {
                int index = rand() % n;
                vector<vector<int>> input = patterns[index].first;
                vector<int> expectedValue = patterns[index].second;
                vector<float> output = process_input(input);
                
                for (int i = 0; i < output.size(); i++) {
                    int outputValue = round(output[i]);
                    int error = expectedValue[i] - outputValue;
                    if (error != 0) {
                        perceptrons[i].adjust_weights(input, expectedValue[i], outputValue);
                        perceptrons[i].adjust_bias(expectedValue[i], outputValue);
                    };
                };
            };
        };

        /**
         * @brief Método utilizado para identificar o reconocer la matriz de entrada a través de la red neuronal 
         * previamente entrenada. Este devuelve un vector de tamaño N.
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros que representa la entrada 
         * de la red neuronal.
         * 
         * @return Vector de números enteros (respuesta de la red neuronal ante un patrón de entrada).
        */
        vector<int> resolve(vector<vector<int>> inputValues) {
            vector<float> output = {};
            for (int i = 0; i < perceptrons.size(); i++) {
                float result = perceptrons[i].activation_function(inputValues);
                output.push_back(result);
            };
            return competition(output);
        };

        /**
         * @brief Método utilizado para determinar que perceptron debe responder de acuerdo a la competencia 
         * entre ellos. La respuesta debe sobrepasar un mínimo valor esperado y, luego, el perceptron con el 
         * mejor acercamiento responderá. 
         * 
         * @param results Parámetro de tipo vector de números de coma flotante que representa los acercamientos 
         * de cada neurona de la red neuronal.
         * 
         * @return Vector de números enteros (vectores canónicos [respuesta única] o vector nulo [sin respuesta]).
        */
        vector<int> competition(vector<float> results) {
            vector<int> output = {0, 0, 0, 0, 0};
            int index = -1;
            float min_output = 0.15;
            for (int i = 0; i < results.size(); i++) {
                if (results[i] > min_output) {
                    min_output = results[i];
                    index = i;
                };
            };

            if (index != -1) {
                output[index] = 1;
            }
            return output;
        };

        /**
         * @brief Método utilizado para mostrar el vector de respuesta de la red neuronal en la consola.
         * 
         * @param output Parámetro de tipo vector de enteros que representa la salida de la red neuronal.
        */
        string show_results(vector<int> output) {
            vector<string> vowels = {"a", "e", "i", "o", "u"};
            for (int i = 0; i < output.size(); i++) {
                if (output[i] == 1) {
                    return "Esto es una vocal " + vowels[i] + ".";
                };
            };
            return "No reconozco esta letra.";
        };

        // KNOWLEDGE BASE

        /**
         * @brief Método utilizado para importar la base de conocimiento de la red neuronal a tarvés de un .txt.
        */
        void import_knowledge_base() {
            FileManager fileManager("base.txt", "read");
            for (Perceptron &perceptron : perceptrons) {
                vector<vector<float>> weights = fileManager.parse_float_matrix();
                float bias = fileManager.parse_float();
                fileManager.line_break();

                perceptron.set_weights(weights);
                perceptron.set_bias(bias);
            }
        };

        /**
         * @brief Método utilizado para exportar la base de conocimiento de la red neuronal a tarvés de un .txt.
        */
        void export_knowledge_base() {
            FileManager fileManager("base.txt", "write");

            for (Perceptron perceptron : perceptrons) {
                string perceptron_info = perceptron.weights_to_string();
                fileManager.write(perceptron_info);
            };
        };
};

/**
 * @brief Función que toma los patrones (a través de unos .txt de ejemplos) utilizados para 
 * el entrenamiento de una red neuronal.
 * 
 * @param expectedValues Parámetro de tipo vector de vectores de enteros que representan los posibles
 * valores esperados de los patrones (vocales).
 * 
 * @return Vector de pares ordenados de matrices de enteros y vectores de enteros 
 * (entradas de la red y su salida esperada).
*/
vector<pair<vector<vector<int>>, vector<int>>> get_patterns(vector<vector<int>> expectedValues) {
    string path = "patterns/";
    vector<string> files = {"ejemplosA.txt", "ejemplosE.txt", "ejemplosI.txt", "ejemplosO.txt", "ejemplosU.txt"};
    vector<pair<vector<vector<int>>, vector<int>>> patterns;
    for (int i = 0; i < files.size(); i++) {
        FileManager fileManager(path + files[i], "read");
        for (int j = 0; j < PATTERNS_NUM; j++) {
            vector<vector<int>> pattern = fileManager.parse_integer_matrix();
            fileManager.line_break();
            pair<vector<vector<int>>, vector<int>> pairIO = {pattern, expectedValues[i]};
            patterns.push_back(pairIO);
        };
    };
    return patterns;
};

int main(){
    srand(static_cast<unsigned int>(std::time(nullptr)));

    NeuralNetwork neuralNetwork(5, ITERATION_NUMBER);
    
    // Importa la base de conocimiento para el reconocimiento de vocales minusculas
    neuralNetwork.import_knowledge_base();

    // Matriz de entrada con la vocal a analizar
    FileManager fileManager("input.txt", "read");
    vector<vector<int>> inputMatrix = fileManager.parse_integer_matrix();

    // Resultados
    cout << "===============================\nJimmy Neuron necesita pensar...\n";
    vector<int> output = neuralNetwork.resolve(inputMatrix);
    string neuralResult = neuralNetwork.show_results(output);
    cout << neuralResult << "\n===============================";

    return 0;
};