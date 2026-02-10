#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {

    string posDir = "hog_dataset/positives/";
    string negDir = "hog_dataset/negatives/";

    vector<String> posFiles, negFiles;
    glob(posDir + "*.png", posFiles);
    glob(negDir + "*.png", negFiles);

    cout << "Positivos: " << posFiles.size() << endl;
    cout << "Negativos: " << negFiles.size() << endl;

    // 🔹 Configuración HOG clásica peatones
    HOGDescriptor hog(
        Size(64,128),   // winSize
        Size(16,16),    // blockSize
        Size(8,8),      // blockStride
        Size(8,8),      // cellSize
        9               // bins
    );

    Mat trainingData;
    vector<int> labels;

    // 🔹 Positivos
    for (const auto& file : posFiles) {
        Mat img = imread(file, IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        resize(img, img, Size(64,128));

        vector<float> descriptors;
        hog.compute(img, descriptors);

        trainingData.push_back(Mat(descriptors).t());
        labels.push_back(+1);
    }

    // 🔹 Negativos
    for (const auto& file : negFiles) {
        Mat img = imread(file, IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        resize(img, img, Size(64,128));

        vector<float> descriptors;
        hog.compute(img, descriptors);

        trainingData.push_back(Mat(descriptors).t());
        labels.push_back(-1);
    }

    cout << "Total muestras: " << trainingData.rows << endl;

    // 🔹 Entrenar SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setC(0.01);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-3));

    svm->train(trainingData, ROW_SAMPLE, labels);

    svm->save("hog_svm.yml");

    cout << "✅ Entrenamiento finalizado y modelo guardado" << endl;

    return 0;
}