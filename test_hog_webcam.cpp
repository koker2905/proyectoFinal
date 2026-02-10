#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <ctime>
#include <fstream>

using namespace cv;
using namespace cv::ml;
using namespace std;

long getMemoryUsageKB() {
    ifstream file("/proc/self/statm");
    long size = 0;
    file >> size;
    return size * 4;
}

int main() {
    // ================= HOG CONFIG =================
    HOGDescriptor hog(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9);

    Ptr<SVM> svm = SVM::load("hog_svm.yml");
    if (svm.empty()) {
        cout << "❌ No se pudo cargar hog_svm.yml" << endl;
        return -1;
    }

    Mat sv = svm->getSupportVectors();
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);
    Mat detector = Mat::zeros(sv.cols + 1, 1, CV_32F);
    memcpy(detector.ptr<float>(), sv.ptr<float>(), sv.cols * sizeof(float));
    detector.at<float>(sv.cols) = (float)-rho;
    hog.setSVMDetector(detector);

    // ================= CAMERA =================
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 360);

    if (!cap.isOpened()) return -1;

    // Reducimos el FPS objetivo para que sea realista con el procesamiento
    int fps_video = 15; 
    double record_seconds = 5.0;
    bool recording = false;
    time_t last_trigger = 0;
    VideoWriter writer;
    string current_video;

    double record_start_tick = 0.0;
    int frame_count = 0;
    double fps = 0.0;
    double t0 = (double)getTickCount();
    long mem_kb = 0;
    double mem_t0 = t0;

    int person_frames = 0;
    const int REQUIRED_FRAMES = 3; // Bajamos esto para compensar el skip
    int hog_skip = 5; // Saltamos más frames para ganar fluidez
    int hog_counter = 0;

    Mat frame, gray, small_frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 1. ESCALADO PARA DETECCIÓN (Clave para la velocidad)
        // Reducimos a la mitad solo para HOG, no para grabar
        resize(frame, small_frame, Size(320, 180));

        // FPS Calculation
        frame_count++;
        double t1 = (double)getTickCount();
        double elapsed = (t1 - t0) / getTickFrequency();
        if (elapsed >= 1.0) {
            fps = frame_count / elapsed;
            frame_count = 0;
            t0 = t1;
        }

        if (recording) {
            writer.write(frame);
        }

        bool detected_this_frame = false;
        hog_counter++;

        if (hog_counter % hog_skip == 0) {
            vector<Rect> detections;
            vector<double> weights;

            // Ajustamos scaleFactor a 1.2 (Mucho más rápido que 1.05)
            hog.detectMultiScale(small_frame, detections, weights, 0.0, Size(8,8), Size(32,32), 1.2, 2);

            for (size_t i = 0; i < detections.size(); i++) {
                if (weights[i] < 0.7) continue; // Ajusta este umbral según tu SVM

                // Escalamos los rectángulos de vuelta al tamaño original (x2)
                Rect r = detections[i];
                r.x *= 2; r.y *= 2; r.width *= 2; r.height *= 2;

                rectangle(frame, r, Scalar(0,255,0), 2);
                detected_this_frame = true;
            }

            if (detected_this_frame) person_frames++;
            else person_frames = max(0, person_frames - 1);
        }

        // Métricas en pantalla
        putText(frame, format("FPS Reales: %.2f", fps), Point(10,25), 1, 1.2, Scalar(0,255,0), 2);
        if (recording) putText(frame, "● GRABANDO", Point(10, 50), 1, 1.2, Scalar(0,0,255), 2);

        time_t now = time(nullptr);
        if (person_frames >= REQUIRED_FRAMES && !recording && difftime(now, last_trigger) > 8) {
            current_video = "detected_" + to_string(now) + ".mp4";
            // IMPORTANTE: Usamos los FPS reales actuales para el VideoWriter
            writer.open(current_video, VideoWriter::fourcc('m','p','4','v'), (fps > 0 ? fps : 15), frame.size());
            
            if (writer.isOpened()) {
                recording = true;
                record_start_tick = (double)getTickCount();
                person_frames = 0;
            }
        }

        if (recording) {
            double rec_elapsed = ((double)getTickCount() - record_start_tick) / getTickFrequency();
            if (rec_elapsed >= record_seconds) {
                writer.release();
                recording = false;
                last_trigger = now;
                cout << "✅ Video guardado: " << current_video << endl;
                // Aquí podrías disparar tu comando curl
            }
        }

        imshow("HOG Optimizado", frame);
        if (waitKey(1) == 27) break;
    }

    return 0;
}