#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>
#include <Face.h>
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    
    //strcpy(argv[1], "haarcascade_frontalface_alt.xml");
    //strcpy(argv[2], "facemeshUNITY.bytes");
    if (argc == 3) {
        Mat frame;
        //--- INITIALIZE VIDEOCAPTURE
        VideoCapture cap;
        // open the default camera using default API
        // cap.open(0);
        // OR advance usage: select any API backend
        int deviceID = 0;             // 0 = open default camera

        // open selected camera using selected API
        cap.open(deviceID);
        // check if we succeeded
        if (!cap.isOpened()) {
            cerr << "ERROR! Unable to open camera\n";
            return -1;
        }


        vector<Point> contour_left;
        vector<Point> contour_right;

        vector<vector<Point>> contour;

        CascadeClassifier face_cascade;
        face_cascade.load(argv[1]);
        if (face_cascade.empty())
            // if(!face_cascade.load("D:\\opencv2410\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"))
        {
            cerr << "Error Loading XML file" << endl;
            return 0;
        }

        facemesh landmark = facemesh(argv[2]);
        for (;;)
        {

            contour_left.clear();
            contour_right.clear();
            contour.clear();
            // wait for a new frame from camera and store it into 'frame'
            cap.read(frame);
            // check if we succeeded
            if (frame.empty()) {
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            // show live and wait for a key with timeout long enough to show images
            

            

            
            // Detect faces
            std::vector<Rect> faces;
            face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
            

            int id = 0;
            int mx = 0;
            if (faces.size() == 0) continue;
            
            // Draw circles on the detected faces
            for (int i = 0; i < faces.size(); i++)
            {
                if (mx < faces[i].width * faces[i].height) {
                    id = i;
                    mx = faces[i].width * faces[i].height;
                }
            }

            Rect roi;
            roi.x = faces[id].x;
            roi.y = faces[id].y;
            roi.width = faces[id].width;
            roi.height = faces[id].height;

            int wei = frame.size().width;
            int hei = frame.size().height;
            if (roi.width * roi.height * 1.0 < wei * hei * 0.07) {
                continue;
            }
            
            Mat crop = frame(roi);

            Mat res;
            res = crop.clone();
            // opencv read stuff in BGR
            cvtColor(crop, crop, COLOR_BGR2RGB);
            //resize(img, img, Size(192, 192), 0.0, 0.0, INTER_AREA);
            crop.convertTo(crop, CV_32FC3);
            // normalize values
            crop = crop / 255.0;

            int h = crop.size().height;
            int w = crop.size().width;

            resize(crop, crop, Size(192, 192));


            // use a 3d array cause fuck opencv mats
            float*** imgArray = (float***)facemesh::createArray(192, 192, 3, sizeof(float));
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < 192; i++) {
                    for (int j = 0; j < 192; j++) {
                        imgArray[i][j][k] = crop.at<Vec3f>(i, j)[k];
                    }
                }
            }


            double rx = w / 192.0;
            double ry = h / 192.0;
            
            

            landmark.forwardProp(imgArray);

            Mat mask = Mat(h, w, res.type());
            mask = Scalar(0, 0, 0);



            int left_arr[] = { 226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31 };
            int right_arr[] = { 464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453 };


            for (int i = 0; i < 16; i++) contour_left.push_back(Point(landmark.l20[left_arr[i] * 3] * rx, landmark.l20[left_arr[i] * 3 + 1] * ry));
            for (int i = 0; i < 16; i++) contour_right.push_back(Point(landmark.l20[right_arr[i] * 3] * rx, landmark.l20[right_arr[i] * 3 + 1] * ry));

            contour.push_back(contour_left);
            drawContours(mask, contour, -1, Scalar(255, 255, 255), -1);
            cvtColor(mask, mask, CV_BGR2GRAY);
            contour.pop_back();
            contour.push_back(contour_right);
            drawContours(mask, contour, -1, Scalar(255, 255, 255), -1);
            contour.pop_back();
            Mat result;

            bitwise_and(res, res, result, mask);

            imshow("Result", result);

            Mat gray_result, th_result;
            cvtColor(result, gray_result, CV_BGR2GRAY);
            threshold(gray_result, th_result, 50, 255, THRESH_BINARY);


            Rect lefteye_rect = boundingRect(contour_left);
            Mat lefteye_cropped = th_result(lefteye_rect);
            Rect righteye_rect = boundingRect(contour_right);
            Mat righteye_cropped = th_result(righteye_rect);


            double lefteye_area = contourArea(contour_left);
            double righteye_area = contourArea(contour_right);

            int lefteye_white_area = countNonZero(lefteye_cropped);
            int righteye_white_area = countNonZero(righteye_cropped);

            if (lefteye_white_area == 0 || righteye_white_area == 0) continue;
          

            int lefteye_cover_rate = int(100 - lefteye_white_area * 100 / lefteye_area);
            int righteye_cover_rate = int(100 - righteye_white_area * 100 / righteye_area);

            if (lefteye_cover_rate > 95) lefteye_cover_rate = 100;
            else if (lefteye_cover_rate < 25) lefteye_cover_rate = 0;
            if (righteye_cover_rate > 95)  righteye_cover_rate = 100;
            else if (righteye_cover_rate < 25) righteye_cover_rate = 0;

            cout << lefteye_cover_rate << "====" << righteye_cover_rate << endl;
            
            if (waitKey(1) >= 0)
                break;
            facemesh::freeArray(192, 192, 3, imgArray);
        }
        
        
    }
    else if(argc == 4){
   
        Mat image;
        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
        // Load Face cascade (.xml file)
        CascadeClassifier face_cascade;
        face_cascade.load(argv[2]);

        if (face_cascade.empty())
            // if(!face_cascade.load("D:\\opencv2410\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"))
        {
            cerr << "Error Loading XML file" << endl;
            return 0;
        }

        // Detect faces
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

        int id = 0;
        int mx = 0;
        // Draw circles on the detected faces
        for (int i = 0; i < faces.size(); i++)
        {
            if (mx < faces[i].width * faces[i].height) {
                id = i;
                mx = faces[i].width * faces[i].height;
            }
        }

        Rect roi;
        roi.x = faces[id].x;
        roi.y = faces[id].y;
        roi.width = faces[id].width;
        roi.height = faces[id].height;

        Mat crop = image(roi);
        Mat res;
        res = crop.clone();
        // opencv read stuff in BGR
        cvtColor(crop, crop, COLOR_BGR2RGB);
        //resize(img, img, Size(192, 192), 0.0, 0.0, INTER_AREA);
        crop.convertTo(crop, CV_32FC3);
        // normalize values
        crop = crop / 255.0;

        int h = crop.size().height;
        int w = crop.size().width;

        resize(crop, crop, Size(192, 192));


        // use a 3d array cause fuck opencv mats
        float*** imgArray = (float***)facemesh::createArray(192, 192, 3, sizeof(float));
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 192; i++) {
                for (int j = 0; j < 192; j++) {
                    imgArray[i][j][k] = crop.at<Vec3f>(i, j)[k];
                }
            }
        }


        double rx = w / 192.0;
        double ry = h / 192.0;
        //string PATH = "facemeshUNITY.bytes";
        string PATH = argv[3];
        facemesh landmark = facemesh(PATH);

        landmark.forwardProp(imgArray);

        Mat mask = Mat(h, w, res.type());
        mask = Scalar(0, 0, 0);


        vector<Point> contour_left;
        vector<Point> contour_right;

        vector<vector<Point>> contour;

        int left_arr[] = { 226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31 };
        int right_arr[] = { 464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453 };


        for (int i = 0; i < 16; i++) contour_left.push_back(Point(landmark.l20[left_arr[i] * 3] * rx, landmark.l20[left_arr[i] * 3 + 1] * ry));
        for (int i = 0; i < 16; i++) contour_right.push_back(Point(landmark.l20[right_arr[i] * 3] * rx, landmark.l20[right_arr[i] * 3 + 1] * ry));

        contour.push_back(contour_left);
        drawContours(mask, contour, -1, Scalar(255, 255, 255), -1);
        cvtColor(mask, mask, CV_BGR2GRAY);
        contour.pop_back();
        contour.push_back(contour_right);
        drawContours(mask, contour, -1, Scalar(255, 255, 255), -1);
        contour.pop_back();
        Mat result;

        bitwise_and(res, res, result, mask);

        Mat gray_result, th_result;
        cvtColor(result, gray_result, CV_BGR2GRAY);
        threshold(gray_result, th_result, 50, 255, THRESH_BINARY);


        Rect lefteye_rect = boundingRect(contour_left);
        Mat lefteye_cropped = th_result(lefteye_rect);
        Rect righteye_rect = boundingRect(contour_right);
        Mat righteye_cropped = th_result(righteye_rect);

        double lefteye_area = contourArea(contour_left);
        double righteye_area = contourArea(contour_right);

        int lefteye_white_area = countNonZero(lefteye_cropped);
        int righteye_white_area = countNonZero(righteye_cropped);

        if (lefteye_white_area == 0 || righteye_white_area == 0) return 0;
        int lefteye_cover_rate = int(100 - lefteye_white_area * 100 / lefteye_area);
        int righteye_cover_rate = int(100 - righteye_white_area * 100 / righteye_area);

        if (lefteye_cover_rate > 95) lefteye_cover_rate = 100;
        else if (lefteye_cover_rate < 25) lefteye_cover_rate = 0;
        if (righteye_cover_rate > 95)  righteye_cover_rate = 100;
        else if (righteye_cover_rate < 25) righteye_cover_rate = 0;

        cout <<lefteye_cover_rate << "====" << righteye_cover_rate << endl;

        facemesh::freeArray(192, 192, 3, imgArray);
        

    }
    else {
        cout << "Failed Parameters" << endl;
    }
    return 0;
}


