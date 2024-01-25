#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/String.h>
#include <lab4_autonomous_driving/classifier.h>
#include <ros/package.h>
#include <fstream> // Include the file stream library
#include <signal.h> // Include the signal library



bool exitRequested = false;

cv::VideoWriter videoWriter; // Define the video writer
std::ofstream csvFile;      // Define the CSV file stream

geometry_msgs::Twist vel_msg;
ros::Publisher vel_pub;

std_msgs::String dir_msg;
ros::Publisher dir_pub;

Classifier *classifier;



// Signal handler function to handle Ctrl+C
void signalHandler(int signum) {
    if (signum == SIGINT) {
        ROS_INFO("Ctrl+C detected. Exiting gracefully...");

        // Set the exit flag to true
        exitRequested = true;

        // Release the video writer when done
        videoWriter.release();

        // Close the CSV file
        csvFile.close();

        // Exit the ROS node
        ros::shutdown();
    }
}




void imageCallback(const sensor_msgs::ImageConstPtr& msg) {

    if (exitRequested) {
        return; // Exit early if exit is requested
    }

    try {
        cv::Mat src = cv_bridge::toCvShare(msg, "bgr8")->image;

        if (!videoWriter.isOpened()) {
            // Initialize the video writer if not already opened
            int codec = cv::VideoWriter::fourcc('h', '2', '6', '4'); // Codec for .avi format
            int frameWidth = src.cols;
            int frameHeight = src.rows;
            int framesPerSecond = 30; // Adjust as needed
            videoWriter.open("output.mkv", codec, framesPerSecond, cv::Size(frameWidth, frameHeight));
        }

        videoWriter.write(src); // Write the frame to the video file

        // Your existing image processing code goes here

        // Publish direction and velocity messages
        // (Assuming you have the necessary topics and message types defined)
        Prediction pred = classifier->Classify(src, 1)[0];
        dir_msg.data = pred.first;
        dir_pub.publish(dir_msg);

        if (pred.first == "FORWARD") {
            vel_msg.linear.x = 0.7;
            vel_msg.angular.z = 0.0;
        } else if (pred.first == "LEFT") {
            vel_msg.linear.x = 0.2;
            vel_msg.angular.z = -0.5;
        } else if (pred.first == "RIGHT") {
            vel_msg.linear.x = 0.2;
            vel_msg.angular.z = 0.5;
        }
        vel_pub.publish(vel_msg);

        // Save the predictions to a CSV file
        csvFile << ros::Time::now() << "," << pred.first << std::endl;

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "drive_inference");
    ros::NodeHandle nh;
    
    // Set up the signal handler for Ctrl+C
    signal(SIGINT, signalHandler);

    image_transport::ImageTransport it(nh);

    // Subscribe to the raw usb camera image
    image_transport::Subscriber raw_image_sub = it.subscribe("/usb_cam/image_raw", 30, imageCallback);

    string base_path = ros::package::getPath("lab4_autonomous_driving");

    string model_file   = base_path + "/neuralnetwork/deploy.prototxt";
    string trained_file = base_path + "/neuralnetwork/models/train_iter_157.caffemodel";
    string mean_file    = base_path + "/resources/data/mean_image.binaryproto";
    string label_file   = base_path + "/neuralnetwork/labels.txt";
    classifier = new Classifier(model_file, trained_file, mean_file, label_file);

    dir_pub = nh.advertise<std_msgs::String>("/lab4_autonomous_driving/direction", 1000);
    vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);

    // Open the CSV file for writing
    csvFile.open("predictions.csv");
    csvFile << "Timestamp,Direction"<< std::endl;

    // Loop for processing images (you can add additional functionality here)

    ros::spin();

    // Release the video writer when done
    videoWriter.release();

    // Close the CSV file
    csvFile.close();

    return 0;
}
