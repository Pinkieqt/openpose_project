// ----------------------------- OpenPose C++ API Tutorial - Example 6 - Face from Image -----------------------------
// It reads an image and the face location, process it, and displays the face keypoints. In addition,
// it includes all the OpenPose configuration flags.
// Input: An image and the face rectangle locations.
// Output: OpenPose face keypoint detection.
// NOTE: This demo is auto-selecting the following flags: `--body 0 --face --face_detector 2`
#include <stdio.h>
#include <iostream>
// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "examples/media/images_hol_car/img00067.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
//true - vypíše jen do jsonu
//false - vypíš a zobrazí výsledek na obrazovuk
DEFINE_bool(no_display,                 true,
    "Enable to disable the visual display.");


/*
//1
//2
//3
//4
*/

//Složka s fotkama
//const std::string inputFilesPath = "examples/media/images_hol_car";
//const std::string inputFilesPath = "examples/media/images_fus_car";
const std::string inputFilesPath = "examples/media/coughs";

//Output soubor
const std::string outputFilePath = "examples/user_code/points_coughs.csv";
//const std::string outputFilePath66 = "examples/user_code/pointsVal66.txt";
//const std::string outputFilePath = "examples/user_code/pointsVal2xxx.txt";

//Pozice obliceje
//const auto posOfFace = op::Rectangle<float>{430.44f, 30.44f, 550.33f, 550.33f};
//const auto posOfFace = op::Rectangle<float>{577.44f, 85.44f, 567.33f, 567.33f};

//pozice coughs
const auto posOfFace = op::Rectangle<float>{484.44f, 4.44f, 688.33f, 688.33f};

//Počet kolik obrázků budeme samplovat
//hol - 356
//fus - 2089
//coughs - 575
const int numOfImages = 575;

//Počet kolik bodů máme na tváři - ppl * 70 * (x,y) + %
//210 na jednoho
const int numOfPoints = 210;

struct Points
{ 
    float x;
    float y;
    float z;
};

class FacePoints  
{  
    public:
        int numOfFace;  
        float Points[numOfPoints];
}; 

class MouthPoints  
{  
    public:
        int numOfFace;  
        Points point;
}; 




// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Testování detekce tváře", cvMat);
            cv::waitKey(0);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            //op::opLog("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            //op::opLog(datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            op::opLog(datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
            (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}




int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose EDITED demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Flagy podle kterých bude openpose pracovat
        FLAGS_body = 0;
        FLAGS_face = true;
        FLAGS_face_detector = 2;

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        //***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_
        /*Code to build, run and test

        #With run
        cd build/ && make -j`nproc` && cd .. && ./build/examples/user_code/myface.bin

        #With output to json
        cd build/ && make -j`nproc` && cd .. && ./build/examples/user_code/myface.bin --write_json ./../programming/keypoints

        */
        //***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_

        //Struktura ve ktere mám uložené veškeré data
        FacePoints fp[numOfImages];


        //0-358
        for(int i = 0; i < numOfImages; i++)
        {
            char buffer[256]; 
            sprintf(buffer, "%05d", i);
            std::string str(buffer);

            std::string img_path = inputFilesPath + "/img" + std::to_string(i) + ".jpg";

            //otevření img
            const cv::Mat cvImageToProcess = cv::imread(img_path);

            //nahrani img do matice openpose
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);

            //Lokace obličejů na obrázku - X, Y, Width, Height
            //width a height se musí rovnat - musí být tzv squared
            const std::vector<op::Rectangle<float>> faceRectangles{posOfFace};

            // nove datum
            auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
            datumsPtr->emplace_back();
            auto& datumPtr = datumsPtr->at(0);
            datumPtr = std::make_shared<op::Datum>();

            //naplnění maticí op
            datumPtr->cvInputData = imageToProcess;
            datumPtr->faceRectangles = faceRectangles;

            //processing
            opWrapper.emplaceAndPop(datumsPtr);

            if (datumsPtr != nullptr)
            {
                //práce s daty - každý datumsPtr má 70 pozic na obličeji
                fp[i].numOfFace = i;                  
                for(int j = 0; j < numOfPoints; j++)
                {
                    fp[i].Points[j] = datumsPtr->at(0)->faceKeypoints[j];
                }
                printf("%s %d %s", "Obrazek c.: ", i, "\n");
                //výpis dat do konzole
                //printKeypoints(datumsPtr);

                //tady to aby se zobrazilo okno, jinak se ukončí jen s výpisem - lze změnit nahoře
                if (!FLAGS_no_display)
                    display(datumsPtr);
            }
            else
                op::opLog("Image could not be processed.", op::Priority::High);

        }

        //Data do textového souboru
        std::ofstream outfile (outputFilePath);
        for (int i = 0; i < numOfImages; i++)
        {
            outfile << i+1;
            outfile << "\t";
            for (int j = 0; j < numOfPoints; j++)
            {
                outfile << fp[i].Points[j];
                if(j < numOfPoints - 1)
                    outfile << "\t";
            }
            outfile << "\n";
        }
        outfile.close();



        //***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_

        //***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_***_

        // Info
        op::opLog("NOTE: In addition with the user flags, this demo has auto-selected the following flags:\n"
                "\t`--body 0 --face --face_detector 2`", op::Priority::High);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}


void generateTxtFromImage(std::string pathDir, std::string pathOutputFile, op::Rectangle<float> facePos, int numOfImgs, int numOfPnts )
{
        MouthPoints mp62[numOfImages];
        MouthPoints mp66[numOfImages];

        std::ofstream outfile (outputFilePath);
        for(int i = 0; i < numOfImages; i++) {
                outfile << mp62[i].point.x;
                outfile << ", ";
        }
        outfile.close();

        std::ofstream outfilex (outputFilePath);
        for(int i = 0; i < numOfImages; i++) {
                outfilex << mp66[i].point.x;
                outfilex << ", ";
        }
        outfilex.close();

}

