

//LIBRERIAS OPENCV

import static com.googlecode.javacv.cpp.opencv_core.CV_32SC1;
import static com.googlecode.javacv.cpp.opencv_contrib.createLBPHFaceRecognizer;
import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;
import static com.googlecode.javacv.cpp.opencv_core.cvLoad;
import static com.googlecode.javacv.cpp.opencv_core.cvSetImageROI;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvSaveImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_BGR2GRAY;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_INTER_LINEAR;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvEqualizeHist;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvResize;
import static com.googlecode.javacv.cpp.opencv_objdetect.cvHaarDetectObjects;
import com.googlecode.javacpp.Loader;

import com.googlecode.javacv.cpp.opencv_contrib.FaceRecognizer;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_core.MatVector;
import com.googlecode.javacv.cpp.opencv_objdetect;
import com.googlecode.javacv.cpp.opencv_objdetect.CvHaarClassifierCascade;

//LIBRERIAS JAVA
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Properties;
import java.util.Set;



import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.objdetect.CascadeClassifier;


/**
 *
 * @author jonathan
 */
public class ReconocimientoCaras {
    
	
    private static String faceDataFolder = "C:/Users/vik/workspace/ReconocimientoFacial/resources/facerecognizer/data/";
    public static String imageDataFolder = faceDataFolder + "images\\";
    
    private static final String trainPath = "C:/users/vik/workspace/ReconocimientoFacial/resources/facerecognizer/data/images/training/";
    private static final Size trainSize = new Size(200, 150);
    private Mat trainingImages;
    private Mat trainingLabels;
    private CvSVM clasificador;
    //private static final String CASCADE_FILE = "C:\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
    //private static final String CASCADE_FILE = "C:/Users/vik/workspace/ReconocimientoFacial/resources/haarcascade_frontalface_alt.xml";
 
  
    private static final String BinaryFile = faceDataFolder + "frBinary.dat";
    public static final String personNameMappingFileName = faceDataFolder + "personNumberMap.properties";

    //final CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(CASCADE_FILE));
    private CascadeClassifier faceDetector = new CascadeClassifier(new File("C:/users/vik/workspace/ReconocimientoFacial/resources/haarcascade_frontalface_alt.xml").getAbsolutePath());
    final CascadeClassifier cascade = new CascadeClassifier(new File("C:/Users/vik/workspace/ReconocimientoFacial/resources/haarcascade_frontalface_alt.xml").getAbsolutePath());
   // final CascadeClassifier faceDetector = new CascadeClassifier("C:/Users/vik/workspace/ReconocimientoFacial/resources/haarcascade_frontalface_alt.xml");
    private Properties dataMap = new Properties();
    private static ReconocimientoCaras instance = new ReconocimientoCaras();

    public static final int NUM_IMAGES_PER_PERSON =10;
    double binaryTreshold = 100.0;
    int highConfidenceLevel = 70;

    FaceRecognizer ptr_binary = null;
    private FaceRecognizer fr_binary = null;
    
    private ReconocimientoCaras() {
            //createModels();
            //loadTrainingData();
    	trainingImages = new Mat();
		trainingLabels = new Mat();
    }
	public void train(String flag) {
		String path;
		if (flag.equalsIgnoreCase("A")) {
			path = trainPath + "ronaldo/";
		} else {
			path = trainPath + "terry/";
		}
	
		System.out.println("train de flag: " + flag);
		for (File file : new File(path).listFiles()) {
			Mat img = new Mat();
			// loading image to Mat object in grayscale
			Mat con = Highgui.imread(file.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_GRAYSCALE);
			// converting image to 32 bit floating point signed depth in one channel 
			con.convertTo(img, CvType.CV_32FC1, 1.0 / 255.0);
			// resizing to the unified size
			Imgproc.resize(img, img, trainSize);
			// adding reshaped sample element to the end of the matrix
			trainingImages.push_back(img.reshape(1, 1));
			System.out.println("trainingImages: "+ trainingImages);
			if (flag.equalsIgnoreCase("A")) {
				trainingLabels.push_back(Mat.ones(new Size(1, 1), CvType.CV_32FC1));
			} else {
				trainingLabels.push_back(Mat.zeros(new Size(1, 1), CvType.CV_32FC1));
			}
			
			
		}
	}
    
	

    public static ReconocimientoCaras getInstance() {
            return instance;
    }

    private void createModels() {
            ptr_binary = createLBPHFaceRecognizer(1, 8, 8, 8, binaryTreshold);
           
            fr_binary = ptr_binary;
    }
    
    public void detectar (ArrayList<Mat> caras)
    {
    	int i = 0;
    	CvSVMParams params = new CvSVMParams();
		// set linear kernel (no mapping, regression is done in the original feature space)
		params.set_kernel_type(CvSVM.LINEAR);
	// train SVM with images in trainingImages, labels in trainingLabels, given params with empty samples
		clasificador = new CvSVM(trainingImages, trainingLabels, new Mat(), new Mat(), params);
		// save generated SVM to file, so we can see what it generated
		clasificador.save("svm.xml");
		// loading previously saved file
		clasificador.load("svm.xml");
		// returnin, if there aren't any samples
		if (caras.isEmpty()) {
			System.out.println("No face detected");
			return;
		}
		for (Mat cara : caras) {
			Mat out = new Mat();
			// converting to 32 bit floating point in gray scale
			cara.convertTo(out, CvType.CV_32FC1);
			Highgui.imwrite("cara a detectar" + i + ".png", out);
			if (clasificador.predict(out.reshape(1, 1)) == 1.0) {
				System.out.println("Detected Ronaldo face");
			} else {
				System.out.println("Detected Terry face");
			}
			i++;
		}
    }
    



    /*protected ArrayList<Mat> detectFace2(String filename) {
        CvSeq faces = null;
        Loader.load(opencv_objdetect.class);
        int i=0;
        ArrayList<Mat> faceslist = new ArrayList<Mat>();
        Mat image = Highgui.imread(filename, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);
        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));
        for (Rect face : faceDetections.toArray()) {
			//Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
        	Mat outFace = image.submat(face);
        	
			//Imgproc.rectangle(image, new Point(face.x, face.y), new Point(face.x + face.width, face.y + face.height), new Scalar(0, 255, 0));
        	faceslist.add(outFace);
			Highgui.imwrite("face" + i + ".png", outFace);
			
			
		}
        return faceslist;
       
}*/

    public String identifyFace(IplImage image) {
            String personName = "";
            Set keys = dataMap.keySet();

            if (keys.size() > 0) {
                    int[] ids = new int[1];
                    double[] distance = new double[1];
                    int result = -1;

                            fr_binary.predict(image, ids, distance);
                            result = ids[0];

                            if (result > -1 && distance[0]<highConfidenceLevel) {
                                    personName = (String) dataMap.get("" + result);
                            }
            }

            return personName;
    }

    public ArrayList<Mat> obtenerRostroEntrada(String filepath)
    {
    	int i = 0;
    	ArrayList<Mat> caras = new ArrayList<Mat>();
    	Mat image = Highgui.imread(filepath, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
    	MatOfRect faceDetections = new MatOfRect();
    	faceDetector.detectMultiScale(image, faceDetections);
    	for (Rect face : faceDetections.toArray()) {
    		Mat outFace = image.submat(face);
			// resizing mouth to the unified size of trainSize
			Imgproc.resize(outFace, outFace, trainSize);
			Highgui.imwrite("caradetectada" + i + ".png", outFace);
			caras.add(outFace);
			i++;
    	}
    	return caras;
    	
    	
    }
    public boolean learnNewFace(String personName, IplImage[] images) throws Exception {
            int memberCounter = dataMap.size();
            if(dataMap.containsValue(personName)){
                    Set keys = dataMap.keySet();
                    Iterator ite = keys.iterator();
                    while (ite.hasNext()) {
                            String personKeyForTraining = (String) ite.next();
                            String personNameForTraining = (String) dataMap.getProperty(personKeyForTraining);
                            if(personNameForTraining.equals(personName)){
                                    memberCounter = Integer.parseInt(personKeyForTraining);
                            }
                    }
            }
            dataMap.put("" + memberCounter, personName);
            storeTrainingImages(personName, images);
            retrainAll();

            return true;
    }


    public IplImage preprocessImage(IplImage image, CvRect r){
            IplImage gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
            IplImage roi = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
            CvRect r1 = new CvRect(r.x()-10, r.y()-10, r.width()+10, r.height()+10);
            cvCvtColor(image, gray, CV_BGR2GRAY);
            cvSetImageROI(gray, r1);
            cvResize(gray, roi, CV_INTER_LINEAR);
            cvEqualizeHist(roi, roi);
            return roi;
    }

    private void retrainAll() throws Exception {
            Set keys = dataMap.keySet();
            if (keys.size() > 0) {
                    MatVector trainImages = new MatVector(keys.size() * NUM_IMAGES_PER_PERSON);
                    CvMat trainLabels = CvMat.create(keys.size() * NUM_IMAGES_PER_PERSON, 1, CV_32SC1);
                    Iterator ite = keys.iterator();
                    int count = 0;

                    System.err.print("Cargando imagenes para entrenamiento ...");
                    while (ite.hasNext()) {
                            String personKeyForTraining = (String) ite.next();
                            String personNameForTraining = (String) dataMap.getProperty(personKeyForTraining);
                            IplImage[] imagesForTraining = readImages(personNameForTraining);
                            
                            for (int i = 0; i < imagesForTraining.length; i++) {
                                    trainLabels.put(count, 0, Integer.parseInt(personKeyForTraining));
                                    IplImage grayImage = IplImage.create(imagesForTraining[i].width(), imagesForTraining[i].height(), IPL_DEPTH_8U, 1);
                                    cvCvtColor(imagesForTraining[i], grayImage, CV_BGR2GRAY);
                                    trainImages.put(count,grayImage);
                                    count++;
                            }
                    }

                    System.err.println("hecho.");

                    System.err.print("Realizando entrenamiento ...");
                    fr_binary.train(trainImages, trainLabels);
                    System.err.println("hecho.");
                    storeTrainingData();
            }

    }

    private void loadTrainingData() {
            try {
                    File personNameMapFile = new File(personNameMappingFileName);
                    if (personNameMapFile.exists()) {
                            FileInputStream fis = new FileInputStream(personNameMappingFileName);
                            dataMap.load(fis);
                            fis.close();
                    }

                    File binaryDataFile = new File(BinaryFile);
                    binaryDataFile.createNewFile();
                    fr_binary.load(BinaryFile);
                    System.err.println("hecho");


            } catch (Exception e) {
                    e.printStackTrace();
            }
    }

    private void storeTrainingData() throws Exception {
            System.err.print("Almacenando modelos ...");

            File binaryDataFile = new File(BinaryFile);
            if (binaryDataFile.exists()) {
                    binaryDataFile.delete();
            }
            fr_binary.save(BinaryFile);

            File personNameMapFile = new File(personNameMappingFileName);
            if (personNameMapFile.exists()) {
                    personNameMapFile.delete();
            }
            FileOutputStream fos = new FileOutputStream(personNameMapFile, false);
            dataMap.store(fos, "");
            fos.close();

            System.err.println("hecho.");
    }


    public void storeTrainingImages(String personName, IplImage[] images) {
            for (int i = 0; i < images.length; i++) {
                    String imageFileName = imageDataFolder + "training\\" + personName + "_" + i + ".bmp";
                    File imgFile = new File(imageFileName);
                    if (imgFile.exists()) {
                            imgFile.delete();
                    }
                    cvSaveImage(imageFileName, images[i]);
            }
    }

    private IplImage[] readImages(String personName) {
            File imgFolder = new File(imageDataFolder);
            IplImage[] images = null;
            if (imgFolder.isDirectory() && imgFolder.exists()) {
                    images = new IplImage[NUM_IMAGES_PER_PERSON];
                    for (int i = 0; i < NUM_IMAGES_PER_PERSON; i++) {
                            String imageFileName = imageDataFolder + "training\\" + personName + "_" + i + ".bmp";
                            IplImage img = cvLoadImage(imageFileName);
                            images[i] = img;
                    }

            }
            return images;
    }
}
