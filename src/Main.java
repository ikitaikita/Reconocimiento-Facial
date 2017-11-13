/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSeqElem;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

/**
 *
 * @author jonathan
 */
public class Main {
	

    public static void main(String args[]){
       
        	System.out.println(Core.VERSION);
    		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    		//String path = trainPath + "faces/";
            ReconocimientoCaras reconocer = ReconocimientoCaras.getInstance();
            ArrayList<Mat> listacaras = reconocer.obtenerRostroEntrada("C:/users/vik/workspace/ReconocimientoFacial/terry_target.jpg");
            reconocer.train("A"); //ronaldo
            reconocer.train("B"); //terry
            reconocer.detectar(listacaras);
           
            //Reconocimiento
        

    }
}
