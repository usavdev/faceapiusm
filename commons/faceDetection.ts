import * as faceapi from 'face-api.js';

 // export const faceDetectionNet = faceapi.nets.ssdMobilenetv1
 export const faceDetectionNet = faceapi.nets.tinyFaceDetector

// SsdMobilenetv1Options
const minConfidence = 0.6

// TinyFaceDetectorOptions
/*
const inputSize = 4480
const scoreThreshold = 0.6
*/

const inputSize = 640
const scoreThreshold = 0.6

function getFaceDetectorOptions(net: faceapi.NeuralNetwork<any>) {
  return net === faceapi.nets.ssdMobilenetv1
    ? new faceapi.SsdMobilenetv1Options({ minConfidence })
    : new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold })
}

export const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet)