import * as faceapi from 'face-api.js'

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons'

const fs = require('fs')
const fsPromises = fs.promises
const weightsPath = './weights'

const superName = 'facebase'
const impJson = `./${superName}.json`;

async function loadJson() {
    try {
        return fsPromises.readFile(impJson);
    } catch (err) {
        console.error('Error occured while reading directory!', err);
    }
}
  
async function run() {

    await faceDetectionNet.loadFromDisk(weightsPath)
    await faceapi.nets.faceLandmark68Net.loadFromDisk(weightsPath)
    await faceapi.nets.faceRecognitionNet.loadFromDisk(weightsPath)
    await faceapi.nets.ageGenderNet.loadFromDisk(weightsPath)  

    const faceMatcherJSON = JSON.parse(await loadJson())
    const faceMatcher = faceapi.FaceMatcher.fromJSON(faceMatcherJSON)

    const img = await canvas.loadImage(`test4.jpg`)

    const singleResult = await faceapi
        .detectSingleFace(img, faceDetectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptor()
        
    if (singleResult) {
        console.log(faceMatcher.labeledDescriptors)
        const bestMatch = faceMatcher.matchDescriptor(singleResult.descriptor)
        console.log(bestMatch.toString())
    }        
}

run()