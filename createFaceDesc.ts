import * as faceapi from 'face-api.js'

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons'

const fs = require('fs')
const fsPromises = fs.promises
const weightsPath = './weights'

const superName = 'tanya'
const impDir = `./${superName}`;

async function listDir() {
    try {
        return fsPromises.readdir(impDir);
    } catch (err) {
        console.error('Error occured while reading directory!', err);
    }
}

async function procesFiles(array) {
    await faceDetectionNet.loadFromDisk(weightsPath)
    await faceapi.nets.faceLandmark68Net.loadFromDisk(weightsPath)
    await faceapi.nets.faceRecognitionNet.loadFromDisk(weightsPath)
    await faceapi.nets.ageGenderNet.loadFromDisk(weightsPath)

    const descriptors = [];
    var i = 0;

    for (const file of array) {
        i++;        
        const img = await canvas.loadImage(`${impDir}/${file}`)
        const results = await faceapi.detectAllFaces(img, faceDetectionOptions)
            .withFaceLandmarks()
            .withFaceDescriptors()
        
        results.forEach(async (item, j) => {
            //const box = item.box;
            const box = item.alignedRect.box;

            const x = box.x
            const y = box.y
            const w = box.width
            const h = box.height

            const canvas2 = canvas.createCanvas(w, h)
            const ctx = canvas2.getContext('2d')
            ctx.drawImage(img, -x, -y)

            saveFile(`face_${i}_${j}.jpg`, canvas2.toBuffer('image/jpeg'))  
            
            descriptors.push(item.descriptor)
       
        })

        console.log(`(${i}/${array.length}):${file}`)
    }

    console.log('Done procesFiles!')

    return descriptors
  }
  

async function run() {
        
  const files = await listDir()
  const descriptors = await procesFiles(files)

  const labelDesc = new faceapi.LabeledFaceDescriptors(
    superName,
    descriptors
  )

  const faceMatcher = new faceapi.FaceMatcher(labelDesc)  
  const faceMatcherJSON = JSON.stringify(faceMatcher.toJSON())

  fs.writeFileSync(`${superName}.json`, faceMatcherJSON)
       
}

run()