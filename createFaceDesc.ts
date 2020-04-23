import * as faceapi from 'face-api.js'

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons'

const fs = require('fs')
const fsPromises = fs.promises
const weightsPath = './weights'

const superName = 'tanya'
const impDir = `./${superName}`;

const minRate = 40
const originalPref = 'original'

async function listDir() {
    try {
        return fsPromises.readdir(impDir);
    } catch (err) {
        console.error('Error occured while reading directory!', err);
    }
}

async function processFiles(array, originalMatcher:faceapi.FaceMatcher = null) {
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

            var procent = 0
            if (originalMatcher) {
                const bestMatch = originalMatcher.matchDescriptor(item.descriptor)
                procent = +(bestMatch.distance * 100).toFixed(0)
            }
            procent = 100 - +procent

            if (procent >= minRate) {
                saveFile(`face_${procent}_${i}_${j}.jpg`, canvas2.toBuffer('image/jpeg'))              
                descriptors.push(item.descriptor)
            }
            
       
        })

        console.log(`(${i}/${array.length}):${file}`)
    }

    console.log('Done processFiles!')

    return descriptors
  }
  

async function run() {
        
  const filesAll = await listDir()
  const filesOriginals = filesAll.filter(file => file.startsWith(originalPref))
  const filesLoaded =filesAll.filter(file => !file.startsWith(originalPref))

  const descriptorsOriginals = await processFiles(filesOriginals)

  const labelDescOriginal = new faceapi.LabeledFaceDescriptors(
    superName,
    descriptorsOriginals
  )
  const faceMatcherOriginal = new faceapi.FaceMatcher(labelDescOriginal)  

  const descriptorsLoaded = await processFiles(filesLoaded,faceMatcherOriginal)

  const descriptorsTotal = descriptorsOriginals.concat(descriptorsLoaded)
  console.log('descriptorsTotal size:',descriptorsTotal.length)
  const labelDescTotal = new faceapi.LabeledFaceDescriptors(
    superName,
    descriptorsTotal
  )
  const faceMatcherTotal = new faceapi.FaceMatcher(labelDescTotal)    

  const faceMatcherJSON = JSON.stringify(faceMatcherTotal.toJSON())
  
  fs.writeFileSync(`${superName}.json`, faceMatcherJSON) 
  
}

run()