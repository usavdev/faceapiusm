import * as faceapi from 'face-api.js'

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons'

export module findFaceMatch {
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
    
    async function run(imgPath) {

        await faceDetectionNet.loadFromDisk(weightsPath)
        await faceapi.nets.faceLandmark68Net.loadFromDisk(weightsPath)
        await faceapi.nets.faceRecognitionNet.loadFromDisk(weightsPath)
        await faceapi.nets.ageGenderNet.loadFromDisk(weightsPath)  

        const faceMatcherJSON = JSON.parse(await loadJson())
        const faceMatcher = faceapi.FaceMatcher.fromJSON(faceMatcherJSON)

        const img = await canvas.loadImage(imgPath)
        const imgOutPath = imgPath.replace(/.jpg/i, '_out.jpg')

        const singleResult = await faceapi
            .detectSingleFace(img, faceDetectionOptions)
            .withFaceLandmarks()
            .withFaceDescriptor()
            
        if (singleResult) {
            // console.log(faceMatcher.labeledDescriptors)
            const bestMatch = faceMatcher.matchDescriptor(singleResult.descriptor)
            const queryDrawBoxes = new faceapi.draw.DrawBox(singleResult.detection.box, { label: bestMatch.toString() })
            const outQuery = faceapi.createCanvasFromMedia(img)
            queryDrawBoxes.draw(outQuery)            
            fs.writeFileSync(imgOutPath, (outQuery as any).toBuffer('image/jpeg'))            
            return {status: 'ok', bestmatch: bestMatch, outpath: imgOutPath}
        } else {
            return {status: 'error', desc:'Can`t find face in this image'}
        }       
    }

    export async function runner(imgPath = 'test.jpg') {
        return await run(imgPath)
    }

}

