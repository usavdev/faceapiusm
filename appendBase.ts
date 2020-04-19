import * as faceapi from 'face-api.js'

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons'

const fs = require('fs')
const fsPromises = fs.promises
const weightsPath = './weights'

const superName = 'facebase'
const impJson = `./${superName}.json`;

const newJson = `./tanya.json`;

async function loadJson(facebasePath) {
    try {
        return fsPromises.readFile(facebasePath);
    } catch (err) {
        console.error('Error occured while reading directory!', err);
    }
}
  
async function run() {

    const faceMatcherJSONCommon = JSON.parse(await loadJson(impJson))
    const faceMatcherCommon = faceapi.FaceMatcher.fromJSON(faceMatcherJSONCommon)

    const faceMatcherJSONNew = JSON.parse(await loadJson(newJson))
    const faceMatcherNew = faceapi.FaceMatcher.fromJSON(faceMatcherJSONNew)
    
    const faceMatcher = new faceapi.FaceMatcher(faceMatcherCommon.labeledDescriptors.concat(faceMatcherNew.labeledDescriptors))  
    const faceMatcherJSON = JSON.stringify(faceMatcher.toJSON())
    fs.writeFileSync(`${superName}.json`, faceMatcherJSON)     
     
}

run()