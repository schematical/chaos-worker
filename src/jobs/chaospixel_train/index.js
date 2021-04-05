/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const fs = require('fs');
const path = require('path');
const canvas = require('canvas');
const _ = require('underscore');
const tf = require('@tensorflow/tfjs');
const RunnableJobBase = require('../RunnableJobBase');
class ChaosPixelTrainJob extends RunnableJobBase{
    constructor(job) {
        super(job);

        let defaults = {
            validationSplit: 0.15,
            batchSize: 128,
            fineTuningEpochs: 100,
            initialTransferEpochs: 250,
            canvasSize: 224,
            topLayerGroupNames: [/*'conv_pw_9', 'conv_pw_10',*/ 'conv_pw_11'],
            gpu: false
        }
        this._jobMeta = _.extend(defaults, job.jobMeta);
        // Name of the layer that will become the top layer of the truncated base.
        this._jobMeta.topLayerName = `${this.jobMeta.topLayerGroupNames[this.jobMeta.topLayerGroupNames.length - 1]}_relu`;

        // Used to scale the first column (0-1 shape indicator) of `yTrue`
        // in order to ensure balanced contributions to the final loss value
        // from shape and bounding-box predictions.
        this._jobMeta.labelMultiplier = [this.jobMeta.canvasSize, 1, 1, 1, 1];

    }
    async loadAndShapeImage(imgSrc) {
        /*return new Promise((resolve, reject)=>{

            let imageEle = new Image();
            imageEle.onload = ()=>{
                return resolve(imageEle);
            }
            imageEle.src = imgSrc;
        });*/
        return new Promise((resolve, reject)=>{
            const fakeCanvas = canvas.createCanvas(this.jobMeta.canvasSize, this.jobMeta.canvasSize)
            //document.body.appendChild(fakeCanvas);
            const fakeCtx = fakeCanvas.getContext('2d');

            let imageEle = new canvas.Image();
            imageEle.style = `image-rendering: auto;
                image-rendering: crisp-edges;
                image-rendering: pixelated;`;
            imageEle.onload = ()=>{
                let scale = this.jobMeta.canvasSize / imageEle.width;
                /*if(imageEle.height < imageEle.width){
                    scale = this.options.canvasHeight / imageEle.height;
                }*/
                let width = imageEle.width * scale;
                let height = imageEle.height * scale;
                fakeCanvas.width = this.jobMeta.canvasSize;// width;
                fakeCanvas.height = this.jobMeta.canvasSize;//height;
                fakeCtx.scale(scale, scale);
                fakeCtx.mozImageSmoothingEnabled = false;
                fakeCtx.webkitImageSmoothingEnabled = false;
                fakeCtx.imageSmoothingEnabled = false;
                fakeCtx.msImageSmoothingEnabled = false;
                fakeCtx.oImageSmoothingEnabled = false;
                console.log("fakeCtx.imageSmoothingEnabled: ", fakeCtx.imageSmoothingEnabled);
                fakeCtx.fillStyle = 'green';
                fakeCtx.fillRect(0, 0, this.jobMeta.canvasSize, this.jobMeta.canvasSize);
                fakeCtx.drawImage(imageEle,0,0,   imageEle.width, imageEle.height); //width, height); //this.options.canvasWidth, this.options.canvasHeight);//
                fakeCtx.setTransform(1, 0, 0, 1, 0, 0);



                //imageEle.width = width;// this.options.canvasWidth;
                //imageEle.height = height;// this.options.canvasHeight;

                let newImageEle = new canvas.Image();
                newImageEle.style = `image-rendering: auto;
                image-rendering: crisp-edges;
                image-rendering: pixelated;`;
                newImageEle.onload = ()=>{

                    //document.body.removeChild(fakeCanvas);
                    return resolve(newImageEle);
                }
                newImageEle.src = fakeCanvas.toDataURL('image/bmp',1);


            }
            imageEle.src = imgSrc;
        });
    }
    async run(){
        const trainingData = await this.downloadFile(this.jobMeta.data_uri);// JSON.parse(fs.readFileSync('C:\\Users\\mlea\\WebstormProjects\\chaos-worker\\224.json').toString())//
        let tagsDict = {};
        let imageEleDict = {};
        let p = Promise.resolve();

        trainingData.forEach((image)=>{


            p = p.then(()=>{
                return canvas.loadImage(image.imgSrc)
            })
            .then(async  (imageEle)=>{
                imageEleDict[image.id] = await this.loadAndShapeImage(imageEle.src);
            });
            image.boxes.forEach((box)=>{
                box.tags.forEach((tag)=>{
                    if(!tagsDict[tag]) {
                        tagsDict[tag] = {
                            tag: tag,
                            id: Object.keys(tagsDict).length
                        }
                    }
                })
            })
        })
        await p;
        //tf.dispose([imageTensors, targetTensors]);




        let tfn;
        if (this.jobMeta.gpu) {
            this.log('Training using GPU.');
            tfn = require('@tensorflow/tfjs-node-gpu');
        } else {
            this.log('Training using CPU.');
            tfn = require('@tensorflow/tfjs-node');
        }

        const modelSaveURL = 'file://./tmp/' + this.job._id + '-v2.2';

        const tBegin = tf.util.now();

        let imageTensors = [];
        let targetTensors = [];
        trainingData.forEach((image)=>{

            image.boxes.forEach((box)=>{
                box.tags.forEach((tag)=> {
                    let data = tf.tidy(() => {
                        const canvasX = canvas.createCanvas();
                        canvasX.height = imageEleDict[image.id].height;
                        canvasX.width = imageEleDict[image.id].width;
                        canvasX.getContext('2d').drawImage(imageEleDict[image.id], 0, 0, imageEleDict[image.id].width, imageEleDict[image.id].height)
                        const imageTensor = tf.browser.fromPixels(canvasX).cast('float32');
                        const shapeClassIndicator = tagsDict[tag].id;
                        const targetTensor =
                            tf.tensor1d([shapeClassIndicator].concat(box.bbox));
                        return {image: imageTensor, target: targetTensor};
                    });
                    imageTensors.push(data.image);
                    targetTensors.push(data.target);
                })
            })
        })
        const images = tf.stack(imageTensors);
        const targets = tf.stack(targetTensors);

        const {model, fineTuningLayers} = await this.buildObjectDetectionModel();
        model.compile({
            loss: this.customLossFunction.bind(this),
            optimizer: tf.train.rmsprop(5e-3),
            metrics: ['accuracy']
        });
        model.summary();
        let callbacks = {
            onEpochEnd: (epoch, logs) => {
                this.log(
                    `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`);
                this.updateStatus({
                    phase: 'initial',
                    acc: logs.acc,
                    epoch: epoch
                })

            },
            onTrainEnd: async (logs) => {
                this.updateStatus({
                    phase: 'initial',
                    state: 'done'
                })
            }
        }
        /*let logDir = 'C:\\Users\\mlea\\WebstormProjects\\chaos-worker\\tmp\\logs';//__dirname + '/tmp/logs';
        callbacks = tfn.node.tensorBoard(logDir, {
            updateFreq: 'epoch'
        })*/
        // Initial phase of transfer learning.
        console.log('Phase 1 of 2: initial transfer learning');
        await model.fit(images, targets, {
            epochs: this.jobMeta.initialTransferEpochs,
            batchSize: this.jobMeta.batchSize,
            validationSplit: this.jobMeta.validationSplit,
            callbacks: callbacks
        });

        // Fine-tuning phase of transfer learning.
        // Unfreeze layers for fine-tuning.
        for (const layer of fineTuningLayers) {
            layer.trainable = true;
        }
        model.compile({
            loss: this.customLossFunction.bind(this),
            optimizer: tf.train.rmsprop(2e-3),
            metrics: ['accuracy']
        });
        model.summary();

        // Do fine-tuning.
        // The batch size is reduced to avoid CPU/GPU OOM. This has
        // to do with the unfreezing of the fine-tuning layers above,
        // which leads to higher memory consumption during backpropagation.
        console.log('Phase 2 of 2: fine-tuning phase');
        let fineCallbacks = {
            onEpochEnd: (epoch, logs) => {
                this.log(
                    `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`);
                this.updateStatus({
                    phase: 'fine',
                    acc: logs.acc,
                    epoch: epoch
                })

            },
            onTrainEnd: async (logs) => {
                this.updateStatus({
                    phase: 'fine',
                    state: 'done'
                })
            }
        }
        await model.fit(images, targets, {
            epochs: this.jobMeta.fineTuningEpochs,
            batchSize: this.jobMeta.batchSize / 2,
            validationSplit: this.jobMeta.validationSplit,
            callbacks: fineCallbacks
        });
        console.log("DOne!");
        // Save model.
        // First make sure that the base directory dists.
        const modelSavePath = modelSaveURL.replace('file://', '');
        const dirName = path.dirname(modelSavePath);
        if (!fs.existsSync(dirName)) {
            fs.mkdirSync(dirName);
        }
        await model.save(modelSaveURL);
        console.log(`Model training took ${(tf.util.now() - tBegin) / 1e3} s`);
        console.log(`Trained model is saved to ${modelSaveURL}`);
        console.log(
            `\nNext, run the following command to test the model in the browser:`);
        console.log(`\n  yarn watch`);
        await this.updateStatus({
            phase: 'all',
            state: 'done'
        })
    }








    /**
     * Custom loss function for object detection.
     *
     * The loss function is a sum of two losses
     * - shape-class loss, computed as binaryCrossentropy and scaled by
     *   `classLossMultiplier` to match the scale of the bounding-box loss
     *   approximatey.
     * - bounding-box loss, computed as the meanSquaredError between the
     *   true and predicted bounding boxes.
     * @param {tf.Tensor} yTrue True labels. Shape: [batchSize, 5].
     *   The first column is a 0-1 indicator for whether the shape is a triangle
     *   (0) or a rectangle (1). The remaining for columns are the bounding boxes
     *   for the target shape: [left, right, top, bottom], in unit of pixels.
     *   The bounding box values are in the range [0, CANVAS_SIZE).
     * @param {tf.Tensor} yPred Predicted labels. Shape: the same as `yTrue`.
     * @return {tf.Tensor} Loss scalar.
     */
    customLossFunction(yTrue, yPred) {
        return tf.tidy(() => {
            // Scale the the first column (0-1 shape indicator) of `yTrue` in order
            // to ensure balanced contributions to the final loss value
            // from shape and bounding-box predictions.
            return tf.metrics.meanSquaredError(yTrue.mul(this.jobMeta.labelMultiplier), yPred);
        });
    }

    /**
     * Loads MobileNet, removes the top part, and freeze all the layers.
     *
     * The top removal and layer freezing are preparation for transfer learning.
     *
     * Also gets handles to the layers that will be unfrozen during the fine-tuning
     * phase of the training.
     *
     * @return {tf.Model} The truncated MobileNet, with all layers frozen.
     */
    async loadTruncatedBase() {
        // TODO(cais): Add unit test.
        const mobilenet = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

        // Return a model that outputs an internal activation.
        const fineTuningLayers = [];
        const layer = mobilenet.getLayer(this.jobMeta.topLayerName);
        const truncatedBase =
            tf.model({inputs: mobilenet.inputs, outputs: layer.output});
        // Freeze the model's layers.
        for (const layer of truncatedBase.layers) {
            layer.trainable = false;
            for (const groupName of this.jobMeta.topLayerGroupNames) {
                if (layer.name.indexOf(groupName) === 0) {
                    fineTuningLayers.push(layer);
                    break;
                }
            }
        }

        tf.util.assert(
            fineTuningLayers.length > 1,
            `Did not find any layers that match the prefixes ${this.jobMeta.topLayerGroupNames}`);
        return {truncatedBase, fineTuningLayers};
    }

    /**
     * Build a new head (i.e., output sub-model) that will be connected to
     * the top of the truncated base for object detection.
     *
     * @param {tf.Shape} inputShape Input shape of the new model.
     * @returns {tf.Model} The new head model.
     */
    buildNewHead(inputShape) {
        const newHead = tf.sequential();
        newHead.add(tf.layers.flatten({inputShape}));
        newHead.add(tf.layers.dense({units: 200, activation: 'relu'}));
        // Five output units:
        //   - The first is a shape indictor: predicts whether the target
        //     shape is a triangle or a rectangle.
        //   - The remaining four units are for bounding-box prediction:
        //     [left, right, top, bottom] in the unit of pixels.
        newHead.add(tf.layers.dense({units: 5}));
        return newHead;
    }
    mapInputShapes(model, newShapeMap) {
        const cfg = {...model.getConfig()};
        cfg.layers = (cfg.layers).map(l => {
            if(l.name in newShapeMap) {
                return {...l, config: {
                    ...l.config,
                    batchInputShape: newShapeMap[l.name]
                }};
            } else {
                return l;
            }
        });

        const map = tf.serialization.SerializationMap.getMap().classNameMap;
        const [cls, fromConfig] = map[model.getClassName()];

        return fromConfig(cls, cfg);
    }
    /**
     * Builds object-detection model from MobileNet.
     *
     * @returns {[tf.Model, tf.layers.Layer[]]}
     *   1. The newly-built model for simple object detection.
     *   2. The layers that can be unfrozen during fine-tuning.
     */
    async buildObjectDetectionModel() {
        const {truncatedBase, fineTuningLayers} = await this.loadTruncatedBase();

        // Build the new head model.
        const newHead = this.buildNewHead(truncatedBase.outputs[0].shape.slice(1));
        const newOutput = newHead.apply(truncatedBase.outputs[0]);
        let model = tf.model({inputs: truncatedBase.inputs, outputs: newOutput});
        /*model = this.mapInputShapes(model, {
            'input_1':[
                60,
                240,
                320,
                3
            ],
            'dense_Dense1':25088
        })*/
        return {model, fineTuningLayers};
    }
}
module.exports = ChaosPixelTrainJob;





